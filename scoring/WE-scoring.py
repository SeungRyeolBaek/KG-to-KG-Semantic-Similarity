###############################################################################
# Embedding-based graph-level scoring pipeline (Word2Vec, FastText, GloVe)
# ––– KGE 파이프라인과 동일한 흐름 유지
# ––– 엔티티 임베딩 생성 단계만 바꿈 (학습: W2V, FastText, GloVe)
###############################################################################
import os
import json
import gc
import time
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm
from gensim.models import Word2Vec, FastText, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from scipy.spatial.distance import cosine

ent_emb_cache = {}

def build_cache(file_to_triples, ent_embeddings):
    global ent_emb_cache
    ent_emb_cache.clear()
    for fpath, triplets in file_to_triples.items():
        entity_set = {h for h, _, t in triplets}.union({t for _, _, t in triplets})
        if entity_set:
            ent_emb_arr = ent_embeddings[list(entity_set)]
        else:
            ent_emb_arr = np.zeros((0, ent_embeddings.shape[1]), dtype=np.float32)
        ent_emb_cache[fpath] = ent_emb_arr

def tokenise(sentence):
    return [tok.lower() for tok in sentence.split()]

def load_kgs_and_build_dicts(folder_paths):
    ent2id, rel2id = {}, {}
    all_triplets, file_to_triples = [], []
    ent_cnt = rel_cnt = 0
    for folder in folder_paths:
        for root, _, files in os.walk(folder):
            for fname in files:
                if not fname.endswith('.json'): continue
                fpath = os.path.join(root, fname)
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                tris = []
                for rel in data["relationships"]:
                    h, r, t = rel["source"]["id"], rel["type"], rel["target"]["id"]
                    for e in [h, t]:
                        if e not in ent2id:
                            ent2id[e] = ent_cnt; ent_cnt += 1
                    if r not in rel2id:
                        rel2id[r] = rel_cnt; rel_cnt += 1
                    tris.append((ent2id[h], rel2id[r], ent2id[t]))
                    all_triplets.append(tris[-1])
                file_to_triples.append((fpath, tris))
    return ent2id, rel2id, all_triplets, dict(file_to_triples)

def train_or_load_embedding(corpus_folders, backend='word2vec', vector_size=200, 
                            window=5, min_count=2, workers=4):
    name_tag = '*'.join(os.path.basename(p) for p in corpus_folders)
    model_path = f"embedding_{backend}_{name_tag}_{vector_size}.bin"
    if os.path.exists(model_path):
        print(f"[Load {backend}] {model_path}")
        if backend == 'glove':
            return KeyedVectors.load(model_path)
        elif backend == 'fasttext':
            return FastText.load(model_path)
        return Word2Vec.load(model_path)

    sentences = []
    for folder in corpus_folders:
        for root, _, files in os.walk(folder):
            for fname in files:
                if fname.endswith(".txt"):
                    with open(os.path.join(root, fname), encoding="utf-8") as f:
                        for line in f:
                            toks = tokenise(line)
                            if toks:
                                sentences.append(toks)
    print(f"[Train {backend}] #sentences = {len(sentences)}")

    if backend == 'word2vec':
        model = Word2Vec(sentences, vector_size=vector_size, window=window,
                         min_count=min_count, workers=workers, sg=1)
        model.save(model_path)
        return model

    elif backend == 'fasttext':
        model = FastText(sentences, vector_size=vector_size, window=window,
                         min_count=min_count, workers=workers, sg=1)
        model.save(model_path)
        return model

    elif backend == 'glove':
        # GloVe는 사전학습 glove.txt가 있다고 가정함
        glove_input = 'glove.6B.200d.txt'
        tmp_output = 'glove.6B.200d.word2vec.txt'
        if not os.path.exists(tmp_output):
            glove2word2vec(glove_input, tmp_output)
        kv = KeyedVectors.load_word2vec_format(tmp_output)
        kv.save(model_path)
        return kv

    raise ValueError(f"Unknown embedding backend: {backend}")

def build_entity_embeddings(ent2id, model, dim):
    kv = model.wv if hasattr(model, 'wv') else model
    emb = np.random.normal(scale=0.01, size=(len(ent2id), dim)).astype(np.float32)
    oov = 0
    for ent, idx in ent2id.items():
        toks = tokenise(ent.replace('_', ' '))
        vecs = [kv[t] for t in toks if t in kv]
        if vecs:
            emb[idx] = np.mean(vecs, axis=0)
        else:
            oov += 1
    print(f"[Embedding] OOV entities: {oov}/{len(ent2id)}")
    return emb

def compute_entity_avg_embedding(path, file2tri, emb):
    ids = {h for h, _, t in file2tri[path]}.union({t for _, _, t in file2tri[path]})
    return emb[list(ids)].mean(0) if ids else None

def compare_graphs_by_average(a, b, file2tri, emb):
    va = compute_entity_avg_embedding(a, file2tri, emb)
    vb = compute_entity_avg_embedding(b, file2tri, emb)
    return 0.0 if va is None or vb is None else 1 - cosine(va, vb)

def compare_graphs_by_pairs(a, b, emb, th=0.97):
    e1 = torch.tensor(ent_emb_cache[a], dtype=torch.float32)
    e2 = torch.tensor(ent_emb_cache[b], dtype=torch.float32)
    if e1.numel() == 0 or e2.numel() == 0: return 0.0
    jm = torch.cat([e1, e2], 0).mean(0, keepdim=True)
    e1, e2 = e1 - jm, e2 - jm
    e1, e2 = torch.nn.functional.normalize(e1, dim=1), torch.nn.functional.normalize(e2, dim=1)
    sim = torch.matmul(e1, e2.T)
    mx = sim.max(1).values if e1.shape[0] < e2.shape[0] else sim.max(0).values
    return torch.mean((mx > th).float()).item()

def compute_entity_weighted_avg_embedding(path, file2tri, emb):
    cnt = Counter(h for h, _, t in file2tri[path]) + Counter(t for _, _, t in file2tri[path])
    ids, ws = zip(*cnt.items()) if cnt else ([], [])
    return (emb[list(ids)].T * np.array(ws)).T.sum(0) / np.sum(ws) if ids else None

def compare_graphs_by_weighted(a, b, file2tri, emb):
    va = compute_entity_weighted_avg_embedding(a, file2tri, emb)
    vb = compute_entity_weighted_avg_embedding(b, file2tri, emb)
    return 0.0 if va is None or vb is None else 1 - cosine(va, vb)

def compute_graph_similarity(a, b, file2tri, emb, method='average'):
    if method == 'average': return compare_graphs_by_average(a, b, file2tri, emb)
    if method == 'pairs': return compare_graphs_by_pairs(a, b, emb)
    if method == 'weighted average': return compare_graphs_by_weighted(a, b, file2tri, emb)
    raise ValueError(f"Unknown method: {method}")

def evaluate_json_folders(orig_dir, mod_dir, out_dir, file2tri, emb, method):
    orig_files = [os.path.join(r, f) for r, _, fs in os.walk(orig_dir) for f in fs if f.endswith('.json')]
    mod_files = [os.path.join(r, f) for r, _, fs in os.walk(mod_dir) for f in fs if f.endswith('.json')]
    os.makedirs(out_dir, exist_ok=True)
    for mf in tqdm(mod_files, desc="Modified files"):
        out_path = os.path.join(out_dir, f"{os.path.basename(mf)}_results.json")
        if os.path.exists(out_path): continue
        res = [{"original": os.path.basename(of), "similarity_score": float(compute_graph_similarity(of, mf, file2tri, emb, method))} for of in orig_files]
        with open(out_path, 'w', encoding='utf-8') as fp:
            json.dump(res, fp, ensure_ascii=False, indent=4)

def emb_scoring(dataset, text_mod, mod_opt, backend='word2vec', method='average', dim=200):
    orig = os.path.join(dataset, 'original_graph')
    mod = os.path.join(dataset, text_mod, mod_opt)
    text_folder = {
        'cc_news_graph': 'cc_news_graph_verbalized',
        'wikitext_graph_data': 'wikitext_graph_data_verbalized'
    }[dataset]
    out = os.path.join('Result', dataset, text_mod, mod_opt, method, backend)

    ent2id, _, _, file2tri = load_kgs_and_build_dicts([orig, mod])
    model = train_or_load_embedding([text_folder], backend=backend, vector_size=dim)
    emb = build_entity_embeddings(ent2id, model, dim)
    build_cache(file2tri, emb)
    evaluate_json_folders(orig, mod, out, file2tri, emb, method)

if __name__ == '__main__':
    datasets = ['cc_news_graph', 'wikitext_graph_data']
    text_mods = ['synonym_replacement', 'context_replacement', 'dipper_paraphraser']
    mod_opt = {
        'synonym_replacement': ['0.3', '0.6'],
        'context_replacement': ['0.3', '0.6'],
        'dipper_paraphraser': ['60_0', '60_20']
    }
    methods = ['pairs', 'weighted average']
    backends = ['word2vec', 'fasttext', 'glove']
    backends = ['glove']

    for backend in backends:
        for ds in datasets:
            for tm in text_mods:
                for opt in mod_opt[tm]:
                    for m in methods:
                        print(f"\n=== {ds} | {tm} {opt} | {m} | {backend} ===")
                        emb_scoring(ds, tm, opt, backend=backend, method=m, dim=200)
                        gc.collect(); torch.cuda.empty_cache()
