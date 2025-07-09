###############################################################################
# Generic Embedding–based graph-level scoring pipeline (v3 – no training)
# ––– Word2Vec, GloVe, FastText 등 사전학습 임베딩만 사용
# ––– 학습 관련 함수(train_or_load_word2vec 등) 완전히 제거
###############################################################################
import os, json, gc, time
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm

import gensim.downloader as api
from scipy.spatial.distance import cosine

###############################################################################
# 전역 캐시 ────────────────────────────────────────────────────────────────────
###############################################################################
ent_emb_cache = {}

def build_cache(file_to_triples, ent_embeddings):
    global ent_emb_cache
    ent_emb_cache.clear()
    for fp, tris in file_to_triples.items():
        ents = {h for h, _, t in tris}.union({t for _, _, t in tris})
        arr = ent_embeddings[list(ents)] if ents else \
              np.zeros((0, ent_embeddings.shape[1]), dtype=np.float32)
        ent_emb_cache[fp] = arr

###############################################################################
# (0) Util ────────────────────────────────────────────────────────────────────
###############################################################################
def tokenise(s: str):
    return [t.lower() for t in s.split()]

###############################################################################
# (1) KG JSON 로드 & dict 빌드 ────────────────────────────────────────────────
############################################################################### 
def load_kgs_and_build_dicts(folders):
    ent2id, rel2id, all_tris, file2tri = {}, {}, [], []
    ei = ri = 0
    for fol in folders:
        for root, _, files in os.walk(fol):
            for fn in (f for f in files if f.endswith('.json')):
                fp = os.path.join(root, fn)
                with open(fp, encoding='utf-8') as f:
                    data = json.load(f)
                tris = []
                for rel in data["relationships"]:
                    h, r, t = rel["source"]["id"], rel["type"], rel["target"]["id"]
                    if h not in ent2id:
                        ent2id[h] = ei; ei += 1
                    if t not in ent2id:
                        ent2id[t] = ei; ei += 1
                    if r not in rel2id:
                        rel2id[r] = ri; ri += 1
                    hid, rid, tid = ent2id[h], rel2id[r], ent2id[t]
                    all_tris.append((hid, rid, tid))
                    tris.append((hid, rid, tid))
                file2tri.append((fp, tris))
    return ent2id, rel2id, all_tris, dict(file2tri)

###############################################################################
# (2) 사전학습 임베딩 로드 ──────────────────────────────────────────────────
###############################################################################
def load_pretrained_kv(backend: str):
    name = {
        'word2vec': 'word2vec-google-news-300',
        'glove':     'glove-wiki-gigaword-300',
        'fasttext':  'fasttext-wiki-news-subwords-300'
    }[backend]
    print(f"[Import] gensim-api.load('{name}') …")
    kv = api.load(name)
    print(f"[Import] dim = {kv.vector_size:,} | vocab = {len(kv):,}")
    return kv

###############################################################################
# (3) 엔티티 임베딩 생성 ─────────────────────────────────────────────────────
###############################################################################
def build_entity_embeddings(ent2id, kv, dim):
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

###############################################################################
# (4) 그래프 유사도 계산 ─────────────────────────────────────────────────────
###############################################################################
def compute_entity_avg_embedding(fp, file2tri, eemb):
    ents = {h for h, _, t in file2tri[fp]}.union({t for _, _, t in file2tri[fp]})
    if not ents:
        return None
    return eemb[list(ents)].mean(0)

def compare_graphs_by_average(a, b, file2tri, eemb):
    va = compute_entity_avg_embedding(a, file2tri, eemb)
    vb = compute_entity_avg_embedding(b, file2tri, eemb)
    if va is None or vb is None:
        return 0.0
    return 1 - cosine(va, vb)

def compare_graphs_by_pairs(a, b, eemb, th=0.7):
    e1 = torch.tensor(ent_emb_cache[a], dtype=torch.float32,
                      device='cuda' if torch.cuda.is_available() else 'cpu')
    e2 = torch.tensor(ent_emb_cache[b], dtype=torch.float32,
                      device='cuda' if torch.cuda.is_available() else 'cpu')
    if e1.numel() == 0 or e2.numel() == 0:
        return 0.0
    jm = torch.cat([e1, e2], 0).mean(0, keepdim=True)
    e1 = torch.nn.functional.normalize(e1 - jm, dim=1)
    e2 = torch.nn.functional.normalize(e2 - jm, dim=1)
    sim = torch.matmul(e1, e2.T)
    mx = sim.max(1).values if e1.shape[0] < e2.shape[0] else sim.max(0).values
    return torch.mean((mx > th).float()).item()

def compute_entity_weighted_avg_embedding(fp, file2tri, eemb):
    cnt = Counter()
    for h, _, t in file2tri[fp]:
        cnt[h] += 1; cnt[t] += 1
    if not cnt:
        return None
    ids, ws = zip(*cnt.items())
    vecs = eemb[list(ids)]
    return (vecs.T * np.array(ws)).T.sum(0) / np.sum(ws)

def compare_graphs_by_weighted(a, b, file2tri, eemb):
    va = compute_entity_weighted_avg_embedding(a, file2tri, eemb)
    vb = compute_entity_weighted_avg_embedding(b, file2tri, eemb)
    if va is None or vb is None:
        return 0.0
    return 1 - cosine(va, vb)

###############################################################################
# (5) 비교 wrapper ───────────────────────────────────────────────────────────
###############################################################################
_elapsed = 0.0

def compute_graph_similarity(a, b, file2tri, eemb, method='average'):
    global _elapsed
    st = time.perf_counter()
    if method == 'average':
        s = compare_graphs_by_average(a, b, file2tri, eemb)
    elif method == 'pairs':
        s = compare_graphs_by_pairs(a, b, eemb, th=0.97)
    elif method == 'weighted average':
        s = compare_graphs_by_weighted(a, b, file2tri, eemb)
    else:
        raise ValueError(f"Unknown method {method}")
    _elapsed += time.perf_counter() - st
    return s

###############################################################################
# (6) 폴더 전체 비교 + 결과 저장 ─────────────────────────────────────────────
###############################################################################
def evaluate_json_folders(orig_dir, mod_dir, out_dir,
                          file2tri, eemb, method='average'):
    orig_files = [
        os.path.join(r, f)
        for r, _, fs in os.walk(orig_dir)
        for f in fs if f.endswith('.json')
    ]
    mod_files = [
        os.path.join(r, f)
        for r, _, fs in os.walk(mod_dir)
        for f in fs if f.endswith('.json')
    ]
    os.makedirs(out_dir, exist_ok=True)
    print(f"[Run] output → {out_dir}")
    for mf in tqdm(mod_files, desc="Modified files"):
        out_path = os.path.join(out_dir, f"{os.path.basename(mf)}_results.json")
        if os.path.exists(out_path):
            continue
        res = []
        for of in orig_files:
            sim = compute_graph_similarity(of, mf, file2tri, eemb, method)
            res.append({"original": os.path.basename(of),
                        "similarity_score": float(sim)})
        with open(out_path, 'w', encoding='utf-8') as fp:
            json.dump(res, fp, ensure_ascii=False, indent=4)
    print(f"⏱ total elapsed: {_elapsed:.2f}s\n✓ All comparisons done → {out_dir}")

###############################################################################
# (7) 파이프라인 진입점 ──────────────────────────────────────────────────────
###############################################################################
def emb_scoring(dataset, text_mod, mod_option,
                kv, method='average',backend='word2vec'):
    orig_dir = os.path.join(dataset, 'original_graph')
    mod_dir = os.path.join(dataset, text_mod, mod_option)
    out_dir = os.path.join('Result', dataset, backend, text_mod, mod_option, method)

    ent2id, _, _, file2tri = load_kgs_and_build_dicts([orig_dir, mod_dir])
    eemb = build_entity_embeddings(ent2id, kv, kv.vector_size)
    build_cache(file2tri, eemb)
    evaluate_json_folders(orig_dir, mod_dir, out_dir, file2tri, eemb, method)

###############################################################################
# (8) main 루프 ──────────────────────────────────────────────────────────────
###############################################################################
if __name__ == "__main__":
    embed_backends = ['word2vec', 'glove', 'fasttext']

    datasets = ['cc_news_graph', 'wikitext_graph_data']
    text_mods = ['synonym_replacement', 'context_replacement', 'dipper_paraphraser']
    mod_opt = {
        'synonym_replacement': ['0.3', '0.6'],
        'context_replacement': ['0.3', '0.6'],
        'dipper_paraphraser': ['60_0', '60_20']
    }
    methods = ['pairs', 'weighted average']

    for backend in embed_backends:
        kv = load_pretrained_kv(backend)
        for ds in datasets:
            for tm in text_mods:
                for opt in mod_opt[tm]:
                    for m in methods:
                        print(f"\n=== {ds} | {tm} {opt} | {m} | {backend} ===")
                        emb_scoring(ds, tm, opt, kv=kv, method=m, backend=backend)
                        gc.collect(); torch.cuda.empty_cache()
