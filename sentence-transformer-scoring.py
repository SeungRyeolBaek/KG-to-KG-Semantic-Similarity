###############################################################################
# Sentence-Transformer-based graph-level scoring pipeline
# ––– KGE · Word2Vec 파이프라인과 **동일한 흐름**을 유지하되
#     “엔티티 임베딩 생성” 단계만 SentenceTransformer(all-MiniLM-L6-v2)로 교체
###############################################################################
import os
import json
import gc
import time
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

###############################################################################
# (0) SentenceTransformer 로드 (전역) ───────────────────────────────────────────
###############################################################################
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
EMB_DIM = ST_MODEL.get_sentence_embedding_dimension()  # 384

###############################################################################
# 전역 캐시 (파일 경로 → numpy 배열) ────────────────────────────────────────────
###############################################################################
ent_emb_cache = {}

def build_cache(file_to_triples, ent_embeddings):
    """
    모든 파일에 대해 엔티티 임베딩을 미리 인덱싱해 ent_emb_cache 저장.
    """
    global ent_emb_cache
    ent_emb_cache.clear()

    for fpath, triplets in file_to_triples.items():
        entity_set = {h for h, _, _ in triplets} | {t for _, _, t in triplets}
        if entity_set:
            ent_emb_cache[fpath] = ent_embeddings[list(entity_set)]
        else:
            ent_emb_cache[fpath] = np.zeros((0, ent_embeddings.shape[1]),
                                            dtype=np.float32)

###############################################################################
# (1) 여러 폴더에서 JSON 읽어 (h, r, t) + 전역 ent2id, rel2id 생성
###############################################################################
def load_kgs_and_build_dicts(folder_paths):
    """
    folder_paths: KG JSON 파일이 들어있는 여러 폴더 리스트
    반환:
      - ent2id, rel2id             : 문자열 → 정수 ID
      - all_triplets               : 전체 (h, r, t)
      - file_to_triples (dict)     : {파일경로: [(h, r, t), …]}
    """
    ent2id, rel2id = {}, {}
    all_triplets, file_to_triples = [], []

    ent_cnt = rel_cnt = 0
    for folder in folder_paths:
        for root, _, files in os.walk(folder):
            for fname in files:
                if not fname.endswith(".json"):
                    continue
                fpath = os.path.join(root, fname)
                with open(fpath, encoding="utf-8") as fp:
                    data = json.load(fp)

                triples_here = []
                for rel_obj in data["relationships"]:
                    h = rel_obj["source"]["id"]
                    r = rel_obj["type"]
                    t = rel_obj["target"]["id"]

                    if h not in ent2id:
                        ent2id[h] = ent_cnt; ent_cnt += 1
                    if t not in ent2id:
                        ent2id[t] = ent_cnt; ent_cnt += 1
                    if r not in rel2id:
                        rel2id[r] = rel_cnt; rel_cnt += 1

                    h_id, r_id, t_id = ent2id[h], rel2id[r], ent2id[t]
                    all_triplets.append((h_id, r_id, t_id))
                    triples_here.append((h_id, r_id, t_id))

                file_to_triples.append((fpath, triples_here))

    return ent2id, rel2id, all_triplets, dict(file_to_triples)

###############################################################################
# (2) 엔티티 임베딩 (SentenceTransformer) ──────────────────────────────────────
###############################################################################
def build_entity_embeddings(ent2id):
    """
    엔티티 문자열을 SentenceTransformer 로 인코딩.
    '_' → ' ' 변환 후 encode; 배치 인코딩으로 속도 최적화.
    """
    num_ent = len(ent2id)
    texts = [e.replace("_", " ") for e in ent2id.keys()]
    # encode → numpy (shape=(N, EMB_DIM))
    embeds = ST_MODEL.encode(texts,
                             batch_size=128,
                             convert_to_numpy=True,
                             show_progress_bar=True)

    ent_emb = np.zeros((num_ent, EMB_DIM), dtype=np.float32)
    for ent, idx in ent2id.items():
        ent_emb[idx] = embeds[idx]
    return ent_emb

###############################################################################
# (3) 그래프 유사도: average / pairs / weighted average ──────────────────────
###############################################################################
def _avg_emb(file_path, file_to_triples, ent_emb):
    ids = {h for h, _, _ in file_to_triples[file_path]} | \
          {t for _, _, t in file_to_triples[file_path]}
    if not ids:
        return None
    return ent_emb[list(ids)].mean(axis=0)

def compare_avg(fA, fB, file_to_triples, ent_emb):
    a = _avg_emb(fA, file_to_triples, ent_emb)
    b = _avg_emb(fB, file_to_triples, ent_emb)
    return 0.0 if a is None or b is None else 1 - cosine(a, b)

def compare_pairs(fA, fB, threshold=0.7):
    global ent_emb_cache
    dev = torch.device(DEVICE)
    e1 = torch.tensor(ent_emb_cache[fA], dtype=torch.float32, device=dev)
    e2 = torch.tensor(ent_emb_cache[fB], dtype=torch.float32, device=dev)
    if e1.numel() == 0 or e2.numel() == 0:
        return 0.0
    joint_mean = torch.cat([e1, e2]).mean(0, keepdim=True)
    e1 = torch.nn.functional.normalize(e1 - joint_mean, dim=1)
    e2 = torch.nn.functional.normalize(e2 - joint_mean, dim=1)
    sims = torch.matmul(e1, e2.T)
    if e1.shape[0] < e2.shape[0]:
        max_sims, _ = sims.max(dim=1)
    else:
        max_sims, _ = sims.max(dim=0)
    return torch.mean((max_sims > threshold).float()).item()

def _weighted_emb(file_path, file_to_triples, ent_emb):
    trips = file_to_triples[file_path]
    if not trips:
        return None
    freq = Counter()
    for h, _, t in trips:
        freq[h] += 1; freq[t] += 1
    ids = list(freq.keys())
    w   = np.array([freq[i] for i in ids])
    e   = ent_emb[ids]
    return (e.T * w).T.sum(axis=0) / w.sum()

def compare_wavg(fA, fB, file_to_triples, ent_emb):
    a = _weighted_emb(fA, file_to_triples, ent_emb)
    b = _weighted_emb(fB, file_to_triples, ent_emb)
    return 0.0 if a is None or b is None else 1 - cosine(a, b)

###############################################################################
# (4) 그래프 쌍 비교 wrapper ───────────────────────────────────────────────────
###############################################################################
_elapsed = 0.0
def graph_similarity(fA, fB, file_to_triples, ent_emb, method):
    global _elapsed
    start = time.perf_counter()
    if method == "average":
        s = compare_avg(fA, fB, file_to_triples, ent_emb)
    elif method == "pairs":
        s = compare_pairs(fA, fB, threshold=0.97)
    elif method == "weighted average":
        s = compare_wavg(fA, fB, file_to_triples, ent_emb)
    else:
        raise ValueError(method)
    _elapsed += time.perf_counter() - start
    return s

###############################################################################
# (5) 폴더 전체 비교 + 결과 저장 ───────────────────────────────────────────────
###############################################################################
def evaluate_json_folders(orig_folder, mod_folder, out_folder,
                          file_to_triples, ent_emb, method):
    o_files = [os.path.join(r, f)
               for r, _, fs in os.walk(orig_folder)
               for f in fs if f.endswith(".json")]
    m_files = [os.path.join(r, f)
               for r, _, fs in os.walk(mod_folder)
               for f in fs if f.endswith(".json")]

    os.makedirs(out_folder, exist_ok=True)
    print(f"[Run] {method} → {out_folder}")

    for mf in tqdm(m_files, desc="modified"):
        out_fp = os.path.join(out_folder,
                              f"{os.path.basename(mf)}_results.json")
        if os.path.exists(out_fp):
            continue
        res = []
        for of in o_files:
            sim = graph_similarity(of, mf, file_to_triples, ent_emb, method)
            res.append({"original": os.path.basename(of),
                        "similarity_score": float(sim)})
        with open(out_fp, "w", encoding="utf-8") as fw:
            json.dump(res, fw, ensure_ascii=False, indent=4)

    print(f"⏱ {_elapsed:.2f}s elapsed   ✓ done → {out_folder}")

###############################################################################
# (6) Sentence-Transformer 파이프라인 진입점 ───────────────────────────────────
###############################################################################
def st_scoring(dataset, text_mod, mod_option, method="average"):
    # ─── 경로
    orig_folder = os.path.join(dataset, "original_graph")
    mod_folder  = os.path.join(dataset, text_mod, mod_option)
    out_folder  = os.path.join("Result_st", dataset, text_mod, mod_option, method)

    # ─── JSON → dicts
    ent2id, _, _, file_to_triples = load_kgs_and_build_dicts(
        [orig_folder, mod_folder])

    # ─── 엔티티 임베딩
    ent_emb = build_entity_embeddings(ent2id)

    # ─── 캐시
    build_cache(file_to_triples, ent_emb)

    # ─── 비교
    evaluate_json_folders(orig_folder, mod_folder, out_folder,
                          file_to_triples, ent_emb, method)

###############################################################################
# (7) main 루프 ────────────────────────────────────────────────────────────────
###############################################################################
if __name__ == "__main__":
    DATASETS = ["c4_subset_graph", "wikitext_graph_data"]
    TEXT_MODS = ["synonym_replacement",
                 "context_replacement",
                 "dipper_paraphraser"]
    MOD_OPTIONS = {
        "synonym_replacement": ["0.3", "0.6"],
        "context_replacement": ["0.3", "0.6"],
        "dipper_paraphraser":  ["60_0", "60_20"]
    }
    METHODS = ["pairs", "average", "weighted average"]
    METHODS = ["pairs", "weighted average"]
    for ds in DATASETS:
        for tm in TEXT_MODS:
            for opt in MOD_OPTIONS[tm]:
                for mtd in METHODS:
                    print(f"\n=== {ds} | {tm} {opt} | {mtd} ===")
                    st_scoring(ds, tm, opt, method=mtd)
                    gc.collect(); torch.cuda.empty_cache()
