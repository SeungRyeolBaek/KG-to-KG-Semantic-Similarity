#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pair-level hit-ratio(%) — per modified KG
----------------------------------------
modified KG 하나를 모든 original KG 와 비교해

    max_sim = max cosine(pairwise)      # compare_graphs_by_pairs_counting 과 동일
    hit(thr) = (max_sim ≥ thr) and (ID 다른 쌍)

이면 그 modified KG 는 해당 threshold 에서 hit.
threshold 별 hit-ratio = (#hit modified) / (#modified).

출력 PNG:
  analysis_pairs_pct/<dataset>/<mod>/<opt>/<model>_ratio.png
"""

import os, json, gc
import numpy as np
import torch
import matplotlib.pyplot as plt, seaborn as sns
from tqdm import tqdm
from torch_geometric.nn.kge import TransE, RotatE, DistMult, ComplEx  # ckpt 로드용
import multiprocessing
from functools import partial

# ────────── 실험 설정 ────────────────────────────────────────────────
DATASETS = ["cc_news_graph", "wikitext_graph_data"]
MODS = {"synonym_replacement": ["0.3", "0.6"],
        "context_replacement": ["0.3", "0.6"],
        "dipper_paraphraser":  ["60_0", "60_20"]}
MODELS = ["TransE", "RotatE", "DistMult", "ComplEx"]

CKPT_DIR = "kge_models"
PLOT_DIR = "analysis_pairs_pct"
THR_LIST = [0.85, 0.90, 0.95, 0.97, 0.98, 0.99]

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

# ════════════════════════════════════════════════════════════════════
# 0. pair-similarity(max-cos)  ── 원본과 동일 로직
# ════════════════════════════════════════════════════════════════════
@torch.no_grad()
def pairs_max_sim(ea: torch.Tensor, eb: torch.Tensor) -> torch.Tensor:
    joint_mean = torch.cat([ea, eb], dim=0).mean(0, keepdim=True)
    ea = torch.nn.functional.normalize(ea - joint_mean, dim=1)
    eb = torch.nn.functional.normalize(eb - joint_mean, dim=1)
    sim = ea @ eb.T
    return sim.max(dim=1)[0] if ea.size(0) <= eb.size(0) else sim.max(dim=0)[0]

# ════════════════════════════════════════════════════════════════════
# 1. KG 로드 & ID 매핑
# ════════════════════════════════════════════════════════════════════

def load_kgs(folders):
    ent2id, file2trip = {}, {}
    eid = 0
    for folder in folders:
        for root, _, files in os.walk(folder):
            for fn in files:
                if not fn.endswith(".json"): continue
                path = os.path.join(root, fn)
                with open(path, encoding="utf-8") as f:
                    js = json.load(f)
                triples = []
                for rel in js.get("relationships", []):
                    h, t = rel["source"]["id"], rel["target"]["id"]
                    for x in (h, t):
                        if x not in ent2id: ent2id[x] = eid; eid += 1
                    triples.append((ent2id[h], ent2id[t]))
                file2trip[path] = triples
    return file2trip, len(ent2id)

# ════════════════════════════════════════════════════════════════════
# 2.  파일별 (id 배열, embedding 배열) 캐시
# ════════════════════════════════════════════════════════════════════
def build_cache(file2trip, ent_emb):
    cache = {}
    for f, triples in file2trip.items():
        ids = sorted({h for h,_ in triples}.union({t for _,t in triples}))
        vec = ent_emb[ids] if ids else np.zeros((0, ent_emb.shape[1]))
        cache[f] = (np.asarray(ids, np.int32), vec)
    return cache

# ════════════════════════════════════════════════════════════════════
# 3.  modified-original 쌍에서 hit 여부
# ════════════════════════════════════════════════════════════════════
@torch.no_grad()
def is_hit(orig_path, mod_path, cache, THR_LIST, hits):
    ids_a, EA = cache[orig_path]
    ids_b, EB = cache[mod_path]
    if EA.size == 0 or EB.size == 0:
        return False

    ea = torch.tensor(EA, device=DEV, dtype=torch.float32)
    eb = torch.tensor(EB, device=DEV, dtype=torch.float32)
    max_sim = pairs_max_sim(ea, eb)                     # (k,)

    if ea.size(0) <= eb.size(0):
        partner = torch.tensor(ids_b, device=DEV)[
            (ea @ eb.T).argmax(dim=1)]
        diff = torch.tensor(ids_a, device=DEV) != partner
    else:
        partner = torch.tensor(ids_a, device=DEV)[
            (ea @ eb.T).argmax(dim=0)]
        diff = torch.tensor(ids_b, device=DEV) != partner
    for thr in THR_LIST: 
        if bool(((max_sim >= thr) & diff).any().item()):
            hits[thr] += 1
    return return

def check_hit_parallel(orig_file, mod_file, cache, thr):
    return is_hit(orig_file, mod_file, cache, thr)

# ════════════════════════════════════════════════════════════════════
# 4. bar-plot helper
# ════════════════════════════════════════════════════════════════════
def make_plot(ratios, title, path):
    sns.set_style("whitegrid")
    xs = [f"{int(t*100)}%" for t in THR_LIST]
    ys = [ratios[t] for t in THR_LIST]
    plt.figure(figsize=(6,4))
    ax = sns.barplot(x=xs, y=ys, palette="viridis")
    ax.set_ylim(0,100); ax.set_ylabel("hit ratio (%)"); ax.set_xlabel("threshold")
    ax.set_title(title)
    for p,v in zip(ax.patches,ys):
        ax.annotate(f"{v:4.1f}", (p.get_x()+p.get_width()/2., v),
                    ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(path, dpi=300); plt.close()

# ════════════════════════════════════════════════════════════════════
# 5.  메인 루프
# ════════════════════════════════════════════════════════════════════
os.makedirs(PLOT_DIR, exist_ok=True)

for ds in DATASETS:
    for mod, opts in MODS.items():
        for opt in opts:
            print(f"\n▶ {ds} / {mod}/{opt}")
            o_dir = os.path.join(ds, "original_graph")
            m_dir = os.path.join(ds, mod, opt)

            orig_files = [os.path.join(r,f)
                          for r,_,fs in os.walk(o_dir) for f in fs if f.endswith(".json")]
            mod_files  = [os.path.join(r,f)
                          for r,_,fs in os.walk(m_dir) for f in fs if f.endswith(".json")]
            if not orig_files or not mod_files:
                print("   ⚠  그래프 파일 부족 → skip");  continue

            file2trip, num_ent = load_kgs([o_dir, m_dir])

            for mdl in MODELS:
                ck = f"{CKPT_DIR}/{ds}_{mod}_{opt}_{mdl}.pt"
                if not os.path.exists(ck):
                    print(f"   ⏩  {mdl:8s} checkpoint 없음"); continue

                print(f"   ▶ model {mdl}")
                ent_emb = torch.load(ck, map_location="cpu")["node_emb.weight"].cpu().numpy()
                cache = build_cache(file2trip, ent_emb)

                ratio = {}               # threshold → hit-ratio
                hits = {}

                for mf in tqdm(mod_files, desc=f"      thr={thr:.2f}", leave=False):
                for thr in THR_LIST:  
                    hit = 0
                    for mf in tqdm(mod_files, desc=f"      thr={thr:.2f}", leave=False):
                        for of in orig_files:
                            if is_hit(of, mf, cache, THR_LIST, hits):
                                hit += 1
                                break          # 다음 modified KG
                    print(f"hit: {hit}")
                    ratio[thr] = hit / len(mod_files) * 100.0

                out_dir = os.path.join(PLOT_DIR, ds, mod, opt)
                os.makedirs(out_dir, exist_ok=True)
                png = os.path.join(out_dir, f"{mdl}_ratio.png")
                make_plot(ratio, f"{mdl} – modified-KG hit ratio", png)
                print(f"      ✓ saved {png}")

                del cache, ent_emb
                torch.cuda.empty_cache(); gc.collect()

print("\n✅  Finished — plots are in  analysis_pairs_pct/  directory.")
