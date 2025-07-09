#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agreement heat-maps (binary for thXX)
  thXX  – 1 if neighbour sets are identical, else 0    (self 포함)
  topK  – |intersection of top-K| / K                 (K ∈ {10,100,1000})

출력 PNG:
  analysis_plots/th85/ … th99/  top10/ top100/ top1000/
"""

import os, itertools, numpy as np, torch, matplotlib.pyplot as plt, seaborn as sns
from tqdm import tqdm

# ── config ────────────────────────────────────────────────────────────
DATASETS = ["cc_news_graph", "wikitext_graph_data"]
MODS = {"synonym_replacement": ["0.3", "0.6"],
        "context_replacement": ["0.3", "0.6"],
        "dipper_paraphraser":  ["60_0", "60_20"]}
MODELS = ["TransE", "RotatE", "DistMult", "ComplEx"]

CKPT_DIR = "kge_models"
OUT_DIR  = "analysis_plots"

THRESH_LIST = [0.85, 0.90, 0.95, 0.97, 0.98, 0.99]
TOP_LIST    = [10, 100, 1000]

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------------------------------------------------------------------

def load_norm(path: str) -> torch.Tensor:
    W = torch.load(path, map_location="cpu")["node_emb.weight"].float()
    return torch.nn.functional.normalize(W, dim=1).to(DEV)

@torch.no_grad()
def build_structures(E: torch.Tensor):
    n = E.size(0)
    S = E @ E.T                      # (n,n) diagonal 1.0

    neighs = {}
    for thr in THRESH_LIST:
        mask = (S >= thr)
        neighs[thr] = [mask[i].nonzero(as_tuple=True)[0].cpu().numpy()
                       for i in range(n)]

    tops = {}
    for K in TOP_LIST:
        k = min(K, n)
        tops[K] = torch.topk(S, k, dim=1).indices.cpu().numpy()
    return neighs, tops

# ── main ──────────────────────────────────────────────────────────────
for ds in DATASETS:
    for mod, opts in MODS.items():
        for opt in opts:
            print(f"\n▶ {ds}/{mod}/{opt}")
            neighs, tops, mdl_list = {}, {}, []

            for mdl in MODELS:
                ck = f"{CKPT_DIR}/{ds}_{mod}_{opt}_{mdl}.pt"
                if not os.path.exists(ck):
                    print("  skip", ck); continue
                nbh, tp = build_structures(load_norm(ck))
                neighs[mdl], tops[mdl] = nbh, tp
                mdl_list.append(mdl)

            if len(mdl_list) < 2:
                print("  <2 models, skip"); continue

            n, M = len(tops[mdl_list[0]][TOP_LIST[0]]), len(mdl_list)

            mats = {}
            for thr in THRESH_LIST:
                mats[f"th{int(thr*100):02d}"] = np.eye(M, dtype=np.float32)
            for K in TOP_LIST:
                mats[f"top{K}"] = np.eye(M, dtype=np.float32)

            # pairwise comparison
            for a, b in itertools.combinations(range(M), 2):
                ma, mb = mdl_list[a], mdl_list[b]
                agree_th = {t:0 for t in THRESH_LIST}
                agreeK   = {K:0.0 for K in TOP_LIST}

                for v in tqdm(range(n), desc=f"{ma} vs {mb}", leave=False):
                    # threshold: exact match
                    for t in THRESH_LIST:
                        A = neighs[ma][t][v];  B = neighs[mb][t][v]
                        if len(A)==len(B) and np.array_equal(np.sort(A), np.sort(B)):
                            agree_th[t] += 1

                    # top-K: proportion overlap (기존과 동일)
                    for K in TOP_LIST:
                        A = tops[ma][K][v];  B = tops[mb][K][v]
                        agreeK[K] += len(np.intersect1d(A, B, assume_unique=True)) / K

                # 채우기
                for t in THRESH_LIST:
                    tag = f"th{int(t*100):02d}"
                    val = agree_th[t] / n
                    mats[tag][a,b] = mats[tag][b,a] = val

                for K in TOP_LIST:
                    tag = f"top{K}"
                    val = agreeK[K] / n
                    mats[tag][a,b] = mats[tag][b,a] = val

            # save heat-maps
            for tag, mat in mats.items():
                path = os.path.join(OUT_DIR, tag);  os.makedirs(path, exist_ok=True)
                plt.figure(figsize=(1.2*M, 1.2*M))
                sns.heatmap(mat, annot=True, fmt=".3f",
                            xticklabels=mdl_list, yticklabels=mdl_list,
                            vmin=0, vmax=1, cmap="viridis")
                plt.title(f"{ds} – {mod}/{opt}  ({tag})")
                plt.tight_layout()
                fn = f"{ds}_{mod}_{opt}_{tag}.png".replace('/', '_')
                plt.savefig(os.path.join(path, fn), dpi=300)
                plt.close()
                print("  saved", tag, fn)

print("\nAll done →", OUT_DIR)
