import os, json, random, time, gc
from collections import defaultdict
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.nn.kge import TransE, RotatE, DistMult, ComplEx

# ───── aggregation (학습 X) ────────────────────────────────────────────────
from torch_geometric.nn.aggr import (  # PyG ≥2.3
    MeanAggregation, MaxAggregation,
    MedianAggregation, VarAggregation, StdAggregation,
    MultiAggregation     
)

from scipy.spatial.distance import cosine

# ──────────────────────── 0. 전역 캐시 ──────────────────────────────────────
#   {agg_name : {file_path : graph_vector}}
graph_repr_cache: dict[str, dict[str, np.ndarray]] = defaultdict(dict)

# ──────────────────────── 1. KG 로딩 ───────────────────────────────────────
def load_kgs_and_build_dicts(folders):
    ent2id, rel2id, all_triplets, file2triples = {}, {}, [], {}
    eid = rid = 0
    for folder in folders:
        for root, _, files in os.walk(folder):
            for fn in files:
                if not fn.endswith(".json"):
                    continue
                path = os.path.join(root, fn)
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                triples = []
                for rel in data["relationships"]:
                    h, r, t = rel["source"]["id"], rel["type"], rel["target"]["id"]
                    if h not in ent2id:
                        ent2id[h] = eid; eid += 1
                    if t not in ent2id:
                        ent2id[t] = eid; eid += 1
                    if r not in rel2id:
                        rel2id[r] = rid; rid += 1
                    h_id, r_id, t_id = ent2id[h], rel2id[r], ent2id[t]
                    all_triplets.append((h_id, r_id, t_id))
                    triples.append((h_id, r_id, t_id))
                file2triples[path] = triples
    return ent2id, rel2id, all_triplets, file2triples

# ──────────────────────── 2. 데이터셋 & neg‑sampling ──────────────────────
class KGDataset(torch.utils.data.Dataset):
    def __init__(self, triples): self.triples = triples
    def __len__(self): return len(self.triples)
    def __getitem__(self, idx): return self.triples[idx]

def negative_sampling(batch, num_entities):
    out = []
    for h, r, t in batch:
        if random.random() < .5:
            out.append([random.randrange(num_entities), r, t])
        else:
            out.append([h, r, random.randrange(num_entities)])
    return torch.tensor(out, dtype=torch.long)

# ──────────────────────── 3. KGE 학습/로드 ────────────────────────────────
def _train(model_cls, triplets, n_ent, n_rel,
           dim=32, margin=1.0, lr=1e-2, epochs=1000, bs=128, patience=10):
    model = model_cls(n_ent, n_rel, hidden_channels=dim, margin=margin)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    loader = DataLoader(KGDataset(triplets), bs, shuffle=True,
                        collate_fn=lambda b: torch.tensor(b, dtype=torch.long))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = 9e9; wait = 0
    for ep in range(epochs):
        tot = 0.
        for batch in loader:
            batch, n_batch = batch.to(dev), negative_sampling(batch, n_ent).to(dev)
            opt.zero_grad()
            pos = model(batch[:,0], batch[:,1], batch[:,2])
            neg = model(n_batch[:,0], n_batch[:,1], n_batch[:,2])
            loss = torch.relu(margin + pos - neg).mean()
            loss.backward(); opt.step(); tot += loss.item()
        avg = tot/len(loader)
        print(f"{model_cls.__name__} Epoch {ep+1} Loss={avg:.4f}")
        if avg < best: best, wait = avg, 0
        else:
            wait += 1
            if wait >= patience: break
    return model.cpu()

def get_kge(model_name, triplets, n_ent, n_rel, save_path):
    cls = {"TransE":TransE, "RotatE":RotatE,
           "DistMult":DistMult, "ComplEx":ComplEx}[model_name]
    if os.path.exists(save_path):
        m = cls(n_ent, n_rel, hidden_channels=32, margin=1.0)
        m.load_state_dict(torch.load(save_path, map_location="cpu"))
        return m
    m = _train(cls, triplets, n_ent, n_rel)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(m.state_dict(), save_path)
    return m

# ──────────────────────── 4. Aggregation → 그래프 벡터 ─────────────────────
#   PyG built‑in aggregators
AGGRS = {
    "mean"   : MeanAggregation(),          # 평균
    "max"    : MaxAggregation(),           # 채널별 최대
    "median" : MedianAggregation(),        # 채널별 중앙값
    "var"    : VarAggregation(),           # 분산
    "std"    : StdAggregation(),           # 표준편차
    "mean_max_var" : MultiAggregation(
        [MeanAggregation(), MaxAggregation(), VarAggregation()],
        mode="cat"             # [d] ⟶ [3d]
    ),
}

def graph_vector(file_path:str, aggr_name:str,
                 file2triples, ent_emb)->np.ndarray|None:
    cache = graph_repr_cache[aggr_name]
    if file_path in cache:
        return cache[file_path]

    node_ids = {h for h,_,t in file2triples[file_path] for h in (h,t)}
    if not node_ids:
        cache[file_path] = None
        return None
    x = torch.tensor(ent_emb[list(node_ids)], dtype=torch.float)         # [N,d]
    batch = torch.zeros(x.size(0), dtype=torch.long)                     # single graph
    vec = AGGRS[aggr_name](x, batch).squeeze(0).numpy()                  # [d] or [2d]
    cache[file_path] = vec
    return vec

def cosine_sim(a,b): return 1 - cosine(a,b)

def compare_graphs(fA, fB, aggr, file2triples, ent_emb):
    vA = graph_vector(fA, aggr, file2triples, ent_emb)
    vB = graph_vector(fB, aggr, file2triples, ent_emb)
    if vA is None or vB is None: return 0.0
    return cosine_sim(vA, vB)

# ──────────────────────── 5. 폴더 전수 비교 ────────────────────────────────
def evaluate_folders(orig_dir, mod_dir, out_dir,
                     file2triples, ent_emb, aggr):
    orig_files = [os.path.join(r,f) for r,_,fs in os.walk(orig_dir) for f in fs if f.endswith(".json")]
    mod_files  = [os.path.join(r,f) for r,_,fs in os.walk(mod_dir)  for f in fs if f.endswith(".json")]
    os.makedirs(out_dir, exist_ok=True)
    for mf in tqdm(mod_files, desc=f"[{aggr}]"):
        out_path = os.path.join(out_dir, os.path.basename(mf)+"_results.json")
        if os.path.exists(out_path): continue
        res=[]
        for of in orig_files:
            s = compare_graphs(of, mf, aggr, file2triples, ent_emb)
            res.append({"original":os.path.basename(of), "similarity_score":float(s)})
        with open(out_path,"w",encoding="utf-8") as f: json.dump(res,f,indent=2,ensure_ascii=False)

# ──────────────────────── 6. 파이프라인 ────────────────────────────────────
def run(dataset, txt_mod, option, model_nm, aggr):
    global graph_repr_cache
    graph_repr_cache.clear(); graph_repr_cache.update({k:{} for k in AGGRS})
    base_out = {"c4_subset_graph":"C4_result",
                "wikitext_graph_data":"Wikitext_result"}.get(dataset,"results")

    orig_dir = os.path.join(dataset,"original_graph")
    mod_dir  = os.path.join(dataset,txt_mod,option)
    out_dir  = os.path.join(base_out,txt_mod,option,"kge",model_nm,aggr)

    ent2id, rel2id, trips, f2t = load_kgs_and_build_dicts([orig_dir,mod_dir])

    model_path = os.path.join("kge_models",f"{dataset}_{txt_mod}_{option}_{model_nm}.pt")
    kge = get_kge(model_nm, trips, len(ent2id), len(rel2id), model_path)
    ent_emb = kge.node_emb.weight.detach().cpu().numpy()

    evaluate_folders(orig_dir, mod_dir, out_dir, f2t, ent_emb, aggr)

# ──────────────────────── 7. 실행 예시 ──────────────────────────────────────
if __name__ == "__main__":
    datasets = ["c4_subset_graph","wikitext_graph_data"]
    txt_mods = ["synonym_replacement","context_replacement","dipper_paraphraser"]
    options  = {
        "synonym_replacement":["0.3","0.6"],
        "context_replacement":["0.3","0.6"],
        "dipper_paraphraser":["60_0","60_20"]
    }
    aggregations = ["mean", "max", "median", "var", "std"]
    model_nm = "DistMult"

    for ds in datasets:
        for tm in txt_mods:
            for opt in options[tm]:
                # for aggr in aggregations:
                aggr = "mean_max_var"
                print(f"▶ {ds}/{tm}/{opt} ‑ {aggr}")
                run(ds, tm, opt, model_nm, aggr)
            gc.collect(); torch.cuda.empty_cache()
