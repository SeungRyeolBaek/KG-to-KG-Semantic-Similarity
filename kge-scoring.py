import os
import json
import time
import gc
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from scipy.spatial.distance import cosine

# PyG KGE models (PyG >= 2.3)
from torch_geometric.nn.kge import TransE, RotatE, DistMult, ComplEx

###############################################################################
# 0. DEVICE & GLOBAL CACHES
###############################################################################
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 파일별: 엔티티 / 릴레이션 임베딩 서브매트릭스
ent_emb_cache: Dict[str, np.ndarray] = {}
rel_emb_cache: Dict[str, np.ndarray] = {}
# 전역 전체 임베딩 (CUDA-side 계산용)
GLOBAL_ENT: torch.Tensor | None = None
GLOBAL_REL: torch.Tensor | None = None

###############################################################################
# 1. KG 읽어 매핑 생성
###############################################################################

def load_kgs_and_build_dicts(folders: List[str]):
    """여러 폴더 → ent2id, rel2id, all_triplets, file_to_triples"""
    ent2id, rel2id = {}, {}
    all_triplets: List[Tuple[int, int, int]] = []
    file_to_triples: Dict[str, List[Tuple[int, int, int]]] = {}

    ent_cnt = rel_cnt = 0
    for folder in folders:
        for root, _, files in os.walk(folder):
            for fn in files:
                if not fn.endswith('.json'):
                    continue
                fp = os.path.join(root, fn)
                with open(fp, 'r', encoding='utf-8') as fr:
                    data = json.load(fr)

                triples_here = []
                for rel_obj in data["relationships"]:
                    h, r, t = rel_obj["source"]["id"], rel_obj["type"], rel_obj["target"]["id"]
                    if h not in ent2id:
                        ent2id[h] = ent_cnt; ent_cnt += 1
                    if t not in ent2id:
                        ent2id[t] = ent_cnt; ent_cnt += 1
                    if r not in rel2id:
                        rel2id[r] = rel_cnt; rel_cnt += 1
                    h_id, r_id, t_id = ent2id[h], rel2id[r], ent2id[t]
                    all_triplets.append((h_id, r_id, t_id))
                    triples_here.append((h_id, r_id, t_id))
                file_to_triples[fp] = triples_here
    return ent2id, rel2id, all_triplets, file_to_triples

###############################################################################
# 2. Dataset & Neg‑Sampling
###############################################################################

class TripletDataset(Dataset):
    def __init__(self, triples: List[Tuple[int,int,int]]):
        self.triples = triples
    def __len__(self):
        return len(self.triples)
    def __getitem__(self, idx):
        return self.triples[idx]

def neg_sample(batch: torch.Tensor, n_ent: int) -> torch.Tensor:
    B = batch.size(0)
    corrupt_head = torch.rand(B, device=DEVICE) < 0.5
    rand_ent = torch.randint(0, n_ent, (B,), device=DEVICE)
    corrupt = batch.clone()
    corrupt[:,0] = torch.where(corrupt_head, rand_ent, corrupt[:,0])
    corrupt[:,2] = torch.where(~corrupt_head, rand_ent, corrupt[:,2])
    return corrupt

###############################################################################
# 3. KGE Model Factory & Trainer (generic)
###############################################################################

def build_model(name: str, n_ent: int, n_rel: int, dim: int=32, margin: float=2.0):
    if name == 'TransE':
        return TransE(n_ent, n_rel, dim, margin=margin, p_norm=1)
    if name == 'RotatE':
        return RotatE(n_ent, n_rel, dim, margin=margin)
    if name == 'DistMult':
        return DistMult(n_ent, n_rel, dim, margin=margin)
    if name == 'ComplEx':
        return ComplEx(n_ent, n_rel, dim)
    raise ValueError(name)

def train_kge(model_name: str, triples, n_ent, n_rel, lr=1e-2, dim=32, margin=2.0,
              epochs=1000, bs=128, patience_lim=10):
    model = build_model(model_name, n_ent, n_rel, dim, margin).to(DEVICE)
    loader = DataLoader(TripletDataset(triples), batch_size=bs, shuffle=True,
                        collate_fn=lambda b: torch.tensor(b, dtype=torch.long), num_workers=4)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best, patience = float('inf'), 0
    for ep in range(epochs):
        total = 0.0
        model.train()
        for batch in loader:
            batch = batch.to(DEVICE)
            neg = neg_sample(batch, n_ent)
            opt.zero_grad()
            pos = model(batch[:,0], batch[:,1], batch[:,2])
            negs= model(neg[:,0], neg[:,1], neg[:,2])
            loss = torch.relu(margin + pos - negs).mean()
            loss.backward(); opt.step()
            total += loss.item()
        avg = total/len(loader)
        print(f"Epoch {ep+1}| loss {avg:.4f}")
        if avg < best: best, patience = avg, 0
        else:
            patience += 1
            if patience >= patience_lim:
                print(f"Early stop @ {ep+1}"); break
    return model

###############################################################################
# 4. Cache Builder (entity & relation)
###############################################################################

def build_cache(file2trip: Dict[str,List[Tuple[int,int,int]]], ent_np: np.ndarray, rel_np: np.ndarray):
    global ent_emb_cache, rel_emb_cache, GLOBAL_ENT, GLOBAL_REL
    GLOBAL_ENT = torch.tensor(ent_np, device=DEVICE)  # (N,d)
    GLOBAL_REL = torch.tensor(rel_np, device=DEVICE)  # (R,d)
    for fp, triples in file2trip.items():
        ents = {h for h,_,t in triples}.union({t for h,_,t in triples})
        rels = {r for _,r,_ in triples}
        ent_emb_cache[fp] = ent_np[list(ents)] if ents else np.zeros((0, ent_np.shape[1]))
        rel_emb_cache[fp] = rel_np[list(rels)] if rels else np.zeros((0, rel_np.shape[1]))

###############################################################################
# 5. Similarity helpers
###############################################################################

def _avg(arr: np.ndarray | None):
    if arr is None or arr.size==0: return None
    return arr.mean(axis=0)

def _cos(a: np.ndarray|None, b: np.ndarray|None):
    return 1-cosine(a,b) if a is not None and b is not None else 0.0

# (A) simple average of entity + relation avg

def sim_average(fA:str, fB:str):
    ea = _avg(ent_emb_cache[fA]); eb = _avg(ent_emb_cache[fB])
    ra = _avg(rel_emb_cache[fA]); rb = _avg(rel_emb_cache[fB])
    return (_cos(ea,eb)+_cos(ra,rb))/2

# (B) weighted average

def _weighted_avg(vecs: np.ndarray, weights: np.ndarray):
    if vecs.size==0: return None
    return (vecs.T @ weights / weights.sum())

def sim_weighted(fA:str, fB:str, file2trip):
    def gather(file, is_rel=False):
        cnt = Counter()
        for h,r,t in file2trip[file]:
            if is_rel: cnt[r]+=1
            else: cnt[h]+=1; cnt[t]+=1
        ids, ws = zip(*cnt.items()) if cnt else ([],[])
        if not ids: return None
        arr = rel_emb_cache[file] if is_rel else ent_emb_cache[file]
        return _weighted_avg(arr, np.array(ws,dtype=float))
    entA, entB = gather(fA), gather(fB)
    relA, relB = gather(fA,True), gather(fB,True)
    return (_cos(entA,entB)+_cos(relA,relB))/2

# (C) pairwise‑max over threshold ratio (entities+relations)
THR=0.85

def _pair_ratio(A: torch.Tensor, B: torch.Tensor):
    if A.size(0)==0 or B.size(0)==0: return torch.tensor([],device=DEVICE)
    A = torch.nn.functional.normalize(A,dim=1); B = torch.nn.functional.normalize(B,dim=1)
    sims = A @ B.T
    return sims.max(1)[0] if A.size(0)<B.size(0) else sims.max(0)[0]

def sim_pairs(fA:str,fB:str):
    EA = torch.tensor(ent_emb_cache[fA],device=DEVICE)
    EB = torch.tensor(ent_emb_cache[fB],device=DEVICE)
    RA = torch.tensor(rel_emb_cache[fA],device=DEVICE)
    RB = torch.tensor(rel_emb_cache[fB],device=DEVICE)
    r1 = _pair_ratio(EA,EB)
    r2 = _pair_ratio(RA,RB)
    all_r = torch.cat([r1,r2]) if r1.numel() and r2.numel() else r1 if r1.numel() else r2
    return (all_r>THR).float().mean().item() if all_r.numel() else 0.0

###############################################################################
# 6. Wrapper
###############################################################################

def graph_similarity(fA,str_fB,file2trip,method):
    if method=='average': return sim_average(fA,str_fB)
    if method=='weighted average': return sim_weighted(fA,str_fB,file2trip)
    if method=='pairs': return sim_pairs(fA,str_fB)
    raise ValueError(method)

###############################################################################
# 7. Evaluation over folders
###############################################################################

def evaluate_folders(orig_dir, mod_dir, out_dir, file2trip, method):
    orig_files = [os.path.join(r,f) for r,_,fs in os.walk(orig_dir) for f in fs if f.endswith('.json')]
    mod_files  = [os.path.join(r,f) for r,_,fs in os.walk(mod_dir)  for f in fs if f.endswith('.json')]
    os.makedirs(out_dir, exist_ok=True)
    for mf in tqdm(mod_files, desc="Comparing"):
        out_fp = os.path.join(out_dir, os.path.basename(mf)+"_results.json")
        if os.path.exists(out_fp): continue
        res=[]
        for of in orig_files:
            s = graph_similarity(of,mf,file2trip,method)
            res.append({"original":os.path.basename(of),"similarity_score":float(s)})
        with open(out_fp,'w',encoding='utf-8') as fw:
            json.dump(res,fw,ensure_ascii=False,indent=2)

###############################################################################
# 8. End‑to‑End
###############################################################################

def run_pipeline(dataset:str, txt_mod:str, mod_opt:str, model_name:str='RotatE', method:str='pairs'):
    orig_dir = os.path.join(dataset,'original_graph')
    mod_dir  = os.path.join(dataset,txt_mod,mod_opt)
    out_dir  = os.path.join('Result', dataset, txt_mod, mod_opt, 'kge', model_name, method)

    ent2id, rel2id, triplets, file2trip = load_kgs_and_build_dicts([orig_dir,mod_dir])

    model = train_kge(model_name, triplets, len(ent2id), len(rel2id), epochs=1000)

    build_cache(file2trip,
                model.node_emb.weight.detach().cpu().numpy(),
                model.rel_emb.weight.detach().cpu().numpy())

    evaluate_folders(orig_dir, mod_dir, out_dir, file2trip, method)

###############################################################################
# 9. Example main
###############################################################################
if __name__ == '__main__':
    datasets=['wikitext_graph_data']
    txt_mods=['synonym_replacement','context_replacement','dipper_paraphraser']
    mod_opts={'synonym_replacement':['0.3','0.6'],
              'context_replacement':['0.3','0.6'],
              'dipper_paraphraser':['60_0','60_20']}
    models=['RotatE']
    methods=['weighted average']
    for ds in datasets:
        for tm in txt_mods:
            for op in mod_opts[tm]:
                for mdl in models:
                    for mth in methods:
                        run_pipeline(ds,tm,op,mdl,mth)
                        ent_emb_cache.clear(); rel_emb_cache.clear()
                        gc.collect(); torch.cuda.empty_cache()
