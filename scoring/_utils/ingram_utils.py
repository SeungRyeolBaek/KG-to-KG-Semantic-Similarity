import numpy as np
import torch
from tqdm import tqdm
import json
import random
import igraph
import copy
import time
import os
import math
from scipy.sparse import csr_matrix
import torch.nn as nn
import torch.nn.functional as F


def remove_duplicate(x):
	return list(dict.fromkeys(x))

def generate_neg(triplets, num_ent, num_neg = 1):
	import torch
	neg_triplets = triplets.unsqueeze(dim=1).repeat(1,num_neg,1)
	rand_result = torch.rand((len(triplets),num_neg)).cuda()
	perturb_head = rand_result < 0.5
	perturb_tail = rand_result >= 0.5
	rand_idxs = torch.randint(low=0, high = num_ent-1, size = (len(triplets),num_neg)).cuda()
	rand_idxs[perturb_head] += rand_idxs[perturb_head] >= neg_triplets[:,:,0][perturb_head]
	rand_idxs[perturb_tail] += rand_idxs[perturb_tail] >= neg_triplets[:,:,2][perturb_tail]
	neg_triplets[:,:,0][perturb_head] = rand_idxs[perturb_head]
	neg_triplets[:,:,2][perturb_tail] = rand_idxs[perturb_tail]
	neg_triplets = torch.cat(torch.split(neg_triplets, 1, dim = 1), dim = 0).squeeze(dim = 1)

	return neg_triplets

def get_rank(triplet, scores, filters, target = 0):
	thres = scores[triplet[0,target]].item()
	scores[filters] = thres - 1
	rank = (scores > thres).sum() + (scores == thres).sum()//2 + 1
	return rank.item()

def get_metrics(rank):
	rank = np.array(rank, dtype = np.int32)
	mr = np.mean(rank)
	mrr = np.mean(1 / rank)
	hit10 = np.sum(rank < 11) / len(rank)
	hit3 = np.sum(rank < 4) / len(rank)
	hit1 = np.sum(rank < 2) / len(rank)
	return mr, mrr, hit10, hit3, hit1

def remove_duplicate(x):
	return list(dict.fromkeys(x))

class TrainData():
    def __init__(self, path):
        self.path = path
        self.rel_info = {}
        self.pair_info = {}
        self.spanning = []
        self.remaining = []
        self.ent2id = None
        self.rel2id = None
        self.train_ratio = 0.4
        self.val_ratio = 0.2
        self.filter_dict = {}
        
        # Load triplets and initialize entity/relation mappings
        self.id2ent, self.id2rel, self.triplets = self.read_triplet(path)
        self.num_triplets = len(self.triplets)
        self.num_ent = len(self.id2ent)
        self.num_rel = len(self.id2rel)

    def read_triplet(self, path):
        id2ent, id2rel, triplets = [], [], []
        
        # Iterate through all JSON files in the directory
        for filename in os.listdir(path):
            if filename.endswith(".json"):
                file_path = os.path.join(path, filename)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    for rel in data["relationships"]:
                        head = rel["source"]["id"]
                        tail = rel["target"]["id"]
                        relation = rel["type"]
                        id2ent.append(head)
                        id2ent.append(tail)
                        id2rel.append(relation)
                        triplets.append((head, relation, tail))

        # Remove duplicates and create entity/relation mappings
        id2ent = remove_duplicate(id2ent)
        id2rel = remove_duplicate(id2rel)
        self.ent2id = {ent: idx for idx, ent in enumerate(id2ent)}
        self.rel2id = {rel: idx for idx, rel in enumerate(id2rel)}
        
        # Convert triplets to ID format
        triplets = [(self.ent2id[h], self.rel2id[r], self.ent2id[t]) for h, r, t in triplets]

        # Create relation and pair information for each triplet
        for (h, r, t) in triplets:
            if (h, t) in self.rel_info:
                self.rel_info[(h, t)].append(r)
            else:
                self.rel_info[(h, t)] = [r]
            if r in self.pair_info:
                self.pair_info[r].append((h, t))
            else:
                self.pair_info[r] = [(h, t)]

        # Create filter dictionary
        for h, r, t in triplets:
            if ('_', r, t) not in self.filter_dict:
                self.filter_dict[('_', r, t)] = [h]
            else:
                self.filter_dict[('_', r, t)].append(h)

            if (h, '_', t) not in self.filter_dict:
                self.filter_dict[(h, '_', t)] = [r]
            else:
                self.filter_dict[(h, '_', t)].append(r)

            if (h, r, '_') not in self.filter_dict:
                self.filter_dict[(h, r, '_')] = [t]
            else:
                self.filter_dict[(h, r, '_')].append(t)

        # Create spanning tree using igraph for transductive split
        G = igraph.Graph.TupleList(np.array(triplets)[:, 0::2])
        G_ent = igraph.Graph.TupleList(np.array(triplets)[:, 0::2], directed=True)
        spanning = G_ent.spanning_tree()
        G_ent.delete_edges(spanning.get_edgelist())
        
        for e in spanning.es:
            e1, e2 = e.tuple
            e1 = spanning.vs[e1]["name"]
            e2 = spanning.vs[e2]["name"]
            self.spanning.append((e1, e2))

        print("-----Train Data Statistics-----")
        print(f"{len(self.ent2id)} entities, {len(self.rel2id)} relations")
        print(f"{len(triplets)} triplets")
        # Define sizes for train, validation, and message sets
        num_train = int(len(triplets) * self.train_ratio)
        num_val = int(len(triplets) * self.val_ratio)
        num_msg = len(triplets) - num_train - num_val
        # Print stats
        print(f"Message set (model input) has {num_msg} triplets")
        print(f"Training set (for loss calculation) has {num_train} triplets")
        print(f"Validation set (for validation loss) has {num_val} triplets")
        
        # Convert triplets to include inverse relations for training
        self.triplets_with_inv = np.array([(t, r + len(id2rel), h) for h, r, t in triplets] + triplets)
        self.triplet2idx = {triplet: idx for idx, triplet in enumerate(triplets)}
        
        return id2ent, id2rel, triplets

    def split_transductive(self):
        # Define sizes for train, validation, and message sets
        num_train = int(len(self.triplets) * self.train_ratio)
        num_val = int(len(self.triplets) * self.val_ratio)
        num_msg = len(self.triplets) - num_train - num_val

        val_triplets = self.triplets[num_msg + num_train:]
        triplets = self.triplets[:num_msg + num_train]
        
        random.shuffle(triplets)

        # Split into msg (model input), train (for train_loss), and val (for validation_loss) triplets
        msg_triplets = triplets[:num_msg]
        train_triplets = triplets[num_msg:num_msg + num_train]

        return np.array(msg_triplets), np.array(train_triplets), np.array(val_triplets)


class TestNewData():
    def __init__(self, path):
        self.path = path
        self.ent2id = None
        self.rel2id = None
        self.id2ent, self.id2rel, self.msg_triplets, self.filter_dict = self.read_triplet()
        self.num_ent, self.num_rel = len(self.id2ent), len(self.id2rel)

    def read_triplet(self):
        id2ent, id2rel, msg_triplets = [], [], []

        # Load and parse triplets
        with open(self.path, 'r') as f:
            data = json.load(f)
            for rel in data["relationships"]:
                head = rel["source"]["id"]
                tail = rel["target"]["id"]
                relation = rel["type"]
                id2ent.append(head)
                id2ent.append(tail)
                id2rel.append(relation)
                msg_triplets.append((head, relation, tail))

        # Remove duplicates and create id mappings
        id2ent = remove_duplicate(id2ent)
        id2rel = remove_duplicate(id2rel)
        self.ent2id = {ent: idx for idx, ent in enumerate(id2ent)}
        self.rel2id = {rel: idx for idx, rel in enumerate(id2rel)}
        num_rel = len(self.rel2id)

        # Convert triplets to id format
        msg_triplets = [(self.ent2id[h], self.rel2id[r], self.ent2id[t]) for h, r, t in msg_triplets]
        msg_inv_triplets = [(t, r + num_rel, h) for h, r, t in msg_triplets]
        msg_triplets = msg_triplets + msg_inv_triplets

        # Filter dictionary generation
        filter_dict = {}
        for h, r, t in msg_triplets:
            if ('_', r, t) not in filter_dict:
                filter_dict[('_', r, t)] = [h]
            else:
                filter_dict[('_', r, t)].append(h)

            if (h, '_', t) not in filter_dict:
                filter_dict[(h, '_', t)] = [r]
            else:
                filter_dict[(h, '_', t)].append(r)
                
            if (h, r, '_') not in filter_dict:
                filter_dict[(h, r, '_')] = [t]
            else:
                filter_dict[(h, r, '_')].append(t)

        return id2ent, id2rel, np.array(msg_triplets), filter_dict

class CombinedTestData():
    def __init__(self, paths):
        self.paths = paths
        self.ent2id = {}
        self.rel2id = {}
        self.id2ent = []
        self.id2rel = []
        self.msg_triplets = []
        self.graph_index_map = {}  # 각 파일별로 인덱스 기록
        self.read_all_triplets()
        self.num_ent = len(self.ent2id)
        self.num_rel = len(self.rel2id)

    def read_all_triplets(self):
        current_ent_index = 0
        current_rel_index = 0

        for path in self.paths:
            with open(path, 'r') as f:
                data = json.load(f)
                graph_name = path  # 전체 경로 혹은 다른 방식으로 유니크하게

                # 그래프별로 freq dict 초기화
                self.graph_index_map[graph_name] = {
                    "entities": set(),
                    "relations": set(),
                    "entity_freq": {},  # {엔티티인덱스: 등장횟수}
                    "rel_freq": {},     # {관계인덱스: 등장횟수}
                }

                for rel in data["relationships"]:
                    head = rel["source"]["id"]
                    tail = rel["target"]["id"]
                    relation = rel["type"]

                    # 엔티티 인덱싱
                    if head not in self.ent2id:
                        self.ent2id[head] = current_ent_index
                        self.id2ent.append(head)
                        current_ent_index += 1
                    if tail not in self.ent2id:
                        self.ent2id[tail] = current_ent_index
                        self.id2ent.append(tail)
                        current_ent_index += 1

                    # 관계 인덱싱
                    if relation not in self.rel2id:
                        self.rel2id[relation] = current_rel_index
                        self.id2rel.append(relation)
                        current_rel_index += 1

                    head_idx = self.ent2id[head]
                    tail_idx = self.ent2id[tail]
                    rel_idx  = self.rel2id[relation]

                    # 인덱스 기록
                    self.graph_index_map[graph_name]["entities"].update([head_idx, tail_idx])
                    self.graph_index_map[graph_name]["relations"].add(rel_idx)

                    # ----- 여기서 freq 카운팅 -----
                    if head_idx not in self.graph_index_map[graph_name]["entity_freq"]:
                        self.graph_index_map[graph_name]["entity_freq"][head_idx] = 0
                    self.graph_index_map[graph_name]["entity_freq"][head_idx] += 1

                    if tail_idx not in self.graph_index_map[graph_name]["entity_freq"]:
                        self.graph_index_map[graph_name]["entity_freq"][tail_idx] = 0
                    self.graph_index_map[graph_name]["entity_freq"][tail_idx] += 1

                    if rel_idx not in self.graph_index_map[graph_name]["rel_freq"]:
                        self.graph_index_map[graph_name]["rel_freq"][rel_idx] = 0
                    self.graph_index_map[graph_name]["rel_freq"][rel_idx] += 1

                    # 트리플릿 저장
                    self.msg_triplets.append((head_idx, rel_idx, tail_idx))

        # 트리플릿의 역방향 추가
        num_rel = len(self.rel2id)
        self.msg_triplets += [(t, r + num_rel, h) for h, r, t in self.msg_triplets]

    def get_graph_indices(self, graph_name):
        """특정 그래프에 속하는 엔티티와 관계 인덱스 및 freq를 반환"""
        if graph_name not in self.graph_index_map:
            raise ValueError(f"Graph {graph_name} not found.")
        gmap = self.graph_index_map[graph_name]
        return (
            gmap["entities"],     # set of entity indices
            gmap["relations"],    # set of relation indices
            gmap["entity_freq"],  # dict: {엔티티인덱스: freq}
            gmap["rel_freq"],     # dict: {릴레이션인덱스: freq}
        )

def evaluate(my_model, eval_triplets, num_entities, init_emb_ent, init_emb_rel, relation_triplets, filter_dict):
    with torch.no_grad():
        my_model.eval()
        
        # Initialize embeddings
        emb_ent, emb_rel = my_model(init_emb_ent, init_emb_rel, eval_triplets, relation_triplets)
        
        head_ranks = []
        tail_ranks = []
        ranks = []

        for triplet in tqdm(eval_triplets):
            triplet = triplet.unsqueeze(dim=0)

            # Head prediction
            head_corrupt = triplet.repeat(num_entities, 1)
            head_corrupt[:, 0] = torch.arange(end=num_entities)
            head_scores = my_model.score(emb_ent, emb_rel, head_corrupt)
            head_filters = filter_dict.get(('_', int(triplet[0, 1].item()), int(triplet[0, 2].item())), set())
            head_rank = get_rank(triplet, head_scores, head_filters, target=0)

            # Tail prediction
            tail_corrupt = triplet.repeat(num_entities, 1)
            tail_corrupt[:, 2] = torch.arange(end=num_entities)
            tail_scores = my_model.score(emb_ent, emb_rel, tail_corrupt)
            tail_filters = filter_dict.get((int(triplet[0, 0].item()), int(triplet[0, 1].item()), '_'), set())
            tail_rank = get_rank(triplet, tail_scores, tail_filters, target=2)

            # Collect ranks
            ranks.append(head_rank)
            head_ranks.append(head_rank)
            ranks.append(tail_rank)
            tail_ranks.append(tail_rank)

        # Print metrics
        print("--------LP--------")
        mr, mrr, hit10, hit3, hit1 = get_metrics(ranks)
        print(f"MR: {mr:.1f}")
        print(f"MRR: {mrr:.3f}")
        print(f"Hits@10: {hit10:.3f}")
        print(f"Hits@1: {hit1:.3f}")
        return mr, mrr, hit10, hit1

class InGramEntityLayer(nn.Module):
    def __init__(self, dim_in_ent, dim_out_ent, dim_rel, bias = True, num_head = 8):
        super(InGramEntityLayer, self).__init__()

        self.dim_out_ent = dim_out_ent
        self.dim_hid_ent = dim_out_ent // num_head
        assert dim_out_ent == self.dim_hid_ent * num_head
        self.num_head = num_head

        self.attn_proj = nn.Linear(2 * dim_in_ent + dim_rel, dim_out_ent, bias = bias)
        self.attn_vec = nn.Parameter(torch.zeros((1, num_head, self.dim_hid_ent)))
        self.aggr_proj = nn.Linear(dim_in_ent + dim_rel, dim_out_ent, bias = bias)

        self.dim_rel = dim_rel
        self.act = nn.LeakyReLU(negative_slope = 0.2)
        self.bias = bias
        self.param_init()
    
    def param_init(self):
        nn.init.xavier_normal_(self.attn_proj.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.attn_vec, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.aggr_proj.weight, gain = nn.init.calculate_gain('relu'))
        if self.bias:
            nn.init.zeros_(self.attn_proj.bias)
            nn.init.zeros_(self.aggr_proj.bias)
    
    def forward(self, emb_ent, emb_rel, triplets): 
        num_ent = len(emb_ent)
        num_rel = len(emb_rel)
        head_idxs = triplets[..., 0]
        rel_idxs = triplets[..., 1]
        tail_idxs = triplets[..., 2]

        ent_freq = torch.zeros((num_ent, )).cuda().index_add(dim = 0, index = tail_idxs, \
                                                             source = torch.ones_like(tail_idxs, dtype = torch.float).cuda()).unsqueeze(dim = 1)
        self_rel = torch.zeros((num_ent, self.dim_rel)).cuda().index_add(dim=0, index = tail_idxs, source = emb_rel[rel_idxs])/(ent_freq + 1e-16)

        # add self-loops
        emb_rels = torch.cat([emb_rel[rel_idxs], self_rel], dim = 0)
        head_idxs = torch.cat([head_idxs, torch.arange(num_ent).cuda()], dim = 0)
        tail_idxs = torch.cat([tail_idxs, torch.arange(num_ent).cuda()], dim = 0)

        concat_mat_att = torch.cat([emb_ent[tail_idxs], emb_ent[head_idxs], \
                                    emb_rels], dim = -1)

        attn_val_raw = (self.act(self.attn_proj(concat_mat_att).view(-1, self.num_head, self.dim_hid_ent)) * 
                       self.attn_vec).sum(dim = -1, keepdim = True)

        scatter_idx = tail_idxs.unsqueeze(dim = -1).repeat(1, self.num_head).unsqueeze(dim = -1)

        attn_val_max = torch.zeros((num_ent, self.num_head, 1)).cuda().scatter_reduce(dim = 0, \
                                                                    index = scatter_idx, \
                                                                    src = attn_val_raw, reduce = 'amax', \
                                                                    include_self = False)
        attn_val = torch.exp(attn_val_raw - attn_val_max[tail_idxs])
        
        attn_sums = torch.zeros((num_ent, self.num_head, 1)).cuda().index_add(dim = 0, index = tail_idxs, source = attn_val)

        beta = attn_val / (attn_sums[tail_idxs]+1e-16)

        concat_mat = torch.cat([emb_ent[head_idxs], emb_rels], dim = -1)

        aggr_val = beta * self.aggr_proj(concat_mat).view(-1, self.num_head, self.dim_hid_ent)
        
        output = torch.zeros((num_ent, self.num_head, self.dim_hid_ent)).cuda().index_add(dim = 0, index = tail_idxs, source = aggr_val)

        return output.flatten(1,-1)

class InGramRelationLayer(nn.Module):
    def __init__(self, dim_in_rel, dim_out_rel, num_bin, bias = True, num_head = 8):
        super(InGramRelationLayer, self).__init__()

        self.dim_out_rel = dim_out_rel
        self.dim_hid_rel = dim_out_rel // num_head
        assert dim_out_rel == self.dim_hid_rel * num_head

        self.attn_proj = nn.Linear(2*dim_in_rel, dim_out_rel, bias = bias)
        self.attn_bin = nn.Parameter(torch.zeros(num_bin, num_head, 1))
        self.attn_vec = nn.Parameter(torch.zeros(1, num_head, self.dim_hid_rel))
        self.aggr_proj = nn.Linear(dim_in_rel, dim_out_rel, bias = bias)
        self.num_head = num_head

        self.act = nn.LeakyReLU(negative_slope = 0.2)
        self.num_bin = num_bin
        self.bias = bias

        self.param_init()
    
    def param_init(self):
        nn.init.xavier_normal_(self.attn_proj.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.attn_vec, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.aggr_proj.weight, gain = nn.init.calculate_gain('relu'))
        if self.bias:
            nn.init.zeros_(self.attn_proj.bias)
            nn.init.zeros_(self.aggr_proj.bias)
    
    def forward(self, emb_rel, relation_triplets):
        num_rel = len(emb_rel)
        
        head_idxs = relation_triplets[..., 0]
        tail_idxs = relation_triplets[..., 1]
        concat_mat = torch.cat([emb_rel[head_idxs], emb_rel[tail_idxs]], dim = -1)

        attn_val_raw = (self.act(self.attn_proj(concat_mat).view(-1, self.num_head, self.dim_hid_rel)) * \
                        self.attn_vec).sum(dim = -1, keepdim = True) + self.attn_bin[relation_triplets[...,2]]

        scatter_idx = head_idxs.unsqueeze(dim = -1).repeat(1, self.num_head).unsqueeze(dim = -1)

        attn_val_max = torch.zeros((num_rel, self.num_head, 1)).cuda().scatter_reduce(dim = 0, \
                                                                    index = scatter_idx, \
                                                                    src = attn_val_raw, reduce = 'amax', \
                                                                    include_self = False)
        attn_val = torch.exp(attn_val_raw - attn_val_max[head_idxs])

        attn_sums = torch.zeros((num_rel, self.num_head, 1)).cuda().index_add(dim = 0, index = head_idxs, source = attn_val)

        beta = attn_val / (attn_sums[head_idxs]+1e-16)
        
        output = torch.zeros((num_rel, self.num_head, self.dim_hid_rel)).cuda().index_add(dim = 0, \
                                                                                            index = head_idxs, 
                                                                                            source = beta * self.aggr_proj(emb_rel[tail_idxs]).view(-1, self.num_head, self.dim_hid_rel))

        return output.flatten(1,-1)

class InGram(nn.Module):
    def __init__(self, dim_ent, hid_dim_ratio_ent, dim_rel, hid_dim_ratio_rel, num_bin=10, num_layer_ent=2, num_layer_rel=2, \
                 num_head = 8, bias = True):
        super(InGram, self).__init__()

        layers_ent = []
        layers_rel = []
        layer_dim_ent = hid_dim_ratio_ent * dim_ent
        layer_dim_rel = hid_dim_ratio_rel * dim_rel
        for _ in range(num_layer_ent):
            layers_ent.append(InGramEntityLayer(layer_dim_ent, layer_dim_ent, layer_dim_rel, \
                                                bias = bias, num_head = num_head))
        for _ in range(num_layer_rel):
            layers_rel.append(InGramRelationLayer(layer_dim_rel, layer_dim_rel, num_bin, \
                                                  bias = bias, num_head = num_head))
        res_proj_ent = []
        for _ in range(num_layer_ent):
            res_proj_ent.append(nn.Linear(layer_dim_ent, layer_dim_ent, bias = bias))
        
        res_proj_rel = []
        for _ in range(num_layer_rel):
            res_proj_rel.append(nn.Linear(layer_dim_rel, layer_dim_rel, bias = bias))

        self.res_proj_ent = nn.ModuleList(res_proj_ent)
        self.res_proj_rel = nn.ModuleList(res_proj_rel)
        self.bias = bias
        self.ent_proj1 = nn.Linear(dim_ent, layer_dim_ent, bias = bias)
        self.ent_proj2 = nn.Linear(layer_dim_ent, dim_ent, bias = bias)
        self.layers_ent = nn.ModuleList(layers_ent)
        self.layers_rel = nn.ModuleList(layers_rel)

        self.rel_proj1 = nn.Linear(dim_rel, layer_dim_rel, bias = bias)
        self.rel_proj2 = nn.Linear(layer_dim_rel, dim_rel, bias = bias)
        self.rel_proj = nn.Linear(dim_rel, dim_ent, bias = bias)
        self.num_layer_ent = num_layer_ent
        self.num_layer_rel = num_layer_rel
        self.act = nn.ReLU()

        self.param_init()
    
    def param_init(self):
        nn.init.xavier_normal_(self.ent_proj1.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.ent_proj2.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.rel_proj1.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.rel_proj2.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.rel_proj.weight, gain = nn.init.calculate_gain('relu'))
        for layer_idx in range(self.num_layer_ent):
            nn.init.xavier_normal_(self.res_proj_ent[layer_idx].weight, gain = nn.init.calculate_gain('relu'))
        for layer_idx in range(self.num_layer_rel):
            nn.init.xavier_normal_(self.res_proj_rel[layer_idx].weight, gain = nn.init.calculate_gain('relu'))
        if self.bias:
            nn.init.zeros_(self.ent_proj1.bias)
            nn.init.zeros_(self.ent_proj2.bias)
            nn.init.zeros_(self.rel_proj1.bias)
            nn.init.zeros_(self.rel_proj2.bias)
            nn.init.zeros_(self.rel_proj.bias)
            for layer_idx in range(self.num_layer_ent):
                nn.init.zeros_(self.res_proj_ent[layer_idx].bias)
            for layer_idx in range(self.num_layer_rel):
                nn.init.zeros_(self.res_proj_rel[layer_idx].bias)
            

    def forward(self, emb_ent, emb_rel, triplets, relation_triplets):

        layer_emb_ent = self.ent_proj1(emb_ent)
        layer_emb_rel = self.rel_proj1(emb_rel)
        
        for layer_idx, layer in enumerate(self.layers_rel):
            layer_emb_rel = layer(layer_emb_rel, relation_triplets) + \
                            self.res_proj_rel[layer_idx](layer_emb_rel)
            layer_emb_rel = self.act(layer_emb_rel)
        
        for layer_idx, layer in enumerate(self.layers_ent):

            layer_emb_ent = layer(layer_emb_ent, layer_emb_rel, triplets) + \
                            self.res_proj_ent[layer_idx](layer_emb_ent)
            layer_emb_ent = self.act(layer_emb_ent)

        return self.ent_proj2(layer_emb_ent), self.rel_proj2(layer_emb_rel)


    def score(self, emb_ent, emb_rel, triplets):

        head_idxs = triplets[..., 0]
        rel_idxs = triplets[..., 1]
        tail_idxs = triplets[..., 2]
        head_embs = emb_ent[head_idxs]
        tail_embs = emb_ent[tail_idxs]
        rel_embs = self.rel_proj(emb_rel[rel_idxs])
        output = (head_embs * rel_embs * tail_embs).sum(dim = -1)
        return output


def create_relation_graph(triplet, num_ent, num_rel):
    ind_h = triplet[:,:2]
    ind_t = triplet[:,1:]


    E_h = csr_matrix((np.ones(len(ind_h)), (ind_h[:, 0], ind_h[:, 1])), shape=(num_ent, 2 * num_rel))
    E_t = csr_matrix((np.ones(len(ind_t)), (ind_t[:, 1], ind_t[:, 0])), shape=(num_ent, 2 * num_rel))

    diag_vals_h = E_h.sum(axis=1).A1
    diag_vals_h[diag_vals_h!=0] = 1/(diag_vals_h[diag_vals_h!=0]**2)

    diag_vals_t = E_t.sum(axis=1).A1
    diag_vals_t[diag_vals_t!=0] = 1/(diag_vals_t[diag_vals_t!=0]**2)


    D_h_inv = csr_matrix((diag_vals_h, (np.arange(num_ent), np.arange(num_ent))), shape=(num_ent, num_ent))
    D_t_inv = csr_matrix((diag_vals_t, (np.arange(num_ent), np.arange(num_ent))), shape=(num_ent, num_ent))


    A_h = E_h.transpose() @ D_h_inv @ E_h
    A_t = E_t.transpose() @ D_t_inv @ E_t

    return A_h + A_t

def get_relation_triplets(G_rel, B):
    rel_triplets = []
    for tup in G_rel.get_edgelist():
        h,t = tup
        tupid = G_rel.get_eid(h,t)
        w = G_rel.es[tupid]["weight"]
        rel_triplets.append((int(h), int(t), float(w)))
    rel_triplets = np.array(rel_triplets)

    nnz = len(rel_triplets)

    temp = (-rel_triplets[:,2]).argsort()
    weight_ranks = np.empty_like(temp)
    weight_ranks[temp] = np.arange(nnz) + 1

    relation_triplets = []
    for idx,triplet in enumerate(rel_triplets):
        h,t,w = triplet
        rk = int(math.ceil(weight_ranks[idx]/nnz*B))-1
        relation_triplets.append([int(h), int(t), rk])
        assert rk >= 0
        assert rk < B

    return np.array(relation_triplets)

def generate_relation_triplets(triplet, num_ent, num_rel, B):
    A = create_relation_graph(triplet, num_ent, num_rel)
    G_rel = igraph.Graph.Weighted_Adjacency(A)
    relation_triplets = get_relation_triplets(G_rel, B)
    return relation_triplets