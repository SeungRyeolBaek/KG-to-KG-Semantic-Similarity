import os
import json
from tqdm import tqdm
import numpy as np
import torch
from ingram_utils import *
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine

# ============================ 하이퍼파라미터 ============================
# wikitext
# DIMENSION_ENTITY = 32
# DIMENSION_RELATION = 32
# HIDDEN_DIM_RATIO_ENTITY = 8
# HIDDEN_DIM_RATIO_RELATION = 2
# NUM_LAYER_ENT = 2
# NUM_LAYER_REL = 1
# NUM_BIN = 10
# NUM_HEAD = 8

# NELL
DIMENSION_ENTITY = 32
DIMENSION_RELATION = 32
HIDDEN_DIM_RATIO_ENTITY = 4
HIDDEN_DIM_RATIO_RELATION = 1
NUM_LAYER_ENT = 3
NUM_LAYER_REL = 2
NUM_BIN = 10
NUM_HEAD = 8

# ============================ 모델 로드 함수 ============================
def load_model(ckpt_path, model):
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Checkpoint loaded from {ckpt_path}")

# ============================ 임베딩 계산(쌍) 함수 ============================
def initialize(test_data, msg, d_e, d_r, B):
    init_emb_ent = torch.zeros((test_data.num_ent, d_e)).cuda()
    init_emb_rel = torch.zeros((2 * test_data.num_rel, d_r)).cuda()
    gain = torch.nn.init.calculate_gain('relu')

    torch.nn.init.xavier_normal_(init_emb_ent, gain=gain)
    torch.nn.init.xavier_normal_(init_emb_rel, gain=gain)

    relation_triplets = generate_relation_triplets(msg, test_data.num_ent, test_data.num_rel, B)
    relation_triplets = torch.tensor(relation_triplets).cuda()
    return init_emb_ent, init_emb_rel, relation_triplets
def compute_pair_embeddings(graph1, graph2, model, d_e, d_r, B):
    """
    (graph1, graph2) 쌍에 대해 CombinedTestData로 InGram 임베딩 계산 후
    두 그래프 각각에 대한 (emb_ent, emb_rel, freq_ent, freq_rel) 반환
    """
    test_data = CombinedTestData([graph1, graph2])
    msg = np.array(test_data.msg_triplets)

    init_emb_ent, init_emb_rel, relation_triplets = initialize(test_data, msg, d_e, d_r, B)

    with torch.no_grad():
        emb_ent, emb_rel = model(init_emb_ent, init_emb_rel, torch.tensor(msg).cuda(), relation_triplets)

    emb_ent = emb_ent.cpu().numpy()
    emb_rel = emb_rel.cpu().numpy()

    # 그래프1 인덱스
    g1_entities, g1_relations, g1_ent_freq, g1_rel_freq = test_data.get_graph_indices(graph1)
    emb_ent1 = emb_ent[list(g1_entities)]
    emb_rel1 = emb_rel[list(g1_relations)]

    # 그래프2 인덱스
    g2_entities, g2_relations, g2_ent_freq, g2_rel_freq = test_data.get_graph_indices(graph2)
    emb_ent2 = emb_ent[list(g2_entities)]
    emb_rel2 = emb_rel[list(g2_relations)]

    return (emb_ent1, emb_rel1, g1_ent_freq, g1_rel_freq,
            emb_ent2, emb_rel2, g2_ent_freq, g2_rel_freq)

# ============================ 쌍 단위 캐시 ============================
pair_cache = {}  # {(graph1, graph2): (emb_ent1, emb_rel1, freq_ent1, freq_rel1, emb_ent2, emb_rel2, freq_ent2, freq_rel2)}
def get_or_compute_pair_embeddings(graph1, graph2, model, d_e, d_r, B):
    """
    캐시 확인 후, 없으면 compute_pair_embeddings를 호출해 계산, 있으면 즉시 반환
    키는 (min(graph1, graph2), max(graph1, graph2)) 로 정렬
    """
    key = tuple(sorted([graph1, graph2]))
    if key in pair_cache:
        return pair_cache[key]

    # 새로 계산
    data = compute_pair_embeddings(key[0], key[1], model, d_e, d_r, B)
    pair_cache[key] = data
    return data

# ============================ 스코어 계산 함수 ============================
def pairwise_scoring(emb_ent1, emb_rel1, emb_ent2, emb_rel2):
    # 캐싱된 엔티티 및 릴레이션 임베딩을 GPU로 로드
    emb_ent1 = torch.tensor(emb_ent1, dtype=torch.float32, device='cuda')
    emb_ent2 = torch.tensor(emb_ent2, dtype=torch.float32, device='cuda')
    emb_rel1 = torch.tensor(emb_rel1, dtype=torch.float32, device='cuda')
    emb_rel2 = torch.tensor(emb_rel2, dtype=torch.float32, device='cuda')
	
    # 코사인 유사도를 위해 정규화
    emb_ent1 = torch.nn.functional.normalize(emb_ent1, dim=1)
    emb_ent2 = torch.nn.functional.normalize(emb_ent2, dim=1)
    emb_rel1 = torch.nn.functional.normalize(emb_rel1, dim=1)
    emb_rel2 = torch.nn.functional.normalize(emb_rel2, dim=1)

    # 엔티티 코사인 유사도 계산
    ent_sim_matrix = torch.matmul(emb_ent1, emb_ent2.T)
    if emb_ent1.shape[0] < emb_ent2.shape[0]:
        max_ent_similarities, _ = ent_sim_matrix.max(dim=1)
    else:
        max_ent_similarities, _ = ent_sim_matrix.max(dim=0)

    # 릴레이션 코사인 유사도 계산
    rel_sim_matrix = torch.matmul(emb_rel1, emb_rel2.T)
    if emb_rel1.shape[0] < emb_rel2.shape[0]:
        max_rel_similarities, _ = rel_sim_matrix.max(dim=1)
    else:
        max_rel_similarities, _ = rel_sim_matrix.max(dim=0)

    # 두 배열 합치고, threshold 이상 비율 계산
	threshold=0.85
    all_similarities = torch.cat([max_ent_similarities, max_rel_similarities])
    ratio = torch.mean((all_similarities > threshold).float()).item()

    return ratio

def average_scoring(emb_ent1, emb_rel1, emb_ent2, emb_rel2):
    e1_mean = np.mean(emb_ent1, axis=0)
    r1_mean = np.mean(emb_rel1, axis=0)
    e2_mean = np.mean(emb_ent2, axis=0)
    r2_mean = np.mean(emb_rel2, axis=0)

    e_sim = 1 - cosine(e1_mean, e2_mean)
    r_sim = 1 - cosine(r1_mean, r2_mean)
    return (e_sim + r_sim) / 2
def weighted_average_scoring(emb_ent1, emb_rel1, freq_ent1, freq_rel1,
                             emb_ent2, emb_rel2, freq_ent2, freq_rel2):
    e1_w = np.average(emb_ent1, axis=0, weights=list(freq_ent1.values()))
    r1_w = np.average(emb_rel1, axis=0, weights=list(freq_rel1.values()))
    e2_w = np.average(emb_ent2, axis=0, weights=list(freq_ent2.values()))
    r2_w = np.average(emb_rel2, axis=0, weights=list(freq_rel2.values()))

    e_sim = 1 - cosine(e1_w, e2_w)
    r_sim = 1 - cosine(r1_w, r2_w)
    return (e_sim + r_sim) / 2

def scoring(method, emb_ent1, emb_rel1, freq_ent1, freq_rel1,
            emb_ent2, emb_rel2, freq_ent2, freq_rel2):
    if method == "pairs":
        return pairwise_scoring(emb_ent1, emb_rel1, emb_ent2, emb_rel2)
    elif method == "average":
        return average_scoring(emb_ent1, emb_rel1, emb_ent2, emb_rel2)
    elif method == "weighted_average":
        return weighted_average_scoring(emb_ent1, emb_rel1, freq_ent1, freq_rel1,
                                        emb_ent2, emb_rel2, freq_ent2, freq_rel2)
    return 0.0

# ============================ JSON 폴더 비교 ============================
def evaluate_json_folders(method, original_folder, modified_folder, output_folder, model, d_e, d_r, B):
    """
    original_folder 안의 모든 json vs modified_folder 안의 모든 json 쌍:
      - get_or_compute_pair_embeddings를 이용하여 임베딩을 계산 or 캐시에서 불러옴
      - scoring() 계산
    """
    original_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(original_folder)
        for file in files if file.endswith(".json")
    ]
    modified_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(modified_folder)
        for file in files if file.endswith(".json")
    ]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for mfile in tqdm(modified_files, desc=f"Processing {method}"):
        modified_name = os.path.basename(mfile)
        results = []

        for ofile in original_files:
            # 캐시 확인 -> 없으면 모델 실행 -> 임베딩 가져옴
            (emb_ent1, emb_rel1, freq_ent1, freq_rel1,
             emb_ent2, emb_rel2, freq_ent2, freq_rel2) = get_or_compute_pair_embeddings(ofile, mfile, model, d_e, d_r, B)

            sim = scoring(method, emb_ent1, emb_rel1, freq_ent1, freq_rel1,
                          emb_ent2, emb_rel2, freq_ent2, freq_rel2)

            results.append({
                "original": os.path.basename(ofile),
                "similarity_score": float(sim)
            })

        out_file = os.path.join(output_folder, f"{modified_name}_results.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"All comparisons completed for {method} in {output_folder}.")
def ingram_scoring(model, dataset, modification, option, method):
    if dataset == "c4_dataset":
        data_folder = "c4_subset_graph"
        result_folder = "C4"
    elif dataset == "wikitext":
        data_folder = "wikitext_graph_data"
        result_folder = "Wikitext"
    elif dataset == "cc_news":
        data_folder = "cc_news_graph"
        result_folder = "CC_news"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    original_folder = os.path.join(data_folder, "original_graph")
    modified_folder = os.path.join(data_folder, modification, option)
    output_folder = os.path.join(result_folder, modification, option, "ingram", method)

    evaluate_json_folders(method, original_folder, modified_folder, output_folder,
                          model, DIMENSION_ENTITY, DIMENSION_RELATION, NUM_BIN)

if __name__ == "__main__":
    CKPT_PATH = "./ingram-checkpoint/best.ckpt"
    my_model = InGram(
        dim_ent=DIMENSION_ENTITY,
        hid_dim_ratio_ent=HIDDEN_DIM_RATIO_ENTITY,
        dim_rel=DIMENSION_RELATION,
        hid_dim_ratio_rel=HIDDEN_DIM_RATIO_RELATION,
        num_bin=NUM_BIN,
        num_layer_ent=NUM_LAYER_ENT,
        num_layer_rel=NUM_LAYER_REL,
        num_head=NUM_HEAD
    ).cuda()
    load_model(CKPT_PATH, my_model)

    datasets = ["wikitext", "c4_dataset","cc_news"]
    methods = ["pairs", "average", "weighted_average"]
    modifications = ["synonym_replacement", "context_replacement", "dipper_paraphraser"]
    modification_options = {
        "synonym_replacement": ["0.3", "0.6"],
        "context_replacement": ["0.3", "0.6"],
        "dipper_paraphraser": ["60_0", "60_20"]
    }

    for ds in datasets:
        for mod in modifications:
            for opt in modification_options[mod]:
                for m in methods:
                    ingram_scoring(my_model, ds, mod, opt, m)
