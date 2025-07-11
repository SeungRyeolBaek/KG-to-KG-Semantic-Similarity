import os
import json
from tqdm import tqdm
import numpy as np
import torch
from ingram_utils import *
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine

# ============================ 하이퍼파라미터 ============================
# NELL (example)
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

# ============================ 임베딩 계산(일괄) 함수 ============================
def initialize(test_data, msg, d_e, d_r, B):
    init_emb_ent = torch.zeros((test_data.num_ent, d_e)).cuda()
    init_emb_rel = torch.zeros((2 * test_data.num_rel, d_r)).cuda()
    gain = torch.nn.init.calculate_gain('relu')

    torch.nn.init.xavier_normal_(init_emb_ent, gain=gain)
    torch.nn.init.xavier_normal_(init_emb_rel, gain=gain)

    relation_triplets = generate_relation_triplets(msg, test_data.num_ent, test_data.num_rel, B)
    relation_triplets = torch.tensor(relation_triplets).cuda()
    return init_emb_ent, init_emb_rel, relation_triplets

# ============================ 스코어 계산 함수들 ============================
def pairwise_scoring(emb_ent1, emb_ent2):
    # emb_ent1, emb_ent2: numpy arrays of shape (num_entities_g1, dim), (num_entities_g2, dim)
    emb_ent1 = torch.tensor(emb_ent1, dtype=torch.float32, device='cuda')
    emb_ent2 = torch.tensor(emb_ent2, dtype=torch.float32, device='cuda')

    joint_mean = torch.cat([emb_ent1, emb_ent2], dim=0).mean(0, keepdim=True)
    emb_ent1 = emb_ent1 - joint_mean        # mean-centering
    emb_ent2 = emb_ent2 - joint_mean

    # Normalize for cosine similarity
    emb_ent1 = torch.nn.functional.normalize(emb_ent1, dim=1)
    emb_ent2 = torch.nn.functional.normalize(emb_ent2, dim=1)

    # Compute similarity
    ent_sim_matrix = torch.matmul(emb_ent1, emb_ent2.T)
    if emb_ent1.shape[0] < emb_ent2.shape[0]:
        max_ent_similarities, _ = ent_sim_matrix.max(dim=1)
    else:
        max_ent_similarities, _ = ent_sim_matrix.max(dim=0)

    threshold = 0.97
    all_similarities = torch.cat([max_ent_similarities])
    # ratio = torch.mean((all_similarities > threshold).float()).item()
    ratio = torch.mean((all_similarities >= threshold).float()).item()
    return ratio

    # return max_ent_similarities.mean().item()
    # eps = 1e-8                               # log(0) 방지용
    # sims = torch.clamp(max_ent_similarities, min=eps)
    # gmean = torch.exp(torch.log(sims).mean())  # exp( mean(log s) )

    # return gmean.item()

def average_scoring(emb_ent1, emb_ent2):
    e1_mean = np.mean(emb_ent1, axis=0)
    e2_mean = np.mean(emb_ent2, axis=0)

    e_sim = 1 - cosine(e1_mean, e2_mean)
    return e_sim

def weighted_average_scoring(emb_ent1, freq_ent1,
                             emb_ent2, freq_ent2):
    e1_w = np.average(emb_ent1, axis=0, weights=list(freq_ent1.values()))
    e2_w = np.average(emb_ent2, axis=0, weights=list(freq_ent2.values()))

    e_sim = 1 - cosine(e1_w, e2_w)
    return e_sim

def scoring(method, emb_ent1, freq_ent1,
            emb_ent2, freq_ent2):
    if method == "pairs":
        return pairwise_scoring(emb_ent1, emb_ent2)
    elif method == "average":
        return average_scoring(emb_ent1, emb_ent2)
    elif method == "weighted_average":
        return weighted_average_scoring(emb_ent1, freq_ent1,
                                        emb_ent2, freq_ent2)
    return 0.0

# ============================ JSON 폴더 비교 (일괄 임베딩) ============================
def evaluate_json_folders(method, original_folder, modified_folder, output_folder, model, d_e, d_r, B):
    """
    1) original_folder 안의 모든 JSON과 modified_folder 안의 모든 JSON을 한 번에 CombinedTestData로 모음.
    2) 모델을 단 한 번 호출하여 모든 그래프의 임베딩을 구함.
    3) graph_cache[파일경로]에 (emb_ent, freq_ent) 등을 저장.
    4) 각 (original, modified) 쌍에 대해 scoring.
    """
    # 모든 파일을 모은다
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

    all_files = original_files + modified_files

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # ------------------- 1) CombinedTestData 생성 -------------------
    test_data = CombinedTestData(all_files)
    # 메시지 추출
    msg = np.array(test_data.msg_triplets)

    # ------------------- 2) 모델로 임베딩 한 방에 계산 -------------------
    init_emb_ent, init_emb_rel, relation_triplets = initialize(test_data, msg, d_e, d_r, B)
    with torch.no_grad():
        emb_ent, emb_rel = model(init_emb_ent, init_emb_rel,
                                 torch.tensor(msg).cuda(), relation_triplets)

    # CPU로 이동
    emb_ent = emb_ent.cpu().numpy()
    emb_rel = emb_rel.cpu().numpy()

    # ------------------- 3) graph_cache 저장 -------------------
    # 각 그래프별로 인덱스, freq 등을 추출하여 캐시에 담는다.
    graph_cache = {}
    for file_path in all_files:
        g_entities, g_relations, g_ent_freq, g_rel_freq = test_data.get_graph_indices(file_path)
        # 해당 그래프 엔티티 임베딩
        emb_ent_g = emb_ent[list(g_entities)]
        graph_cache[file_path] = (emb_ent_g, g_ent_freq)  # 필요한 경우 관계 임베딩/빈도도 추가 가능

    # ------------------- 4) 각 (original, modified) 쌍에 대해 점수 계산 -------------------
    for mfile in tqdm(modified_files, desc=f"Processing {method}"):
        modified_name = os.path.basename(mfile)
        results = []

        emb_ent_mod, freq_ent_mod = graph_cache[mfile]

        for ofile in original_files:
            emb_ent_orig, freq_ent_orig = graph_cache[ofile]
            sim = scoring(method, emb_ent_orig, freq_ent_orig,
                          emb_ent_mod, freq_ent_mod)

            results.append({
                "original": os.path.basename(ofile),
                "similarity_score": float(sim)
            })

        out_file = os.path.join(output_folder, f"{modified_name}_results.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"All comparisons completed for {method} in {output_folder}.")

def ingram_scoring(model, dataset, modification, option, method):
    """
    dataset: "c4_dataset" or "wikitext"
    modification: e.g., "synonym_replacement", "context_replacement", ...
    option: e.g., "0.3", "0.6", ...
    method: "pairs", "average", "weighted_average"
    """
    if dataset == "c4_dataset":
        data_folder = "c4_subset_graph"
        result_folder = "C4"
    elif dataset == "wikitext":
        data_folder = "wikitext_graph_data"
        result_folder = "Wikitext"
    elif dataset == "cc_news":
        data_folder = "cc_news_graph"
        result_folder = "CC_news"
    elif dataset == "book":
        data_folder = "bookcorpus_graph"
        result_folder = "Book"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    original_folder = os.path.join(data_folder, "original_graph")
    modified_folder = os.path.join(data_folder, modification, option)
    output_folder = os.path.join("Result",result_folder, modification, option, "ingram", method)

    evaluate_json_folders(method, original_folder, modified_folder, output_folder,
                          model, DIMENSION_ENTITY, DIMENSION_RELATION, NUM_BIN)

# ============================ 메인 루틴 (예시) ============================
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

    datasets = ["wikitext", "c4_dataset","cc_news","book"]
    methods = ["pairs", "average", "weighted_average"]
    modifications = ["synonym_replacement", "context_replacement", "dipper_paraphraser"]
    modification_options = {
        "synonym_replacement": ["0.3", "0.6"],
        "context_replacement": ["0.3", "0.6"],
        "dipper_paraphraser": ["60_0", "60_20"]
    }

    datasets = ["book"]
    method = ["pairs"]
    # Just an example usage:
    for ds in datasets:
        for mod in modifications:
            for opt in modification_options[mod]:
                for m in method:
                    ingram_scoring(my_model, ds, mod, opt, m)
