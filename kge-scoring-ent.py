import os
import json
from tqdm import tqdm
import torch
import numpy as np
import random
from collections import Counter
import gc
import time


# PyG의 TransE 모델 (PyG >= 2.3)
from torch_geometric.nn.kge import TransE, ComplEx, DistMult, RotatE
from torch.utils.data import DataLoader

# 유사도 계산
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

###############################################################################
# 전역 캐시 (파일 경로 -> 넘파이 배열)
###############################################################################
ent_emb_cache = {}

def build_cache(file_to_triples, ent_embeddings):
    """
    모든 파일에 대해 엔티티·릴레이션 임베딩을 미리 인덱싱해
    전역변수 ent_emb_cache, rel_emb_cache에 저장.
    """
    global ent_emb_cache, rel_emb_cache

    for fpath, triplets in file_to_triples.items():
        entity_set = set()
        relation_set = set()

        for (h, r, t) in triplets:
            entity_set.add(h)
            entity_set.add(t)

        if len(entity_set) > 0:
            ent_emb_arr = ent_embeddings[list(entity_set)]
        else:
            ent_emb_arr = np.zeros((0, ent_embeddings.shape[1]))

        ent_emb_cache[fpath] = ent_emb_arr

###############################################################################
# 1) Data Preprocessing: 여러 폴더에서 JSON 읽어 (h, r, t) + 전역 ent2id, rel2id
###############################################################################
def load_kgs_and_build_dicts(folder_paths):
    """
    folder_paths: KG가 들어있는 여러 폴더 경로 리스트
    반환: 
      - ent2id, rel2id: 전체 엔티티/릴레이션 문자열 -> ID 매핑
      - all_triplets: 전체 (h, r, t) (전역 ID로 변환된 것) 리스트
      - file_to_triples: {파일경로: [(h, r, t), ...]} 형태
    """
    ent2id = {}
    rel2id = {}
    all_triplets = []
    file_to_triples = {}

    ent_cnt = 0
    rel_cnt = 0

    for folder in folder_paths:
        for root, _, files in os.walk(folder):
            for fname in files:
                if not fname.endswith('.json'):
                    continue
                fpath = os.path.join(root, fname)
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                triples_in_this_file = []
                for rel_obj in data["relationships"]:
                    head = rel_obj["source"]["id"]
                    relation = rel_obj["type"]
                    tail = rel_obj["target"]["id"]

                    # 엔티티 ID 할당
                    if head not in ent2id:
                        ent2id[head] = ent_cnt
                        ent_cnt += 1
                    if tail not in ent2id:
                        ent2id[tail] = ent_cnt
                        ent_cnt += 1
                    # 릴레이션 ID 할당
                    if relation not in rel2id:
                        rel2id[relation] = rel_cnt
                        rel_cnt += 1

                    h_id = ent2id[head]
                    r_id = rel2id[relation]
                    t_id = ent2id[tail]

                    all_triplets.append((h_id, r_id, t_id))
                    triples_in_this_file.append((h_id, r_id, t_id))

                file_to_triples[fpath] = triples_in_this_file

    return ent2id, rel2id, all_triplets, file_to_triples

###############################################################################
# 2) PyG TransE 학습을 위한 Dataset, negative sampling 등
###############################################################################
class MyKGDataset(torch.utils.data.Dataset):
    def __init__(self, triplets):
        super().__init__()
        self.triplets = triplets
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        return self.triplets[idx]  # (h, r, t)

def negative_sampling(batch, num_entities, device, num_neg=1):
    """
    GPU 상에서 head 또는 tail을 무작위로 바꿔서 negative sample 생성.
    batch: (B, 3) tensor on device
    return: (B, 3) negative sample tensor on device
    """
    B = batch.size(0)
    corrupted = batch.clone()
    corrupt_head = torch.rand(B, device=device) < 0.5
    rand_ent = torch.randint(0, num_entities, (B,), device=device)
    corrupted[:, 0] = torch.where(corrupt_head, rand_ent, corrupted[:, 0])
    corrupted[:, 2] = torch.where(~corrupt_head, rand_ent, corrupted[:, 2])
    return corrupted


###############################################################################
# 3) 모델 학습 함수 (TransE, RotatE, DistMult, ComplEx)
###############################################################################
def train_transe_model(all_triplets, num_entities, num_relations,
                       learning_rate=0.01, distance_type='L1',
                       emb_dim=32, margin=1.0, epochs=10,
                       batch_size=128, early_stop_patience=5):
    dataset = MyKGDataset(all_triplets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=lambda b: torch.tensor(b, dtype=torch.long),
                        pin_memory=True, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TransE(
        num_nodes=num_entities,
        num_relations=num_relations,
        hidden_channels=emb_dim,
        margin=margin,
        p_norm=1 if distance_type == 'L1' else 2
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float('inf')
    patience = 0

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            neg_batch = negative_sampling(batch, num_entities, device)

            optimizer.zero_grad()
            pos_score = model(batch[:, 0], batch[:, 1], batch[:, 2])
            neg_score = model(neg_batch[:, 0], neg_batch[:, 1], neg_batch[:, 2])
            loss = torch.relu(margin + pos_score - neg_score).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {ep+1}/{epochs} | TransE Loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience = 0
        else:
            patience += 1
            if patience >= early_stop_patience:
                print(f"Early stopping at epoch {ep+1} | Best Loss={best_loss:.4f}")
                break

    return model

def train_rotate_model(all_triplets, num_entities, num_relations,
                       learning_rate=0.01, emb_dim=32, margin=1.0,
                       epochs=10, batch_size=128, early_stop_patience=5):
    dataset = MyKGDataset(all_triplets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=lambda b: torch.tensor(b, dtype=torch.long),
                        pin_memory=True, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = RotatE(
        num_nodes=num_entities,
        num_relations=num_relations,
        hidden_channels=emb_dim,
        margin=margin
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float('inf')
    patience = 0

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            neg_batch = negative_sampling(batch, num_entities, device)

            optimizer.zero_grad()
            pos_score = model(batch[:, 0], batch[:, 1], batch[:, 2])
            neg_score = model(neg_batch[:, 0], neg_batch[:, 1], neg_batch[:, 2])
            loss = torch.relu(margin + pos_score - neg_score).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {ep+1}/{epochs} | RotatE Loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience = 0
        else:
            patience += 1
            if patience >= early_stop_patience:
                print(f"Early stopping at epoch {ep+1} | Best Loss={best_loss:.4f}")
                break

    return model

def train_distmult_model(all_triplets, num_entities, num_relations,
                         learning_rate=0.01, emb_dim=32, margin=1.0,
                         epochs=10, batch_size=128, early_stop_patience=5):
    dataset = MyKGDataset(all_triplets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=lambda b: torch.tensor(b, dtype=torch.long),
                        pin_memory=True, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DistMult(
        num_nodes=num_entities,
        num_relations=num_relations,
        hidden_channels=emb_dim,
        margin=margin
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float('inf')
    patience = 0

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            neg_batch = negative_sampling(batch, num_entities, device)

            optimizer.zero_grad()
            pos_score = model(batch[:, 0], batch[:, 1], batch[:, 2])
            neg_score = model(neg_batch[:, 0], neg_batch[:, 1], neg_batch[:, 2])
            loss = torch.relu(margin + pos_score - neg_score).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {ep+1}/{epochs} | DistMult Loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience = 0
        else:
            patience += 1
            if patience >= early_stop_patience:
                print(f"Early stopping at epoch {ep+1} | Best Loss={best_loss:.4f}")
                break

    return model

def train_complex_model(all_triplets, num_entities, num_relations,
                        learning_rate=0.01, emb_dim=32, margin=1.0,
                        epochs=10, batch_size=128, early_stop_patience=5):
    dataset = MyKGDataset(all_triplets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=lambda b: torch.tensor(b, dtype=torch.long),
                        pin_memory=True, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ComplEx(
        num_nodes=num_entities,
        num_relations=num_relations,
        hidden_channels=emb_dim
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float('inf')
    patience = 0

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            neg_batch = negative_sampling(batch, num_entities, device)

            optimizer.zero_grad()
            pos_score = model(batch[:, 0], batch[:, 1], batch[:, 2])
            neg_score = model(neg_batch[:, 0], neg_batch[:, 1], neg_batch[:, 2])
            loss = torch.relu(margin + pos_score - neg_score).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {ep+1}/{epochs} | ComplEx Loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience = 0
        else:
            patience += 1
            if patience >= early_stop_patience:
                print(f"Early stopping at epoch {ep+1} | Best Loss={best_loss:.4f}")
                break

    return model

def train_kge_model(model_name, all_triplets, num_entities, num_relations,
                    learning_rate=0.01, distance_type='L1', emb_dim=32,
                    margin=1.0, epochs=10, batch_size=128, early_stop_patience=5):
    if model_name == "TransE":
        return train_transe_model(
            all_triplets=all_triplets,
            num_entities=num_entities,
            num_relations=num_relations,
            learning_rate=learning_rate,
            emb_dim=emb_dim,
            margin=margin,
            epochs=epochs,
            batch_size=batch_size,
            early_stop_patience=early_stop_patience
        )
    elif model_name == "RotatE":
        return train_rotate_model(
            all_triplets=all_triplets,
            num_entities=num_entities,
            num_relations=num_relations,
            learning_rate=learning_rate,
            emb_dim=emb_dim,
            margin=margin,
            epochs=epochs,
            batch_size=batch_size,
            early_stop_patience=early_stop_patience
        )
    elif model_name == "DistMult":
        return train_distmult_model(
            all_triplets=all_triplets,
            num_entities=num_entities,
            num_relations=num_relations,
            learning_rate=learning_rate,
            emb_dim=emb_dim,
            margin=margin,
            epochs=epochs,
            batch_size=batch_size,
            early_stop_patience=early_stop_patience
        )
    elif model_name == "ComplEx":
        return train_complex_model(
            all_triplets=all_triplets,
            num_entities=num_entities,
            num_relations=num_relations,
            learning_rate=learning_rate,
            emb_dim=emb_dim,
            margin=margin,
            epochs=epochs,
            batch_size=batch_size,
            early_stop_patience=early_stop_patience
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

###############################################################################
# 4) 그래프별 임베딩 비교: average
###############################################################################
def compute_entity_avg_embedding(file_path, file_to_triples, ent_embeddings):
    triplets = file_to_triples[file_path]
    ent_ids = set()
    for (h, r, t) in triplets:
        ent_ids.add(h)
        ent_ids.add(t)
    if len(ent_ids) == 0:
        return None
    emb_list = [ent_embeddings[e] for e in ent_ids]
    emb_list = np.array(emb_list)
    return emb_list.mean(axis=0)

def compare_graphs_by_average(fileA, fileB, file_to_triples, ent_embeddings):
    entA = compute_entity_avg_embedding(fileA, file_to_triples, ent_embeddings)
    entB = compute_entity_avg_embedding(fileB, file_to_triples, ent_embeddings)

    # 엔티티가 없으면 0.0, 있으면 코사인 유사도
    if entA is None or entB is None:
        ent_sim = 0.0
    else:
        ent_sim = 1 - cosine(entA, entB)

    return ent_sim

###############################################################################
# (B) pairwise 매트릭스에서 max + threshold 비율 (캐시 사용)
###############################################################################
THRESHOLD = 0.8
def compare_graphs_by_pairs_counting(fileA, fileB, ent_embeddings, threshold=0.7):
    global ent_emb_cache

    emb_ent1 = torch.tensor(ent_emb_cache[fileA], dtype=torch.float32, device='cuda')  # GPU 로드
    emb_ent2 = torch.tensor(ent_emb_cache[fileB], dtype=torch.float32, device='cuda')  # GPU 로드

    # joint_mean = torch.cat([emb_ent1, emb_ent2], dim=0).mean(0, keepdim=True)
    # emb_ent1 = emb_ent1 - joint_mean        # mean-centering
    # emb_ent2 = emb_ent2 - joint_mean    

    # 코사인 유사도 계산
    emb_ent1 = torch.nn.functional.normalize(emb_ent1, dim=1)
    emb_ent2 = torch.nn.functional.normalize(emb_ent2, dim=1)
    ent_sim_matrix = torch.matmul(emb_ent1, emb_ent2.T)  # 코사인 유사도 계산

    # 가장 높은 유사도 찾기
    if emb_ent1.shape[0] < emb_ent2.shape[0]:
        max_ent_similarities, _ = ent_sim_matrix.max(dim=1)
    else:
        max_ent_similarities, _ = ent_sim_matrix.max(dim=0)

    # Threshold 이상 비율 계산
    ratio = torch.mean((max_ent_similarities > threshold).float()).item()
    return ratio

###############################################################################
# (C) weighted average 비교
###############################################################################
def compute_entity_weighted_avg_embedding(file_path, file_to_triples, ent_embeddings):
    triplets = file_to_triples[file_path]
    if len(triplets) == 0:
        return None
    
    ent_freq = Counter()
    for (h, r, t) in triplets:
        ent_freq[h] += 1
        ent_freq[t] += 1

    if len(ent_freq) == 0:
        return None
    
    ent_ids = list(ent_freq.keys())
    weights = np.array([ent_freq[e] for e in ent_ids])
    emb_list = np.array([ent_embeddings[e] for e in ent_ids])

    weighted_sum = (emb_list.T * weights).T.sum(axis=0)
    weighted_avg = weighted_sum / weights.sum()

    return weighted_avg
def compare_graphs_by_weighted_average(fileA, fileB, file_to_triples, ent_embeddings):
    entA = compute_entity_weighted_avg_embedding(fileA, file_to_triples, ent_embeddings)
    entB = compute_entity_weighted_avg_embedding(fileB, file_to_triples, ent_embeddings)
	
    if entA is None or entB is None:
        ent_sim = 0.0
    else:
        ent_sim = 1 - cosine(entA, entB)

    return ent_sim

###############################################################################
# 7) "두 그래프" 비교 함수: 평균 vs pairs vs weighted
###############################################################################
elapsed_time = 0

def compute_graph_similarity(fileA, fileB,
                             file_to_triples,
                             ent_embeddings,
                             method='average'):
    global elapsed_time
    if method == 'average':
        start_time = time.perf_counter()  # ⏱ 시작 시간 기록
        result = compare_graphs_by_average(
            fileA, fileB,
            file_to_triples,
            ent_embeddings
        )
        end_time = time.perf_counter()  # ⏱ 종료 시간 기록
        elapsed_time += end_time - start_time  # ⏱ 경과 시간 계산
        return result
    elif method == 'pairs':
        start_time = time.perf_counter()  # ⏱ 시작 시간 기록
        result = compare_graphs_by_pairs_counting(
            fileA, fileB,
            ent_embeddings,
            threshold=THRESHOLD
        )
        end_time = time.perf_counter()  # ⏱ 종료 시간 기록
        elapsed_time += end_time - start_time  # ⏱ 경과 시간 계산
        return result
    elif method == 'weighted average':
        start_time = time.perf_counter()  # ⏱ 시작 시간 기록
        result = compare_graphs_by_weighted_average(
            fileA, fileB,
            file_to_triples,
            ent_embeddings
        )
        end_time = time.perf_counter()  # ⏱ 종료 시간 기록
        elapsed_time += end_time - start_time  # ⏱ 경과 시간 계산
        return result
    else:
        raise ValueError(f"Unknown method: {method}")

###############################################################################
# 8) 폴더별 파일 쌍 비교 + JSON 결과 저장
###############################################################################
def evaluate_json_folders(original_folder, modified_folder, output_folder,
                          file_to_triples,
                          ent_embeddings,
                          method='average'):
    """
    original_folder, modified_folder: 각각 json이 들어있는 폴더
    output_folder: 결과 저장 폴더
    file_to_triples: {파일경로: [(h,r,t), ...]}
    ent_embeddings, rel_embeddings: (num_ent, d), (num_rel, d)
    method: 'average' or 'pairs' or 'weighted average'
    """

    # original, modified 폴더 내 모든 json 파일 목록
    original_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(original_folder)
        for file in files if file.endswith('.json')
    ]
    modified_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(modified_folder)
        for file in files if file.endswith('.json')
    ]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print(f"processing for {output_folder}.")
    for mod_file in tqdm(modified_files, desc="Processing modified files"):
        mod_basename = os.path.basename(mod_file)
        out_path = os.path.join(output_folder, f"{mod_basename}_results.json")

        # 이미 결과 파일이 존재하면 스킵
        if os.path.exists(out_path):
            continue

        results = []
        for orig_file in original_files:
            similarity_score = compute_graph_similarity(
                orig_file, mod_file,
                file_to_triples,
                ent_embeddings,
                method=method
            )
            results.append({
                "original": os.path.basename(orig_file),
                "similarity_score": float(similarity_score)
            })

        # 결과 저장
        with open(out_path, 'w', encoding='utf-8') as fout:
            json.dump(results, fout, ensure_ascii=False, indent=4)
    print(f"{elapsed_time}s processed")
    print(f"All comparisons completed and results saved in {output_folder}.")

###############################################################################
# 9) main example
###############################################################################
def kge_scoring(dataset, text_modification, modification_option, model_name, method):
    # Data
    if dataset == 'bookcorpus_graph':
        output = 'Book'
    elif dataset == 'wikitext_graph_data':
        output = 'Wikitext'
    elif dataset == 'cc_news_graph':
        output = 'CC_news'
    else:
        # 필요에 따라 default 처리
        output = 'results'

    original_folder = os.path.join(dataset, 'original_graph')
    modified_folder = os.path.join(dataset, text_modification, modification_option)

    # 결과 저장 폴더
    output_folder = os.path.join(
        "Result", output, text_modification, modification_option,
        'kge', model_name , method
    )

    # (1) 폴더에서 JSON 로딩 → ent2id, rel2id 매핑, 전체 all_triplets
    ent2id, rel2id, all_triplets, file_to_triples = load_kgs_and_build_dicts(
        [original_folder, modified_folder]
    )

    num_entities = len(ent2id)
    num_relations = len(rel2id)

    # 하이퍼파라미터 설정
    emb_dim = 32
    margin = 2.0
    epochs = 1000
    batch_size = 128
    learning_rate = 0.01
    early_stop_patience = 10

    ############################################################################
    # 모델 체크: 이전에 동일한 (dataset, text_modification, option, model_name)
    # 조건으로 학습한 모델이 있는지 확인
    ############################################################################
    model_save_dir = "kge_models"
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    # 모델 파일 이름: 주요 하이퍼파라미터까지 포함하면 더 안전
    model_save_path = os.path.join(
        model_save_dir,
        f"{dataset}_{text_modification}_{modification_option}_{model_name}.pt"
    )

    # 2) KGE 모델 정의 (학습 or 로드)
    if os.path.exists(model_save_path):
        # 기존 모델이 있으므로 로드만 하고, 학습은 스킵
        print(f"[Load Model] Found existing model at {model_save_path}. Loading...")
        # 아래처럼 모델 구조를 먼저 정의한 뒤 load_state_dict() 해야 함
        if model_name == "TransE":
            model = TransE(
                num_nodes=num_entities,
                num_relations=num_relations,
                hidden_channels=emb_dim,
                margin=margin,
                p_norm=1  # distance_type='L1' 가정
            )
        elif model_name == "RotatE":
            model = RotatE(
                num_nodes=num_entities,
                num_relations=num_relations,
                hidden_channels=emb_dim,
                margin=margin
            )
        elif model_name == "DistMult":
            model = DistMult(
                num_nodes=num_entities,
                num_relations=num_relations,
                hidden_channels=emb_dim,
                margin=margin
            )
        elif model_name == "ComplEx":
            model = ComplEx(
                num_nodes=num_entities,
                num_relations=num_relations,
                hidden_channels=emb_dim
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # CPU 로드 가정 (필요 시 cuda 로드)
        model.load_state_dict(torch.load(model_save_path, map_location='cpu'))
        print("[Load Model] Successfully loaded the saved state_dict.")
    else:
        # 모델이 없으므로 새로 학습
        print(f"[Train Model] No existing model found at {model_save_path}. Training new model...")
        model = train_kge_model(
            model_name=model_name,
            all_triplets=all_triplets,
            num_entities=num_entities,
            num_relations=num_relations,
            learning_rate=learning_rate,
            emb_dim=emb_dim,
            margin=margin,
            epochs=epochs,
            batch_size=batch_size,
            early_stop_patience=early_stop_patience
        )
        # 학습 완료 후 저장
        torch.save(model.state_dict(), model_save_path)
        print(f"[Train Model] Model saved at {model_save_path}.")

    # (3) 임베딩 추출
    entity_embeddings = model.node_emb.weight.detach().cpu().numpy()   # shape=(num_entities, emb_dim)

    # (4) === 전역 캐시 빌드 ===
    build_cache(file_to_triples, entity_embeddings)

    # (5) 폴더별 파일 쌍 전수 비교 → JSON 저장
    evaluate_json_folders(
        original_folder,
        modified_folder,
        output_folder,
        file_to_triples,
        entity_embeddings,
        method=method
    )
if __name__ == "__main__":
    # 'wikitext_graph_data' or 'cc_news_graph' or 'bookcorpus_graph'
    dataset_name = ['cc_news_graph','wikitext_graph_data']
    dataset_name = ['wikitext_graph_data']
    # context_replacement or synonym_replacement or dipper_paraphraser
    text_modification = ['synonym_replacement','context_replacement','dipper_paraphraser']
    # '0.3' or '0.6' or '60_0' or '60_20' 
    modification_option = {
        'synonym_replacement' : ['0.3','0.6'],
        'context_replacement' : ['0.3','0.6'],
        'dipper_paraphraser' : ['60_0','60_20']
    }
    # KGE 모델 종류: ["TransE", "RotatE", "ComplEx", "DistMult"]
    model_name = ["TransE", "DistMult", "ComplEx", "RotatE"]
    model_name = ["RotatE"]
    # 그래프 비교 방식: 'pairs', 'average', 'weighted average'
    methods = ['average']
    for dataset in dataset_name:
        for modification in text_modification:
            options = modification_option[modification]
            for option in options:
                for model in model_name:
                    for method in methods:
                        print(f"mod:{modification}")
                        print(f"option:{option}")
                        kge_scoring(dataset, modification, option, model, method)
            del ent_emb_cache
            ent_emb_cache = {}
            # GPU 및 메모리 정리
            gc.collect()
            torch.cuda.empty_cache()
