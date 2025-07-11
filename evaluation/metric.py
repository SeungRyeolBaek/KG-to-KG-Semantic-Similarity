import os
import json
import math

# 상위 k개의 항목을 가져오는 함수 (공동 순위는 무시하고 정확히 상위 k개의 항목만 선택)
def get_top_k_items(data, k):
    sorted_data = sorted(data, key=lambda x: x['rank'])
    return sorted_data[:k]

# DCG 계산 함수
def calculate_dcg(rank, relevance_score):
    return relevance_score / math.log2(rank + 1) if rank > 0 else 0

# IDCG 계산 함수 (정답 파일이 1등일 때의 이상적인 DCG)
def calculate_idcg(relevance_score):
    return relevance_score # 이상적인 순위는 1위이므로 DCG는 relevance_score / log2(1+1) = 1

def calculate_nsg(data, relevance_score):
    scores = [item['similarity_score'] for item in data]
    max_score = max(scores)
    min_score = min(scores)
    return (max_score - relevance_score) / (max_score - min_score) if max_score > min_score else 0

def calculate_err(data, top_k=10):
    sorted_data = sorted(data, key=lambda x: x['rank'])
    err = 0.0
    prob_not_satisfied = 1.0
    for r, item in enumerate(sorted_data[:top_k], start=1):
        R = item['similarity_score']  # 0~1 범위라고 가정
        err += prob_not_satisfied * R / r
        prob_not_satisfied *= (1 - R)
    return err

# 각 JSON 파일에서 metric을 계산하는 함수
def calculate_metrics(data, ground_truth):
    ground_truth_index = ground_truth.split('_')[0]
    ground_truth_item = next(
        (item for item in data if item['original'].split('_')[0] == ground_truth_index),
        None
    )

    if not ground_truth_item:
        return 0, 0, 0, 0, 0, 0, 0  # 모든 metric이 0

    rank = ground_truth_item['rank']
    relevance_score = ground_truth_item['similarity_score']

    # Hits@k
    hits_at_1 = 1 if rank <= 1 else 0
    hits_at_3 = 1 if rank <= 3 else 0
    hits_at_5 = 1 if rank <= 5 else 0

    # MRR
    mrr = 1.0 / rank

    # DCG / IDCG / NDCG
    dcg = relevance_score / math.log2(rank + 1) if rank > 0 else 0
    idcg = relevance_score  # 이상적 DCG
    ndcg = dcg / idcg if idcg > 0 else 0

    nsg = calculate_nsg(data, relevance_score)
    err = calculate_err(data)

    return hits_at_1, hits_at_3, hits_at_5, ndcg, mrr, nsg, err

# 폴더 내 모든 JSON 파일에 대해 metric을 계산하고 기록하는 함수
def process_metrics_for_folder(folder_path):
    result = {
        'Hits@1': 0,
        'Hits@3': 0,
        'Hits@5': 0,
        'NDCG': 0,
        'MRR': 0,
        'NSG': 0,
        'ERR': 0,
        'count': 0
    }

    # JSON 파일을 찾음
    json_files = [file for file in os.listdir(folder_path) if file.endswith('.json')]

    if not json_files:
        return  # 해당 폴더에 JSON 파일이 없으면 작업을 수행하지 않음

    # JSON 파일들을 처리
    for file in json_files:
        json_file = os.path.join(folder_path, file)

        # JSON 파일 읽기
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 각 파일의 이름에서 modified 그래프 이름의 _ 앞부분을 추출
        modified_name = file.split('_')[0]  # 파일명에서 첫 번째 언더스코어 앞의 인덱스를 추출

        # Metric 계산
        hits_at_1, hits_at_3, hits_at_5, ndcg, mrr, nsg, err = calculate_metrics(data, modified_name)

        # 결과 누적
        result['Hits@1'] += hits_at_1
        result['Hits@3'] += hits_at_3
        result['Hits@5'] += hits_at_5
        result['NDCG'] += ndcg
        result['MRR'] += mrr
        result['NSG'] += err
        result['ERR'] += err
        result['count'] += 1
            
    # 평균 계산
    if result['count'] > 0:
        result['Hits@1'] /= result['count']
        result['Hits@3'] /= result['count']
        result['Hits@5'] /= result['count']
        result['NDCG'] /= result['count']
        result['MRR'] /= result['count']
        result['NSG'] /= result['count']
        result['ERR'] /= result['count']

    # 결과를 파일로 저장
    folder_path = os.path.join("metric_result", folder_path)
    os.makedirs(folder_path, exist_ok=True)  # 디렉토리 생성 (이미 존재하면 무시)
    result_file = os.path.join(folder_path, "result.txt")
    with open(result_file, 'w') as f:
        # f.write(f"Hits@1: {result['Hits@1']}\n")
        # f.write(f"Hits@3: {result['Hits@3']}\n")
        f.write(f"Hits@5: {result['Hits@5']}\n")
        f.write(f"NDCG: {result['NDCG']}\n")
        f.write(f"MRR: {result['MRR']}\n")
        # f.write(f"NSG: {result['NSG']}\n")
        # f.write(f"ERR: {result['ERR']}\n")

    print(f"Metrics saved to {result_file}")

# 래퍼 함수: 폴더 내의 모든 하위 폴더에 대해 메트릭을 계산
def process_all_folders_in_directory(base_folder):
    for root, dirs, _ in os.walk(base_folder):
        for dir_name in dirs:
            folder_path = os.path.join(root, dir_name)
            process_metrics_for_folder(folder_path)

# 프로그램 실행
if __name__ == "__main__":
    folder_path = 'Result' 
    process_all_folders_in_directory(folder_path)
