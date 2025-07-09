import os
import json
from collections import defaultdict

# JSON 파일에서 데이터를 읽어오는 함수
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# JSON 파일로 데이터를 저장하는 함수
def write_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# 각 항목에 순위를 매겨주는 함수
def assign_ranks_to_json(json_file):
    # JSON 데이터 읽어오기
    data = read_json(json_file)

    # 점수에 따라 순위를 매기는 로직
    def assign_ranks(data):
        # 점수별로 항목을 그룹화
        score_to_items = defaultdict(list)
        for item in data:
            score = item['similarity_score']
            score_to_items[score].append(item)

        # 점수를 내림차순으로 정렬
        unique_scores = sorted(score_to_items.keys(), reverse=True)

        rank = 1
        for score in unique_scores:
            items = score_to_items[score]
            num_items = len(items)
            # 같은 점수를 가진 항목들에게 가장 낮은 등수 부여
            for item in items:
                item['rank'] = rank + num_items - 1  # 가장 낮은 등수를 할당
            # 다음 순위를 업데이트
            rank += num_items

        # 데이터 전체를 rank 기준으로 오름차순 정렬
        data.sort(key=lambda x: x['rank'])
        return data

    # 순위를 매긴 데이터
    ranked_data = assign_ranks(data)

    # 순위가 매겨진 데이터를 원래 파일에 덮어쓰기
    write_json(ranked_data, json_file)

    print(f"Ranked data saved to {json_file}")

# 주어진 폴더 내 모든 JSON 파일을 찾아 순위 매기는 함수 (재귀적)
def process_json_files_in_folder(folder_path):
    # 폴더를 재귀적으로 탐색
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                # JSON 파일 경로
                json_file = os.path.join(root, file)
                print(f"Processing {json_file}")
                # 각 JSON 파일에 대해 순위 매기기
                assign_ranks_to_json(json_file)

# 실행을 위한 메인 함수
if __name__ == "__main__":
    folder_path = 'Result' 
    process_json_files_in_folder(folder_path)