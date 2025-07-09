import os
import json
import re
from tqdm import tqdm

def count_entities_relations(json_folder):
    """
    주어진 JSON 폴더에서 전체 고유 엔티티 수와 관계 유형 수를 계산하는 함수.
    - 엔티티는 "nodes"에서 고유한 ID를 기준으로 집계
    - 관계 유형은 "relationships"에서 "type" 값을 기준으로 집계
    """
    total_entities = set()
    total_relations = set()

    json_files = [os.path.join(json_folder, file) for file in os.listdir(json_folder) if file.endswith(".json")]

    for json_file in tqdm(json_files, desc="Processing KG JSON Files"):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # "nodes"에 있는 모든 엔티티 ID를 추가
            if "nodes" in data:
                for node in data["nodes"]:
                    if "id" in node:
                        total_entities.add(node["id"])

            # "relationships"에서 관계 타입을 추가
            if "relationships" in data:
                for rel in data["relationships"]:
                    if "type" in rel:
                        total_relations.add(rel["type"])

    return len(total_entities), len(total_relations)

def count_tokens(txt_folder):
    """
    주어진 TXT 폴더에서 전체 토큰 수를 계산하는 함수
    """
    total_tokens = 0
    txt_files = [os.path.join(txt_folder, file) for file in os.listdir(txt_folder) if file.endswith(".txt")]

    for txt_file in tqdm(txt_files, desc="Processing Text Files"):
        with open(txt_file, 'r', encoding='utf-8') as f:
            text = f.read()
            tokens = re.findall(r'\w+', text)  # 간단한 토큰화 (단어 단위로 추출)
            total_tokens += len(tokens)

    return total_tokens

def compare_kg_text(json_folder, txt_folder):
    """
    KG 폴더와 텍스트 폴더를 비교하여 총 엔티티 수, 관계 수, 토큰 수를 출력하는 함수
    """
    total_entities, total_relations = count_entities_relations(json_folder)
    total_tokens = count_tokens(txt_folder)

    print("\n===== Comparison Results =====")
    print(f"Total Entities in KG: {total_entities}")
    print(f"Total Relations in KG: {total_relations}")
    print(f"Total Tokens in Text Data: {total_tokens}")

# 사용 예시
kg_folder = "wikitext_graph_data/original_graph"  # KG JSON 폴더 경로
text_folder = "wikitext_text_data/original_text"  # 텍스트 문서 폴더 경로
compare_kg_text(kg_folder, text_folder)