import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import time
elapsed_time = 0
# 사전 훈련된 Sentence Transformer 모델 로드
model = SentenceTransformer("all-mpnet-base-v2")

# 모든 텍스트를 한 번에 벡터화하고 유사도를 계산하는 함수
def compute_all_similarities(original_folder, modified_folder, output_folder):
    original_files = [os.path.join(root, file) for root, _, files in os.walk(original_folder) for file in files if file.endswith('.txt')]
    modified_files = [os.path.join(root, file) for root, _, files in os.walk(modified_folder) for file in files if file.endswith('.txt')]
    global elapsed_time

    # 모든 파일의 텍스트를 모아서 한 번에 임베딩
    all_texts = []
    all_file_paths = []

    for file_path in original_files + modified_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            all_texts.append(f.read())
            all_file_paths.append(file_path)

    start_time = time.perf_counter()  # ⏱ 시작 시간 기록
    # Sentence Transformer로 텍스트 임베딩 계산
    embeddings = model.encode(all_texts)
    # 임베딩 유사도 계산
    similarities = model.similarity(embeddings, embeddings)
    end_time = time.perf_counter()  # ⏱ 종료 시간 기록
    elapsed_time += end_time - start_time  # ⏱ 경과 시간 계산

    # 텍스트 파일 간의 유사도 계산 및 저장
    num_originals = len(original_files)
    for i, modified_file_path in enumerate(tqdm(modified_files, desc="Comparing text files")):
        modified_file_name = os.path.basename(modified_file_path).replace('.txt', '')  # .txt 제거

        results = []
        for j, original_file_path in enumerate(original_files):
            original_file_name = os.path.basename(original_file_path).replace('.txt', '.json')  # .txt 제거

            # 유사도 가져오기 및 float 변환
            similarity_score = float(similarities[num_originals + i][j])

            # 결과 저장
            results.append({
                "original": original_file_name,
                "similarity_score": np.round(similarity_score, 2)
            })

        # 비교 결과 저장
        save_results(output_folder, modified_file_name, results)
    print(f"{elapsed_time}s processed")

# 결과 저장을 위한 함수
def save_results(output_folder, modified_file_name, results):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    output_file = os.path.join(output_folder, f'{modified_file_name}.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

def sbert_scoring(dataset_name, text_modification, modification_option):
    global elapsed_time
    elapsed_time=0
    if dataset_name == 'bookcorpus_graph_verbalized':
        output_name = 'Book_text'
    if dataset_name == 'wikitext_graph_data_verbalized':
        output_name = 'wikitext_result_text'
    if dataset_name == 'cc_news_graph_verbalized':
        output_name = 'CC_news_result_text'
    original_folder = f'{dataset_name}/original_graph'
    modified_folder = f'{dataset_name}/{text_modification}/{modification_option}'
    output_folder = f'Result/{output_name}/{text_modification}/{modification_option}/sbert'

    compute_all_similarities(original_folder, modified_folder, output_folder)

# 프로그램 실행
if __name__ == "__main__":
    # 'wikitext_graph_data_verbalized' or 'c4_subset_graph_verbalized' or 'cc_news_graph_verbalized' or 'bookcorpus_graph_verbalized'
    dataset_name = ['bookcorpus_graph_verbalized']
    # context_replacement or synonym_replacement or dipper_paraphraser
    text_modification = ['synonym_replacement' ,'context_replacement','dipper_paraphraser']
    # '0.3' or '0.6' or '60_0' or '60_20' 
    modification_option = {'synonym_replacement' : ['0.3','0.6'] ,'context_replacement' : ['0.3','0.6'],'dipper_paraphraser' : ['60_0','60_20']} 

    for dataset in dataset_name:
        for modification in text_modification:
            options = modification_option[modification]
            for option in options:
                sbert_scoring(dataset,modification,option)

# 769*~~