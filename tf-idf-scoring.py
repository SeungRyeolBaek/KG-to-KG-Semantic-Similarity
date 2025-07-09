import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import time
elapsed_time = 0

# 모든 텍스트를 한 번에 벡터화하고 유사도를 계산하는 함수
def compute_all_similarities(original_folder, modified_folder, output_folder):
    original_files = [os.path.join(root, file) for root, _, files in os.walk(original_folder) for file in files if file.endswith('.txt')]
    modified_files = [os.path.join(root, file) for root, _, files in os.walk(modified_folder) for file in files if file.endswith('.txt')]

    # 모든 파일의 텍스트를 모아서 한 번에 벡터화
    all_texts = []
    all_file_paths = []

    for file_path in original_files + modified_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            all_texts.append(f.read())
            all_file_paths.append(file_path)

    global elapsed_time
    # TF-IDF 벡터화 (한 번에 모든 텍스트에 대해)
    vectorizer = TfidfVectorizer(stop_words='english')
    start_time = time.perf_counter()  # ⏱ 시작 시간 기록
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    end_time = time.perf_counter()  # ⏱ 종료 시간 기록
    elapsed_time += end_time - start_time  # ⏱ 경과 시간 계산

    # 각 파일의 벡터를 구분하여 저장
    original_vectors = tfidf_matrix[:len(original_files)]
    modified_vectors = tfidf_matrix[len(original_files):]

    # 텍스트 파일 간의 유사도 계산
    for i, modified_file_path in enumerate(tqdm(modified_files, desc="Comparing text files")):
        modified_file_name = os.path.basename(modified_file_path)
        modified_file_name = modified_file_name.replace('.txt', '')  # .txt 제거

        results = []
        for j, original_file_path in enumerate(original_files):
            original_file_name = os.path.basename(original_file_path)
            original_file_name= original_file_name.replace('.txt', '.json')  # .txt 제거

            start_time = time.perf_counter()  # ⏱ 시작 시간 기록            
            # 코사인 유사도 계산
            similarity_score = cosine_similarity(modified_vectors[i], original_vectors[j])[0][0]
            end_time = time.perf_counter()  # ⏱ 종료 시간 기록
            elapsed_time += end_time - start_time  # ⏱ 경과 시간 계산

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

def tf_idf_scoring(dataset_name, text_modification, modification_option):
    global elapsed_time
    elapsed_time=0
    if dataset_name == 'c4_subset_graph_textualized':
        output_name = 'c4_result_text'
    if dataset_name == 'wikitext_graph_data_textualized':
        output_name = 'wikitext_result_text'

    original_folder = f'{dataset_name}/original_graph'
    modified_folder = f'{dataset_name}/{text_modification}/{modification_option}'
    output_folder = f'{output_name}/{text_modification}/{modification_option}/tf-idf'

    compute_all_similarities(original_folder, modified_folder, output_folder)

# 프로그램 실행
if __name__ == "__main__":
    # 'c4_subset_text' or 'wikitext_text_data' 
    dataset_name = ['wikitext_graph_data_textualized','c4_subset_graph_textualized']
    # context_replacement or synonym_replacement or dipper_paraphraser
    text_modification = ['synonym_replacement' ,'context_replacement','dipper_paraphraser']
    # '0.3' or '0.6' or '60_0' or '60_20' 
    modification_option = {'synonym_replacement' : ['0.3','0.6'] ,'context_replacement' : ['0.3','0.6'],'dipper_paraphraser' : ['60_0','60_20']} 


    for dataset in dataset_name:
        for modification in text_modification:
            options = modification_option[modification]
            for option in options:
                tf_idf_scoring(dataset,modification,option)