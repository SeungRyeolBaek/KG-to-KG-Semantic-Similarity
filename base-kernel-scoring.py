from grakel import Graph
from grakel.kernels import VertexHistogram, EdgeHistogram
import os
import json
from tqdm import tqdm
import numpy as np
import time
elapsed_time = 0
# 엣지의 source, target이 노드에 없는 경우 해당 엣지 삭제, 엣지가 남아있지 않으면 빈 그래프로 간주
def is_empty_graph(data):
    nodes = data.get("nodes", [])
    edges = data.get("relationships", [])

    if len(nodes) == 0:
        return True

    node_ids = {node["id"] for node in nodes if "id" in node}
    valid_edges = [
        edge for edge in edges
        if edge.get("source", {}).get("id") in node_ids and edge.get("target", {}).get("id") in node_ids
    ]

    return len(valid_edges) == 0

def find_empty_graphs(folder_path):
    empty_graphs = []
    json_files = [os.path.join(root, file) for root, _, files in os.walk(folder_path) for file in files if file.endswith('.json')]
    
    for file_path in json_files:
        data = load_json(file_path)
        if is_empty_graph(data):
            empty_graphs.append(os.path.basename(file_path))
    return empty_graphs

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_graph_from_json(file_path):
    data = load_json(file_path)

    node_id_mapping = {}
    node_labels = {}
    edge_labels = {}

    for idx, node in enumerate(data.get("nodes", [])):
        node_id = node.get("id")
        node_type = node.get("type", "Unknown")
        if node_id is None:
            continue
        node_labels[idx] = node_id
        node_id_mapping[node_id] = idx

    edges = []
    for rel in data.get("relationships", []):
        source_id = rel.get("source", {}).get("id")
        target_id = rel.get("target", {}).get("id")
        edge_type = rel.get("type", "Unknown")

        if source_id is None or target_id is None:
            continue

        source_idx = node_id_mapping.get(source_id)
        target_idx = node_id_mapping.get(target_id)
        if source_idx is None or target_idx is None:
            continue

        edges.append((source_idx, target_idx))
        edge_labels[(source_idx, target_idx)] = edge_type

    if len(node_labels) == 0 or len(edges) == 0:
        return None

    return Graph(initialization_object=edges, node_labels=node_labels, edge_labels=edge_labels)

def load_graphs_from_folder(folder_path, empty_graphs=[]):
    graph_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.json') and file not in empty_graphs]
    graphs, graph_names = [], []
    
    for file_path in tqdm(graph_files, desc=f"Loading graphs from {folder_path}"):
        G = load_graph_from_json(file_path)
        if G is not None:
            graphs.append(G)
            graph_names.append(os.path.basename(file_path))
    
    return graphs, graph_names

def compute_similarity_and_save_results(graphs_A, graph_names_A, graphs_B, graph_names_B, output_folder, base_graph_kernel="VertexHistogram"):
    # 그래프 커널 초기화 (WL Kernel 제거)
    if base_graph_kernel == "VertexHistogram":
        kernel = VertexHistogram()
    elif base_graph_kernel == "EdgeHistogram":
        kernel = EdgeHistogram()
    else:
        raise ValueError("Unsupported base_graph_kernel.")
    global elapsed_time

    start_time = time.perf_counter()  # ⏱ 시작 시간 기록
    # 그래프 집합 A에 대해 커널 학습
    kernel.fit(graphs_A)
    end_time = time.perf_counter()  # ⏱ 종료 시간 기록
    elapsed_time += end_time - start_time  # ⏱ 경과 시간 계산

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for idx, graph_B in enumerate(tqdm(graphs_B, desc="Processing results")):
        modified_graph_name = graph_names_B[idx]
        results = []

        start_time = time.perf_counter()  # ⏱ 시작 시간 기록
        sim_vector = kernel.transform([graph_B])[0]
        end_time = time.perf_counter()  # ⏱ 종료 시간 기록
        elapsed_time += end_time - start_time  # ⏱ 경과 시간 계산
        for original_idx, sim_value in enumerate(sim_vector):
            results.append({
                "original": graph_names_A[original_idx],
                "similarity_score": float(sim_value)
            })

        output_file = os.path.join(output_folder, f"{modified_graph_name}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

def evaluate_json_folders(original_folder, suspected_folder, output_folder, base_graph_kernel="VertexHistogram"):
    empty_original = find_empty_graphs(original_folder)
    empty_modified = find_empty_graphs(suspected_folder)
    empty_graphs = set(empty_original + empty_modified)

    graphs_A, graph_names_A = load_graphs_from_folder(original_folder, empty_graphs=empty_graphs)
    graphs_B, graph_names_B = load_graphs_from_folder(suspected_folder, empty_graphs=empty_graphs)
    compute_similarity_and_save_results(graphs_A, graph_names_A, graphs_B, graph_names_B, output_folder, base_graph_kernel=base_graph_kernel)

    print(f"All comparisons completed and saved in {output_folder}.")
    print(f"{elapsed_time}s processed")

def base_kernel_scoring(graph_kernel, dataset_name, text_modification, modification_option):

    if dataset_name == 'c4_subset_graph':
        output_name = 'C4'
    if dataset_name == 'wikitext_graph_data':
        output_name = 'Wikitext'
    if dataset_name == 'cc_news_graph':
        output_name = 'CC_News'

    original_folder = dataset_name + '/original_graph'
    modified_folder = dataset_name + '/' + text_modification + '/' + modification_option
    output_folder = f'Result/{output_name}/{text_modification}/{modification_option}/base-kernel/{graph_kernel}'

    evaluate_json_folders(original_folder, modified_folder, output_folder, base_graph_kernel=graph_kernel)

if __name__ == "__main__":
    # "VertexHistogram" or "EdgeHistogram"
    base_graph_kernel = ["VertexHistogram"]
    # 'c4_subset_graph' or 'wikitext_graph_data' 
    dataset_name = ['cc_news_graph','wikitext_graph_data']
    # context_replacement or synonym_replacement or dipper_paraphraser
    text_modification = ['synonym_replacement' ,'context_replacement','dipper_paraphraser']
    # '0.3' or '0.6' or '60_0' or '60_20' 
    modification_option = {'synonym_replacement' : ['0.3','0.6'] ,
    'context_replacement' : ['0.3','0.6'],
    'dipper_paraphraser' : ['60_0','60_20']} 
    dataset_name = ['cc_news_graph']
    for dataset in dataset_name:
        for modification in text_modification:
            options = modification_option[modification]
            for option in options:
                for base_kernel in base_graph_kernel:
                    base_kernel_scoring(base_kernel,dataset,modification,option)