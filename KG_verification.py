import os
import json

# JSON 로딩
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 빈 그래프 판별
def is_empty_graph(data):
    nodes = data.get("nodes", [])
    edges = data.get("relationships", [])

    if len(nodes) == 0:
        return True

    node_ids = {node["id"] for node in nodes if "id" in node}

    valid_edges = []
    for edge in edges:
        source_id = edge.get("source", {}).get("id")
        target_id = edge.get("target", {}).get("id")
        if source_id in node_ids and target_id in node_ids:
            valid_edges.append(edge)

    return len(valid_edges) == 0

# 폴더 트리 전체를 탐색하면서 빈 그래프 탐색
def find_all_empty_graphs(root_folder):
    empty_graphs_by_folder = {}

    for dirpath, _, filenames in os.walk(root_folder):
        json_files = [f for f in filenames if f.endswith('.json')]
        if not json_files:
            continue

        empty_graphs = []
        for filename in json_files:
            file_path = os.path.join(dirpath, filename)
            try:
                data = load_json(file_path)
                if is_empty_graph(data):
                    empty_graphs.append(filename)
            except Exception as e:
                print(f"Failed to read {file_path}: {e}")
        
        if empty_graphs:
            empty_graphs_by_folder[dirpath] = empty_graphs

    return empty_graphs_by_folder

# 결과 출력
def print_empty_graphs_by_folder(empty_graphs_by_folder):
    if not empty_graphs_by_folder:
        print("✅ No empty graphs found.")
        return

    print("🚨 Empty graphs found in the following folders:\n")
    for folder, files in empty_graphs_by_folder.items():
        print(f"[{folder}]")

        # 숫자 기준 정렬
        try:
            sorted_files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))
        except ValueError:
            # 정수로 변환 안 되면 문자열 기준 fallback
            sorted_files = sorted(files)

        for fname in sorted_files:
            print(f"  - {fname}")
        print("")

# 실행 예시
if __name__ == "__main__":
    ROOT_DIR = "bookcorpus_graph"  # 사용자가 지정한 루트 폴더
    results = find_all_empty_graphs(ROOT_DIR)
    print_empty_graphs_by_folder(results)
