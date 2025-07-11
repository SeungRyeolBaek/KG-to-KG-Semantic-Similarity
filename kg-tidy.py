import os, json
from tqdm import tqdm

# --------- 유틸 ----------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# --------- 핵심 로직 ----------
def tidy_graph(data):
    """nodes / relationships 일관성 정리 → 변경 여부, 수정된 data 반환"""
    nodes = data.get("nodes", [])
    rels  = data.get("relationships", [])

    # 현재 node 집합
    id2node = {n["id"]: n for n in nodes if "id" in n}

    # 관계에 등장한 ID 집합
    used_ids = set()
    for e in rels:
        s = e.get("source", {}).get("id")
        t = e.get("target", {}).get("id")
        if s: used_ids.add(s)
        if t: used_ids.add(t)

    added, removed = 0, 0

    # 1) 누락 노드 추가
    for nid in used_ids:
        if nid not in id2node:
            id2node[nid] = {"id": nid}
            added += 1

    # 2) 고아 노드 제거
    for nid in list(id2node.keys()):
        if nid not in used_ids:
            del id2node[nid]
            removed += 1

    if added or removed:
        data["nodes"] = list(id2node.values())
        return True, added, removed, data
    return False, 0, 0, data

# --------- 폴더 순회 ----------
def tidy_all_graphs(root_dir):
    changed_cnt = 0
    for dirpath, _, files in os.walk(root_dir):
        json_files = [f for f in files if f.endswith(".json")]
        for jf in tqdm(json_files, desc=f"[{dirpath}]", leave=False):
            fp = os.path.join(dirpath, jf)
            try:
                data = load_json(fp)
                changed, add_n, del_n, new_data = tidy_graph(data)
                if changed:
                    save_json(fp, new_data)
                    changed_cnt += 1
                    print(f"✏️  {fp}: +{add_n}  -{del_n}")
            except Exception as e:
                print(f"⚠️  {fp} 읽기 실패: {e}")
    print(f"\n✨ Done. {changed_cnt} file(s) updated.")

# --------- 실행 ----------
if __name__ == "__main__":
    ROOT = "bookcorpus_graph"   # ← 그래프 파일이 들어 있는 최상위 폴더
    tidy_all_graphs(ROOT)
