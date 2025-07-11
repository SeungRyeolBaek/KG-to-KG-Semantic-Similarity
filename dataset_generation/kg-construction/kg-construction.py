import os
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
import json
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from tqdm import tqdm

# 1. 환경 변수 설정 및 OpenAI 연결
load_dotenv()

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# LLM 및 LLMGraphTransformer 초기화
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
llm_transformer = LLMGraphTransformer(llm=llm)

# 텍스트 파일이 저장된 디렉토리 및 JSON 저장 디렉토리 설정
input_dir = 'bookcorpus_text/original_text'
json_output_dir = 'bookcorpus_graph/original_graph'

os.makedirs(json_output_dir, exist_ok=True)

# Node 객체를 JSON으로 직렬화할 수 있도록 변환하는 함수
def serialize_node(node):
    return {
        "id": node.id,
        "type": node.type,
        "properties": node.properties
    }

# Relationship 객체를 JSON으로 직렬화할 수 있도록 변환하는 함수
def serialize_relationship(relationship):
    return {
        "source": serialize_node(relationship.source),
        "target": serialize_node(relationship.target),
        "type": relationship.type,
        "properties": relationship.properties
    }

# 텍스트를 청크로 나누는 함수
def chunk_text(text, max_tokens):
    words = text.split()
    chunks = []
    chunk = []

    for word in words:
        chunk.append(word)
        if len(' '.join(chunk)) > max_tokens:
            chunks.append(' '.join(chunk))
            chunk = []

    if chunk:
        chunks.append(' '.join(chunk))

    return chunks

# 4. 텍스트 파일을 순회하면서 지식 그래프 생성 및 JSON 저장
max_chunk_size = 14000  # 적절한 청크 크기를 설정하세요.

file_list = os.listdir(input_dir)
file_list = [f for f in file_list if f.endswith('.txt')]  # .txt 파일만 필터링

for filename in tqdm(file_list, desc="Processing files"):
    json_file_path = os.path.join(json_output_dir, f"{os.path.splitext(filename)[0]}.json")

    # JSON 파일이 이미 존재하면 건너뜀
    if os.path.exists(json_file_path):
        print(f"JSON 파일이 이미 존재합니다: {json_file_path}, 건너뜁니다.", flush=True)
        continue

    file_path = os.path.join(input_dir, filename)

    # 텍스트 파일 읽기
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 텍스트를 청크로 나눔
    if len(text) <= max_chunk_size:
        chunks = [text]
    else:
        chunks = chunk_text(text, max_chunk_size)

    all_nodes = []
    all_relationships = []

    for chunk in chunks:
        document = Document(page_content=chunk)
        graph_documents = llm_transformer.convert_to_graph_documents([document])

        all_nodes.extend(graph_documents[0].nodes)
        all_relationships.extend(graph_documents[0].relationships)

    # 모든 노드와 관계를 JSON으로 변환
    graph_json = {
        "nodes": [serialize_node(node) for node in all_nodes],
        "relationships": [serialize_relationship(rel) for rel in all_relationships]
    }

    # JSON 파일로 저장
    with open(json_file_path, 'w', encoding='utf-8') as json_f:
        json.dump(graph_json, json_f, indent=2)
    print(f"Saved JSON: {json_file_path}", flush=True)

print("작업 완료!")
