# llm_only_text_scoring.py
# ──────────────────────────────────────────────────────────────
# SBERT 제외 - BGE(HF) & OpenAI Emb-3 전용 점수 계산 스크립트
# 파일 구조는 이전 코드와 100 % 동일

import os, json, time, numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
load_dotenv()                         # .env 에서 OPENAI_API_KEY 로드

# ──────────────────────────────────────────────────────────────
# 1) 임베딩 모델 로더
# ──────────────────────────────────────────────────────────────
def load_embedder(model_tag: str):
    """
    model_tag 예시
      • 'hf:BAAI/bge-large-en-v1.5'
      • 'openai:text-embedding-3-large'
    반환: encode(List[str]) -> np.ndarray (N, dim)  (L2-정규화 포함)
    """
    src, name = model_tag.split(":", 1)

    # ------------------ HuggingFace ------------------
    if src == "hf":
        from sentence_transformers import SentenceTransformer
        st_model = SentenceTransformer(name)

        def encode(texts, bs: int = 64):
            return st_model.encode(
                texts, batch_size=bs, show_progress_bar=False,
                normalize_embeddings=True                      # 이미 L2 정규화
            )
        return encode

    # ------------------ OpenAI (>=1.0) ------------------
    if src == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        assert client.api_key, "❌  OPENAI_API_KEY 환경변수가 비어 있습니다."

        def encode(texts, bs: int = 96):
            vecs_all = []
            for i in range(0, len(texts), bs):
                chunk = texts[i:i + bs]
                resp = client.embeddings.create(model=name, input=chunk)
                # 순서가 그대로 보장됨
                vecs = np.asarray([d.embedding for d in resp.data],
                                  dtype=np.float32)
                # L2-정규화 (OpenAI 임베딩은 정규화 X)
                vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
                vecs_all.append(vecs)
            return np.vstack(vecs_all)

        return encode

    raise ValueError(f"unknown model_tag: {model_tag}")

# ──────────────────────────────────────────────────────────────
# 2) 코어: 임베딩 → cosine similarity → JSON
# ──────────────────────────────────────────────────────────────
def compute_similarities(orig_dir, mod_dir, out_dir, model_tag):
    encode = load_embedder(model_tag)

    orig_files = [os.path.join(r, f)
                  for r, _, fs in os.walk(orig_dir)
                  for f in fs if f.endswith(".txt")]
    mod_files  = [os.path.join(r, f)
                  for r, _, fs in os.walk(mod_dir)
                  for f in fs if f.endswith(".txt")]

    # 텍스트 로드
    texts = [open(p, encoding="utf-8").read().strip()
             for p in orig_files + mod_files]

    t0 = time.perf_counter()
    emb = encode(texts)                    # (N, dim)  L2-norm = 1
    sims = cosine_similarity(emb)          # (N, N)
    print(f"[{model_tag}] embed+sim ⏱ {time.perf_counter() - t0:,.1f}s")

    n_orig = len(orig_files)
    os.makedirs(out_dir, exist_ok=True)

    # 결과 파일 저장
    for i_mod, m_path in enumerate(tqdm(mod_files, desc="Comparing")):
        m_name = os.path.basename(m_path).replace(".txt", "")
        result = []
        for j_orig, o_path in enumerate(orig_files):
            o_name = os.path.basename(o_path).replace(".txt", ".json")
            score  = round(float(sims[n_orig + i_mod, j_orig]), 2)
            result.append({"original": o_name,
                           "similarity_score": score})

        with open(os.path.join(out_dir, f"{m_name}.json"),
                  "w", encoding="utf-8") as fp:
            json.dump(result, fp, ensure_ascii=False, indent=4)

# ──────────────────────────────────────────────────────────────
# 3) 데이터셋 래퍼
# ──────────────────────────────────────────────────────────────
def llm_scoring(dataset, modification, option, model_tag):
    out_map = {
        'c4_subset_graph_verbalized'    : 'c4_result_text',
        'wikitext_graph_data_verbalized': 'wikitext_result_text',
        'cc_news_graph_verbalized'      : 'CC_news_result_text'
    }
    out_root = out_map[dataset]

    orig_dir = f"{dataset}/original_graph"
    mod_dir  = f"{dataset}/{modification}/{option}"
    out_dir  = f"Result/{out_root}/{modification}/{option}/{model_tag.replace(':', '_')}"

    compute_similarities(orig_dir, mod_dir, out_dir, model_tag)

# ──────────────────────────────────────────────────────────────
# 4) 실행 예시
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":

    datasets = [
        'cc_news_graph_verbalized',
        'wikitext_graph_data_verbalized'
    ]

    modifications = ['synonym_replacement',
                     'context_replacement',
                     'dipper_paraphraser']

    options = {
        'synonym_replacement': ['0.3', '0.6'],
        'context_replacement': ['0.3', '0.6'],
        'dipper_paraphraser' : ['60_0', '60_20']
    }

    model_tags = [
        "hf:BAAI/bge-large-en-v1.5",      # BGE-Large (HF)
        "openai:text-embedding-3-large"   # OpenAI Emb-3
    ]

    for ds in datasets:
        for mod in modifications:
            for opt in options[mod]:
                for tag in model_tags:
                    print(f"\n=== {ds} | {mod}/{opt} | {tag} ===")
                    llm_scoring(ds, mod, opt, tag)
