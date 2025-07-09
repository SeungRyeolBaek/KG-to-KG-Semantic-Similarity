# bookcorpus_extract.py
from datasets import load_dataset
import os, textwrap

# ── 설정 ────────────────────────────────────────────────
SAVE_ROOT   = "bookcorpus_text/original_text_1"
TARGET_SIZE = 600
MIN_TOK, MAX_TOK = 1_000, 1_200
MAX_EXAMPLES_SCAN = 200_000        # 스캔 상한

# 1) BookCorpus 스트리밍 로드 ───────────────────────────
ds_iter = load_dataset(
    "bookcorpus",
    split="train",
    streaming=True,
    trust_remote_code=True      # (필수 옵션)
)

# 2) 저장 폴더 생성 ────────────────────────────────────
os.makedirs(SAVE_ROOT, exist_ok=True)

# 3) 단락을 이어 붙여 원하는 길이의 ‘챕터’ 만들기 ──────
saved      = 0                  # 저장된 파일 수
buffer     = []                 # 누적 단락
tok_count  = 0                  # 누적 토큰 수

for ex_idx, ex in enumerate(ds_iter):
    if ex_idx >= MAX_EXAMPLES_SCAN or saved >= TARGET_SIZE:
        break

    paragraph = textwrap.dedent(ex["text"]).strip()
    if not paragraph:
        continue

    p_tok = len(paragraph.split())

    # 너무 긴 단락은 건너뛴다
    if p_tok > MAX_TOK:
        continue

    # 단락 추가
    buffer.append(paragraph)
    tok_count += p_tok

    # 길이가 범위에 들어오면 저장
    if MIN_TOK <= tok_count <= MAX_TOK:
        saved += 1
        out_path = os.path.join(SAVE_ROOT, f"{saved}.txt")
        with open(out_path, "w", encoding="utf-8") as fp:
            fp.write("\n\n".join(buffer))   # 단락 사이 두 줄 공백

        # 버퍼 초기화
        buffer, tok_count = [], 0

# ───────────────────────────────────────────────────────
print(f"✔ Done. Saved {saved} chapters to '{SAVE_ROOT}'")