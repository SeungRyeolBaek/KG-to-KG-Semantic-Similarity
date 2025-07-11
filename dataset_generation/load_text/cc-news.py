from datasets import load_dataset
from itertools import islice
import os

# 설정 -----------------------------------------------------------------
SAVE_ROOT   = "cc_news_text/original_text"   # 저장 폴더
TARGET_SIZE = 550                            # 파일 개수
MIN_TOK     = 1000                            # 길이 하한
MAX_TOK     = 1200                            # 길이 상한

# 1. CC-News 스트리밍 로드 ---------------------------------------------
ds = load_dataset("cc_news", split="train", streaming=True)

# 2. 폴더 생성 ----------------------------------------------------------
os.makedirs(SAVE_ROOT, exist_ok=True)

# 3. 길이 필터링하며 600편 저장 ----------------------------------------
saved = 0
for art in ds:
    text = art["text"].strip()
    ntok = len(text.split())
    if MIN_TOK <= ntok <= MAX_TOK:
        idx = saved + 1
        with open(f"{SAVE_ROOT}/{idx}.txt", "w", encoding="utf-8") as f:
            f.write(text)
        saved += 1
        if saved >= TARGET_SIZE:
            break

print(f"Done. Saved {saved} articles to {SAVE_ROOT}")
