import os
import re
from datasets import load_dataset

# WikiText 데이터셋 로드
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

# 텍스트 파일을 저장할 디렉토리 생성
output_dir = 'wikitext_articles'
os.makedirs(output_dir, exist_ok=True)

# 문서 카운터 초기화
doc_count = 0
max_docs = 1000
current_content = []
current_title = None

def process_dataset(dataset_split):
    global doc_count, current_content, current_title
    for item in ds[dataset_split]:
        text = item['text']
        lines = text.split('\n')

        for line in lines:
            line = line.strip()

            # 큰 제목을 찾음 (= Title = 형식)
            if line.startswith('= ') and line.endswith(' =') and not (
                (line.startswith('= = ')) or (line.startswith('= = = ')) or
                re.match(r'^= \d', line) or
                re.match(r'^.*\d =$', line) or
                re.match(r'^= [a-z]', line) or
                ';' in line or
                re.match(r'^= \w+ , \w+ =$', line) or
                re.search(r', Pts', line) or
                re.search(r', Bwl', line)
            ):
                # 현재 제목이 있고, 내용이 수집되었다면 저장
                if current_title and current_content:
                    safe_title = current_title.strip('=').strip().replace("/", "_").replace("\\", "_")
                    file_path = os.path.join(output_dir, f"{doc_count + 1}_{safe_title}.txt")
                    with open(file_path, 'w', encoding='utf-8') as out_f:
                        out_f.write("\n".join(current_content))
                    print(f"Saved: {file_path}")
                    doc_count += 1

                    # 최대 문서 수에 도달하면 종료
                    if doc_count >= max_docs:
                        return

                # 새로운 문서 시작
                current_title = line
                current_content = [line]  # 제목도 내용에 포함
            else:
                # 문서 내용을 리스트에 추가
                current_content.append(line)
        
        # 최대 문서 수에 도달하면 종료
        if doc_count >= max_docs:
            return

# train, validation, test 데이터셋 순차적으로 처리
for split in ['train', 'validation', 'test']:
    process_dataset(split)
    if doc_count >= max_docs:
        break

# 마지막 문서 저장 (반복문 종료 후 마지막으로 남은 문서 저장)
if doc_count < max_docs and current_title and current_content:
    safe_title = current_title.strip('=').strip().replace("/", "_").replace("\\", "_")
    file_path = os.path.join(output_dir, f"{doc_count + 1}_{safe_title}.txt")
    with open(file_path, 'w', encoding='utf-8') as out_f:
        out_f.write("\n".join(current_content))
    print(f"Saved: {file_path}")

print("작업 완료!")
