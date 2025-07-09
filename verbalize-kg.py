import os
import json
import nltk
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer

# 필수 NLTK 리소스 다운로드
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')  # Python 3.10 이상 대응

lemmatizer = WordNetLemmatizer()

def verbalize_openie_like(subject, relation, obj):
    """
    관계에 상관없이 항상 완전한 문장으로 verbalize
    """
    # 관계를 단어로 분해
    rel_tokens = word_tokenize(relation.replace("_", " ").lower())

    # 관계가 명사 같으면: "is the REL of"
    # 관계가 동사 같으면: "RELs" (동사 활용)
    rel_pos = pos_tag(rel_tokens, lang='eng')
    main_word, main_pos = rel_pos[0] if rel_pos else ('related_to', 'NN')

    if main_pos.startswith("VB"):  # 동사형
        rel_verb = lemmatizer.lemmatize(main_word, 'v')
        return f"{subject} {rel_verb}s {obj}."
    else:  # 명사형 또는 애매한 경우
        rel_phrase = ' '.join(rel_tokens)
        return f"{subject} is the {rel_phrase} of {obj}."

def textualize_graph(data):
    """
    JSON KG 데이터 → 자연어 문장들로 변환
    """
    lines = []
    for rel in data.get("relationships", []):
        head = rel["source"]["id"]
        relation = rel["type"]
        tail = rel["target"]["id"]
        sentence = verbalize_openie_like(head, relation, tail)
        lines.append(sentence)
    return "\n".join(lines)

def convert_folder_json_to_text(base_input_dir, base_output_dir):
    """
    base_input_dir: json이 있는 상위 폴더
    base_output_dir: 결과 텍스트를 저장할 상위 폴더
    """
    for root, _, files in os.walk(base_input_dir):
        for fname in files:
            if not fname.endswith(".json"):
                continue
            json_path = os.path.join(root, fname)
            rel_path = os.path.relpath(json_path, base_input_dir)
            txt_path = os.path.join(base_output_dir, rel_path)
            txt_path = txt_path.replace(".json", ".txt")

            os.makedirs(os.path.dirname(txt_path), exist_ok=True)

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            text = textualize_graph(data)

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)

    print(f"[INFO] Converted all JSON KGs in '{base_input_dir}' to OpenIE-style text in '{base_output_dir}'.")

# 실행
if __name__ == "__main__":
    input_dir = "bookcorpus_graph"
    output_dir = "bookcorpus_graph_verbalized"
    convert_folder_json_to_text(input_dir, output_dir)
