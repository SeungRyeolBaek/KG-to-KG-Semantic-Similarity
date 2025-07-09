import time
import random
import nltk
from nltk.corpus import wordnet
from transformers import T5Tokenizer, T5ForConditionalGeneration, BertTokenizer, BertForMaskedLM
import torch
from nltk.tokenize import sent_tokenize, word_tokenize
import os
import warnings
import logging
from tqdm import tqdm
import gc

# 경고 메시지 억제 설정 추가
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)

# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('punkt_tab')

def replace_synonyms(text, target_prob=0.2):
    words = text.split()
    num_words = len(words)
    real_replace = 0

    replaceable_indices = []

    # First pass: Identify replaceable words
    for i, word in enumerate(words):
        synonyms = wordnet.synsets(word)
        synonyms = [syn for syn in synonyms if len(syn.lemmas()) > 1]
        if synonyms:
            replaceable_indices.append(i)

    # Calculate the number of words to replace
    num_to_replace = int(min(target_prob, len(replaceable_indices) / num_words) * num_words)

    # Randomly select words to replace
    indices_to_replace = random.sample(replaceable_indices, num_to_replace)

    # Perform replacement
    for i in indices_to_replace:
        synonyms = wordnet.synsets(words[i])
        synonyms = [syn for syn in synonyms if len(syn.lemmas()) > 1]
        if synonyms:
            chosen_syn = random.choice(synonyms)
            words[i] = random.choice(chosen_syn.lemmas()[1:]).name().replace('_', ' ')
            real_replace += 1

    return ' '.join(words), real_replace / num_words

def get_synonyms_from_wordnet(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

def replace_with_context(text, target_prob=0.9):
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    model = BertForMaskedLM.from_pretrained('bert-large-uncased').to('cuda')  # 모델을 GPU로 이동

    words = text.split()
    num_words = len(words)
    replaceable_indices = []

    for i, word in enumerate(words):
        if get_synonyms_from_wordnet(word):
            replaceable_indices.append(i)

    num_to_replace = int(min(target_prob, len(replaceable_indices) / num_words) * num_words)
    indices_to_replace = random.sample(replaceable_indices, num_to_replace)

    real_replace = 0
    for i in tqdm(indices_to_replace, desc="Replacing words", leave=False, position=1):  # tqdm 사용
        original_word = words[i]

        # Create a sentence with a [MASK] token
        masked_sentence = words[:i] + ['[MASK]'] + words[i+1:]
        masked_text = " ".join(masked_sentence)

        # Use BERT to predict the token for [MASK]
        inputs = tokenizer(masked_text, return_tensors='pt', padding=True, truncation=True).to('cuda')  # 입력을 GPU로 이동

        # Check if [MASK] token is found in input_ids
        mask_position = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0]
        if mask_position.numel() == 0:
            continue

        mask_position = mask_position.item()

        with torch.no_grad():
            outputs = model(**inputs)
        predictions = outputs.logits[0, mask_position]
        predicted_indices = torch.argsort(predictions, descending=True)
        predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_indices[:1])
        words[i] = predicted_tokens[0]
        real_replace += 1

    return ' '.join(words), real_replace / num_words


class DipperParaphraser(object):
    def __init__(self, model="SamSJackson/paraphrase-dipper-no-ctx", verbose=True):
        time1 = time.time()
        self.tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-large')
        # self.model = T5ForConditionalGeneration.from_pretrained(model, device_map='auto')
        self.model = T5ForConditionalGeneration.from_pretrained(model).to('cuda')
        if verbose:
            print(f"{model} model loaded in {time.time() - time1:.2f} seconds")
        self.model.eval()

    def paraphrase_batch(self, chunks, lex_diversity, order_diversity, **kwargs):
        """Paraphrase a list of text chunks using a single batch call."""
        assert lex_diversity in [0, 20, 40, 60, 80, 100], "Lexical diversity must be 0–100 in steps of 20"
        assert order_diversity in [0, 20, 40, 60, 80, 100], "Order diversity must be 0–100 in steps of 20"

        lex_code = int(100 - lex_diversity)
        order_code = int(100 - order_diversity)

        # Prepare batch inputs
        formatted_inputs = []
        for chunk in chunks:
            chunk = " ".join(chunk.split())
            formatted_input = f"lexical = {lex_code}, order = {order_code} <sent> {chunk} </sent>"
            formatted_inputs.append(formatted_input)

        # Tokenize and generate in batch
        tokenized = self.tokenizer(formatted_inputs, return_tensors="pt", padding=True, truncation=True)
        tokenized = {k: v.cuda() for k, v in tokenized.items()}

        with torch.inference_mode():
            outputs = self.model.generate(**tokenized, **kwargs)

        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return " ".join(decoded_outputs)

    def paraphrase_text(self, text, lex_diversity, order_diversity):
        print(f"Paraphrasing text with lex_diversity={lex_diversity}, order_diversity={order_diversity}")
        chunks = chunk_text(text, chunk_size=512)  # shorter chunk size for speed
        return self.paraphrase_batch(
            chunks,
            lex_diversity=lex_diversity,
            order_diversity=order_diversity,
            do_sample=True,
            top_p=0.75,
            top_k=None,
            max_length=384  # limit output length for faster generation
        )


def chunk_text(text, chunk_size=512):
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    tokens = tokenizer.tokenize(text)

    chunks = [' '.join(tokens[i:i + chunk_size]) for i in range(0, len(tokens), chunk_size)]

    # 토큰으로 자른 후 다시 텍스트로 변환
    chunks = [tokenizer.convert_tokens_to_string(chunk.split()) for chunk in chunks]
    return chunks

dp = DipperParaphraser(model="SamSJackson/paraphrase-dipper-no-ctx")
# 폴더 경로 설정
input_folder = 'bookcorpus_text/chunk1'
output_folders = ['bookcorpus_text/synonym_replacement/0.3', 'bookcorpus_text/synonym_replacement/0.6','bookcorpus_text/context_replacement/0.3', 'bookcorpus_text/context_replacement/0.6', 'bookcorpus_text/dipper_paraphraser/60_20', 'bookcorpus_text/dipper_paraphraser/60_0']

# 출력 폴더 생성
for folder in output_folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

# dp = DipperParaphraser(model="SamSJackson/paraphrase-dipper-no-ctx")
# 파일 읽기 및 변형 적용 후 저장
for filename in os.listdir(input_folder):
# for filename in tqdm(os.listdir(input_folder), desc="Files", leave=False, position=1):  # tqdm 사용
    if filename.endswith('.txt'):

        prompt = f"This is a Wikipedia article about {filename}, please paraphrase the text while keeping the subject matter intact."
        with open(os.path.join(input_folder, filename), 'r', encoding='utf-8') as file:
            text = file.read()

        chunks = chunk_text(text, chunk_size=256)

		# 첫 번째 chunk에서 첫 200 토큰만 사용
        first_200_tokens = ' '.join(chunks[0].split()[:200])

        # 변형 1: Synonym Replacement
        if os.path.exists(os.path.join(output_folders[0], filename)):
            print(f"파일이 이미 존재합니다: {os.path.join(output_folders[0], filename)}, 건너뜁니다.")
        else:
            transformed_text, replace_rate = replace_synonyms(text=text,target_prob=0.3)
            print(f"{filename}: Synonym Replacement - Target Rate: 0.3, Real Rate: {replace_rate}")
            with open(os.path.join(output_folders[0], filename), 'w', encoding='utf-8') as out_file:
                out_file.write(transformed_text)

        # 변형 1: Synonym Replacement
        if os.path.exists(os.path.join(output_folders[1], filename)):
            print(f"파일이 이미 존재합니다: {os.path.join(output_folders[1], filename)}, 건너뜁니다.")
        else:
            transformed_text, replace_rate = replace_synonyms(text=text,target_prob=0.6)
            print(f"{filename}: Synonym Replacement - Target Rate: 0.6, Real Rate: {replace_rate}")
            with open(os.path.join(output_folders[1], filename), 'w', encoding='utf-8') as out_file:
                out_file.write(transformed_text)

        # 변형 2: Context Replacement (Chunk 처리)
        if os.path.exists(os.path.join(output_folders[2], filename)):
            print(f"파일이 이미 존재합니다: {os.path.join(output_folders[2], filename)}, 건너뜁니다.")
        else:
            transformed_text = ""
            total_replace_rate = 0
            for i, chunk in enumerate(chunks, 1):
                print(f"Processing chunk {i}/{len(chunks)}")
                chunk_transformed, chunk_replace_rate = replace_with_context(text=chunk, target_prob=0.3)
                transformed_text += chunk_transformed
                total_replace_rate += chunk_replace_rate
            print(f"{filename}: Context Replacement - Target Rate: 0.3 Real Rate: {total_replace_rate / len(chunks)}")
            with open(os.path.join(output_folders[2], filename), 'w', encoding='utf-8') as out_file:
                out_file.write(transformed_text)

        # 변형 2: Context Replacement (Chunk 처리)
        if os.path.exists(os.path.join(output_folders[3], filename)):
            print(f"파일이 이미 존재합니다: {os.path.join(output_folders[3], filename)}, 건너뜁니다.")
        else:
            transformed_text = ""
            total_replace_rate = 0
            for i, chunk in enumerate(chunks, 1):
                print(f"Processing chunk {i}/{len(chunks)}")
                chunk_transformed, chunk_replace_rate = replace_with_context(text=chunk, target_prob=0.6)
                transformed_text += chunk_transformed
                total_replace_rate += chunk_replace_rate
            print(f"{filename}: Context Replacement - Target Rate: 0.6 Real Rate: {total_replace_rate / len(chunks)}")
            with open(os.path.join(output_folders[3], filename), 'w', encoding='utf-8') as out_file:
                out_file.write(transformed_text)

        # 변형 3: Dipper Paraphraser (batch 처리)
        if os.path.exists(os.path.join(output_folders[4], filename)):
            a = 1
            # print(f"파일이 이미 존재합니다: {os.path.join(output_folders[4], filename)}, 건너뜁니다.")
        else:
            transformed_text = dp.paraphrase_text(text, lex_diversity=60, order_diversity=20)
            print(f"{filename}: Dipper Paraphraser completed.")
            with open(os.path.join(output_folders[4], filename), 'w', encoding='utf-8') as out_file:
                out_file.write(transformed_text)
            del transformed_text


        # 변형 4: Dipper Paraphraser (batch 처리, 다른 옵션)
        if os.path.exists(os.path.join(output_folders[5], filename)):
            print(f"파일이 이미 존재합니다: {os.path.join(output_folders[5], filename)}, 건너뜁니다.")
        else:
            transformed_text = dp.paraphrase_text(text, lex_diversity=60, order_diversity=0)
            print(f"{filename}: Dipper Paraphraser completed.")
            with open(os.path.join(output_folders[5], filename), 'w', encoding='utf-8') as out_file:
                out_file.write(transformed_text)
            del transformed_text
		
		# 메모리 해제
        del text, chunks  # 파이썬 객체 삭제
        torch.cuda.empty_cache()  # GPU 메모리 해제
        gc.collect()  # 파이썬 가비지 컬렉션 실행

print("All files have been processed and saved.")
