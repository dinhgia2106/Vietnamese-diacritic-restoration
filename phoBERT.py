import torch
from torch.utils.data import Dataset, DataLoader
# THAY ĐỔI: Gọi thẳng RobertaTokenizerFast và loại bỏ AutoTokenizer
from transformers import RobertaForTokenClassification, RobertaConfig, RobertaTokenizerFast
from tqdm import tqdm
import re
import os

# --- 1. Helper Functions ---
# (Không thay đổi)
def remove_vietnamese_accents(text):
    s = re.sub(r'[àáạảãâầấậẩẫăằắặẳẵ]', 'a', text)
    s = re.sub(r'[ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴ]', 'A', s)
    s = re.sub(r'[èéẹẻẽêềếệểễ]', 'e', s)
    s = re.sub(r'[ÈÉẸẺẼÊỀẾỆỂỄ]', 'E', s)
    s = re.sub(r'[òóọỏõôồốộổỗơờớợởỡ]', 'o', s)
    s = re.sub(r'[ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ]', 'O', s)
    s = re.sub(r'[ìíịỉĩ]', 'i', s)
    s = re.sub(r'[ÌÍỊỈĨ]', 'I', s)
    s = re.sub(r'[ùúụủũưừứựửữ]', 'u', s)
    s = re.sub(r'[ÙÚỤỦŨƯỪỨỰỬỮ]', 'U', s)
    s = re.sub(r'[ỳýỵỷỹ]', 'y', s)
    s = re.sub(r'[ỲÝỴỶỸ]', 'Y', s)
    s = re.sub(r'đ', 'd', s)
    s = re.sub(r'Đ', 'D', s)
    return s

def split_to_syllables(text):
    return text.strip().split()

# --- 2. Vocabulary Builder ---
# (Không thay đổi)
class SyllableVocabulary:
    def __init__(self, special_tokens=None):
        if special_tokens is None:
            special_tokens = {"[PAD]": 0, "[UNK]": 1}
        self.accented_syllable_to_id = special_tokens
        self.id_to_accented_syllable = {v: k for k, v in special_tokens.items()}
        self.unaccented_to_accented_map = {}
    def build_from_data(self, data_corpus):
        print("Building syllable vocabulary from data...")
        all_accented_syllables = set()
        for sentence in tqdm(data_corpus, desc="Scanning corpus"):
            syllables = split_to_syllables(sentence)
            for s in syllables:
                all_accented_syllables.add(s)
        for syllable in sorted(list(all_accented_syllables)):
            if syllable not in self.accented_syllable_to_id:
                new_id = len(self.accented_syllable_to_id)
                self.accented_syllable_to_id[syllable] = new_id
                self.id_to_accented_syllable[new_id] = syllable
        for accented, _ in self.accented_syllable_to_id.items():
            if accented in ["[PAD]", "[UNK]"]:
                continue
            unaccented = remove_vietnamese_accents(accented)
            if unaccented not in self.unaccented_to_accented_map:
                self.unaccented_to_accented_map[unaccented] = []
            self.unaccented_to_accented_map[unaccented].append(accented)
        print(f"Vocabulary built. Found {self.get_vocab_size()} unique accented syllables.")
    def get_vocab_size(self):
        return len(self.accented_syllable_to_id)
    def get_id(self, syllable):
        return self.accented_syllable_to_id.get(syllable, self.accented_syllable_to_id["[UNK]"])
    def get_syllable(self, id):
        return self.id_to_accented_syllable.get(id, "[UNK]")

# --- 3. Dataset Class ---
# (Không thay đổi)
class AccentSyllableDataset(Dataset):
    def __init__(self, data_corpus, tokenizer, syllable_vocab, max_length=128):
        self.tokenizer = tokenizer
        self.syllable_vocab = syllable_vocab
        self.max_length = max_length
        self.data = self._preprocess(data_corpus)
    def _preprocess(self, data_corpus):
        processed_data = []
        print("Preprocessing data for the dataset...")
        for sentence in tqdm(data_corpus, desc="Processing sentences"):
            target_syllables = split_to_syllables(sentence)
            input_syllables = [remove_vietnamese_accents(s) for s in target_syllables]
            input_text = " ".join(input_syllables)
            inputs = self.tokenizer(
                input_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            word_ids = inputs.word_ids(batch_index=0)
            labels = torch.full(inputs.input_ids.shape, -100, dtype=torch.long)
            previous_word_idx = None
            for token_idx, word_idx in enumerate(word_ids):
                if word_idx is None:
                    continue
                if word_idx != previous_word_idx:
                    if word_idx < len(target_syllables):
                        target_syllable = target_syllables[word_idx]
                        label_id = self.syllable_vocab.get_id(target_syllable)
                        labels[0, token_idx] = label_id
                previous_word_idx = word_idx
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
            inputs['labels'] = labels.squeeze(0)
            processed_data.append(inputs)
        return processed_data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

# --- 4. Model Wrapper ---
class PhobertAccentRestorer:
    def __init__(self, model_path, syllable_vocab, device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        print(f"Using device: {self.device}")
        self.syllable_vocab = syllable_vocab
        self.num_labels = self.syllable_vocab.get_vocab_size()
        
        # THAY ĐỔI: Sử dụng RobertaTokenizerFast thay vì AutoTokenizer
        print("Loading FAST tokenizer (RobertaTokenizerFast)...")
        self.tokenizer = RobertaTokenizerFast.from_pretrained("vinai/phobert-base")

        print(f"Loading PhoBERT model for {self.num_labels} labels from path: {model_path}")
        config = RobertaConfig.from_pretrained("vinai/phobert-base", num_labels=self.num_labels)
        self.model = RobertaForTokenClassification.from_pretrained(
            model_path,
            config=config,
            ignore_mismatched_sizes=True,
        )
        self.model.to(self.device)
        self.model.eval()
    def predict(self, unaccented_text):
        input_syllables = split_to_syllables(unaccented_text)
        inputs = self.tokenizer(unaccented_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        predictions = torch.argmax(logits, dim=2)
        word_ids = inputs.word_ids(batch_index=0)
        restored_syllables = []
        previous_word_idx = None
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None: continue
            if word_idx != previous_word_idx:
                if word_idx < len(input_syllables):
                    predicted_id = predictions[0, token_idx].item()
                    predicted_syllable = self.syllable_vocab.get_syllable(predicted_id)
                    if remove_vietnamese_accents(predicted_syllable) != input_syllables[word_idx]:
                        restored_syllables.append(input_syllables[word_idx])
                    else:
                        restored_syllables.append(predicted_syllable)
            previous_word_idx = word_idx
        return " ".join(restored_syllables)
    
    def predict_beam_search(self, unaccented_text, beam_width=3, num_suggestions=3):
        """
        Restores accents using beam search to provide multiple suggestions.
        """
        input_syllables = split_to_syllables(unaccented_text)
        
        inputs = self.tokenizer(unaccented_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        # Chuyển logits sang log probabilities để tính toán (tránh underflow và dễ cộng)
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)
        
        word_ids = inputs.word_ids(batch_index=0)
        
        # Khởi tạo beam: mỗi phần tử là (chuỗi_âm_tiết, tổng_log_prob)
        beams = [([], 0.0)]
        
        last_word_idx = -1
        
        # Lặp qua từng token để xây dựng các câu giả thuyết
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx == last_word_idx:
                continue

            last_word_idx = word_idx
            
            # Lấy log_probs cho token đầu tiên của âm tiết hiện tại
            token_log_probs = log_probs[0, token_idx, :]
            
            # Lấy top k âm tiết có khả năng cao nhất cho vị trí này
            # k ở đây là beam_width để đảm bảo có đủ lựa chọn
            topk_log_probs, topk_ids = torch.topk(token_log_probs, beam_width)
            
            new_beams = []
            
            # Mở rộng các beam hiện tại
            for seq, score in beams:
                for i in range(beam_width):
                    syllable_id = topk_ids[i].item()
                    syllable_log_prob = topk_log_probs[i].item()
                    
                    predicted_syllable = self.syllable_vocab.get_syllable(syllable_id)
                    
                    # Kiểm tra tính hợp lệ: âm tiết dự đoán phải có gốc không dấu khớp
                    if remove_vietnamese_accents(predicted_syllable) == input_syllables[word_idx]:
                        new_seq = seq + [predicted_syllable]
                        new_score = score + syllable_log_prob
                        new_beams.append((new_seq, new_score))
            
            # Sắp xếp và giữ lại các beam tốt nhất
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_width]

        # Chuyển đổi kết quả cuối cùng thành chuỗi
        final_suggestions = [" ".join(seq) for seq, score in beams]
        
        return final_suggestions[:num_suggestions]

# --- 5. Main Execution Block: Example Usage ---
if __name__ == "__main__":
    data_corpus = [
        "hôm nay trời đẹp",
        "tôi đi học",
        "việt nam quê hương tôi",
        "phở là một món ăn ngon của hà nội",
        "chúng ta cùng nhau học máy",
    ]
    syllable_vocab = SyllableVocabulary()
    syllable_vocab.build_from_data(data_corpus)
    
    MODEL_NAME = "vinai/phobert-base"
    # THAY ĐỔI: Sử dụng RobertaTokenizerFast thay vì AutoTokenizer
    print("\nLoading FAST tokenizer (RobertaTokenizerFast) for dataset creation...")
    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)
    
    print("\n--- Initializing a new model (for demonstration) ---")
    num_labels = syllable_vocab.get_vocab_size()
    config = RobertaConfig.from_pretrained(MODEL_NAME, num_labels=num_labels)
    model = RobertaForTokenClassification.from_pretrained(
        MODEL_NAME,
        config=config,
        ignore_mismatched_sizes=True 
    )
    
    print("\n--- Preparing Dataset and DataLoader (for training) ---")
    dataset = AccentSyllableDataset(data_corpus, tokenizer, syllable_vocab)
    dataloader = DataLoader(dataset, batch_size=2)
    
    sample_batch = next(iter(dataloader))
    print("\nSample batch keys:", sample_batch.keys())
    print("Input IDs shape:", sample_batch['input_ids'].shape)
    print("Labels shape:", sample_batch['labels'].shape)
    
    outputs = model(
        input_ids=sample_batch['input_ids'], 
        attention_mask=sample_batch['attention_mask'],
        labels=sample_batch['labels']
    )
    print("Model output loss (random):", outputs.loss.item())
    
    TEMP_MODEL_PATH = "./temp_phobert_accent_model"
    if not os.path.exists(TEMP_MODEL_PATH):
        os.makedirs(TEMP_MODEL_PATH)
    
    print(f"\nSaving untrained model to '{TEMP_MODEL_PATH}' for demonstration...")
    model.save_pretrained(TEMP_MODEL_PATH)

    print("\n--- Demonstrating Inference ---")
    restorer = PhobertAccentRestorer(
        model_path=TEMP_MODEL_PATH,
        syllable_vocab=syllable_vocab
    )
    
    test_sentence_1 = "hom nay troi dep"
    restored_sentence_1 = restorer.predict(test_sentence_1)
    print(f"\nInput:    '{test_sentence_1}'")
    print(f"Restored: '{restored_sentence_1}' (Output is random without training)")

    test_sentence_2 = "chung ta cung nhau hoc may"
    restored_sentence_2 = restorer.predict(test_sentence_2)
    print(f"Input:    '{test_sentence_2}'")
    print(f"Restored: '{restored_sentence_2}' (Output is random without training)")