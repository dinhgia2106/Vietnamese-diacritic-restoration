import torch
from transformers import RobertaForTokenClassification, RobertaConfig, RobertaTokenizerFast
from tqdm import tqdm
import re
import os

# --- Helper Functions ---
def remove_vietnamese_accents(text):
    s = re.sub(r'[àáạảãâầấậẩẫăằắặẳẵ]', 'a', text); s = re.sub(r'[ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴ]', 'A', s)
    s = re.sub(r'[èéẹẻẽêềếệểễ]', 'e', s); s = re.sub(r'[ÈÉẸẺẼÊỀẾỆỂỄ]', 'E', s)
    s = re.sub(r'[òóọỏõôồốộổỗơờớợởỡ]', 'o', s); s = re.sub(r'[ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ]', 'O', s)
    s = re.sub(r'[ìíịỉĩ]', 'i', s); s = re.sub(r'[ÌÍỊỈĨ]', 'I', s)
    s = re.sub(r'[ùúụủũưừứựửữ]', 'u', s); s = re.sub(r'[ÙÚỤỦŨƯỪỨỰỬỮ]', 'U', s)
    s = re.sub(r'[ỳýỵỷỹ]', 'y', s); s = re.sub(r'[ỲÝỴỶỸ]', 'Y', s)
    s = re.sub(r'đ', 'd', s); s = re.sub(r'Đ', 'D', s)
    return s

def split_to_syllables(text):
    return text.strip().split()

# --- Vocabulary Builder ---
class SyllableVocabulary:
    def __init__(self, special_tokens=None):
        if special_tokens is None: special_tokens = {"[PAD]": 0, "[UNK]": 1}
        self.accented_syllable_to_id = special_tokens
        self.id_to_accented_syllable = {v: k for k, v in special_tokens.items()}

    def build_from_data(self, data_corpus_path):
        print(f"Building syllable vocabulary from full data at {data_corpus_path}...")
        all_accented_syllables = set()
        try:
            with open(data_corpus_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="Scanning full corpus"):
                    syllables = split_to_syllables(line)
                    all_accented_syllables.update(syllables)
        except FileNotFoundError:
            print(f"ERROR: Data file not found at {data_corpus_path}")
            return

        for syllable in sorted(list(all_accented_syllables)):
            if syllable not in self.accented_syllable_to_id:
                new_id = len(self.accented_syllable_to_id)
                self.accented_syllable_to_id[syllable] = new_id
                self.id_to_accented_syllable[new_id] = syllable
        print(f"Vocabulary built. Found {self.get_vocab_size()} unique accented syllables.")

    def get_vocab_size(self): return len(self.accented_syllable_to_id)
    
    # <<-- THÊM KIỂM TRA TÍNH HỢP LỆ -->>
    def get_id(self, syllable):
        # Trả về ID của UNK nếu không tìm thấy, đảm bảo an toàn
        return self.accented_syllable_to_id.get(syllable, self.accented_syllable_to_id["[UNK]"])
        
    def get_syllable(self, id): return self.id_to_accented_syllable.get(id, "[UNK]")

# --- Model Wrapper ---
class PhobertAccentRestorer:
    def __init__(self, model_path, syllable_vocab, device=None):
        if device is None: self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else: self.device = device
        print(f"Using device: {self.device}")
        
        self.syllable_vocab = syllable_vocab
        self.num_labels = self.syllable_vocab.get_vocab_size()
        
        print("Loading FAST tokenizer (RobertaTokenizerFast)...")
        tokenizer_path = model_path if os.path.isdir(model_path) else "vinai/phobert-base"
        self.tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path)

        print(f"Loading PhoBERT model for {self.num_labels} labels from path: {model_path}")
        config_path = model_path if os.path.isdir(model_path) else "vinai/phobert-base"
        config = RobertaConfig.from_pretrained(config_path, num_labels=self.num_labels)
        
        self.model = RobertaForTokenClassification.from_pretrained(model_path, config=config, ignore_mismatched_sizes=True)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, unaccented_text):
        input_syllables = split_to_syllables(unaccented_text)
        inputs = self.tokenizer(unaccented_text, return_tensors="pt").to(self.device)
        with torch.no_grad(): logits = self.model(**inputs).logits
        predictions = torch.argmax(logits, dim=2)
        word_ids = inputs.word_ids(batch_index=0)
        restored_syllables, previous_word_idx = [], None
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx == previous_word_idx: continue
            if word_idx < len(input_syllables):
                predicted_id = predictions[0, token_idx].item()
                predicted_syllable = self.syllable_vocab.get_syllable(predicted_id)
                if remove_vietnamese_accents(predicted_syllable) != input_syllables[word_idx]:
                    restored_syllables.append(input_syllables[word_idx])
                else: restored_syllables.append(predicted_syllable)
            previous_word_idx = word_idx
        return " ".join(restored_syllables)

    def predict_beam_search(self, unaccented_text, beam_width=5, num_suggestions=3):
        input_syllables = split_to_syllables(unaccented_text)
        if not input_syllables: return []
        inputs = self.tokenizer(unaccented_text, return_tensors="pt").to(self.device)
        with torch.no_grad(): logits = self.model(**inputs).logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)
        word_ids = inputs.word_ids(batch_index=0)
        beams = [([], 0.0)]
        last_word_idx = -1
        first_tokens_indices = []
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx != last_word_idx:
                first_tokens_indices.append((token_idx, word_idx))
                last_word_idx = word_idx
        for token_idx, word_idx in first_tokens_indices:
            token_log_probs = log_probs[0, token_idx, :]
            topk_log_probs, topk_ids = torch.topk(token_log_probs, beam_width)
            new_beams = []
            for seq, score in beams:
                for i in range(beam_width):
                    syllable_id = topk_ids[i].item()
                    syllable_log_prob = topk_log_probs[i].item()
                    predicted_syllable = self.syllable_vocab.get_syllable(syllable_id)
                    if word_idx < len(input_syllables) and remove_vietnamese_accents(predicted_syllable) == input_syllables[word_idx]:
                        new_seq = seq + [predicted_syllable]
                        new_score = score + syllable_log_prob
                        new_beams.append((new_seq, new_score))
            if not new_beams:
                new_beams = [(seq + [input_syllables[word_idx]], score) for seq, score in beams]
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_width]
        final_suggestions = [" ".join(seq) for seq, score in beams]
        return final_suggestions[:num_suggestions]