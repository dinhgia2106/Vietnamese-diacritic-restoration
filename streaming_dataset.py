import torch
from torch.utils.data import IterableDataset
from tqdm import tqdm

# Import các hàm cần thiết từ phoBERT.py
from phoBERT import split_to_syllables, remove_vietnamese_accents

class StreamingAccentSyllableDataset(IterableDataset):
    def __init__(self, file_path, tokenizer, syllable_vocab, chunk_size=10000, max_length=128):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.syllable_vocab = syllable_vocab
        self.chunk_size = chunk_size
        self.max_length = max_length

    def _process_line(self, line):
        """Hàm xử lý một dòng dữ liệu để tạo ra một sample."""
        target_syllables = split_to_syllables(line)
        if not target_syllables:
            return None
        
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
            if word_idx is None: continue
            if word_idx != previous_word_idx:
                if word_idx < len(target_syllables):
                    target_syllable = target_syllables[word_idx]
                    label_id = self.syllable_vocab.get_id(target_syllable)
                    labels[0, token_idx] = label_id
            previous_word_idx = word_idx
        
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs['labels'] = labels.squeeze(0)
        return inputs

    def __iter__(self):
        """Hàm iterator sẽ đọc file theo từng chunk."""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                while True:
                    chunk = [line.strip() for _, line in zip(range(self.chunk_size), f) if line.strip()]
                    if not chunk:
                        break # Kết thúc file
                    
                    # Xử lý và yield từng sample trong chunk
                    for line in chunk:
                        processed = self._process_line(line)
                        if processed:
                            yield processed
        except FileNotFoundError:
            print(f"Error: Data file not found at {self.file_path}")
            return