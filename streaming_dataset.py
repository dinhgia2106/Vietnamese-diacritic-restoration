import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
import glob
from typing import List, Iterator, Tuple
import pickle
from tqdm import tqdm

class StreamingDatasetManager:
    """
    Quản lý việc chia dataset lớn thành các chunks nhỏ để training.
    """
    def __init__(self, data_dir: str, chunk_size: int = 1000000):
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.chunk_files = []
        self.current_chunk = 0
        
    def split_large_dataset(self, input_file: str, output_dir: str = None):
        """
        Chia dataset lớn thành các chunks nhỏ.
        """
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(input_file), "chunks")
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Splitting {input_file} into chunks of {self.chunk_size} samples...")
        
        chunk_count = 0
        current_chunk_data = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm(f, desc="Processing lines")):
                line = line.strip()
                if line:
                    current_chunk_data.append(line)
                    
                    if len(current_chunk_data) >= self.chunk_size:
                        # Save current chunk
                        chunk_filename = os.path.join(output_dir, f"chunk_{chunk_count:03d}.json")
                        with open(chunk_filename, 'w', encoding='utf-8') as chunk_f:
                            json.dump(current_chunk_data, chunk_f, ensure_ascii=False, indent=2)
                        
                        print(f"Saved chunk {chunk_count} with {len(current_chunk_data)} samples")
                        chunk_count += 1
                        current_chunk_data = []
        
        # Save remaining data
        if current_chunk_data:
            chunk_filename = os.path.join(output_dir, f"chunk_{chunk_count:03d}.json")
            with open(chunk_filename, 'w', encoding='utf-8') as chunk_f:
                json.dump(current_chunk_data, chunk_f, ensure_ascii=False, indent=2)
            print(f"Saved final chunk {chunk_count} with {len(current_chunk_data)} samples")
        
        self.chunk_files = glob.glob(os.path.join(output_dir, "chunk_*.json"))
        self.chunk_files.sort()
        print(f"Total chunks created: {len(self.chunk_files)}")
        
    def get_chunk_files(self, chunk_dir: str = None):
        """
        Lấy danh sách các chunk files.
        """
        if chunk_dir is None:
            chunk_dir = self.data_dir
            
        self.chunk_files = glob.glob(os.path.join(chunk_dir, "chunk_*.json"))
        self.chunk_files.sort()
        return self.chunk_files
    
    def load_chunk(self, chunk_index: int):
        """
        Load một chunk cụ thể.
        """
        if chunk_index >= len(self.chunk_files):
            return None
            
        chunk_file = self.chunk_files[chunk_index]
        print(f"Loading chunk {chunk_index}: {chunk_file}")
        
        with open(chunk_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        return data
    
    def get_total_chunks(self):
        """
        Trả về tổng số chunks.
        """
        return len(self.chunk_files)

class ChunkDataset(Dataset):
    """
    Dataset class cho một chunk cụ thể.
    """
    def __init__(self, data: List[str], tokenizer, syllable_vocab, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.syllable_vocab = syllable_vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Import functions from phoBERT.py
        from phoBERT import remove_vietnamese_accents, split_to_syllables
        
        sentence = self.data[idx]
        accented_syllables = split_to_syllables(sentence)
        unaccented_text = remove_vietnamese_accents(sentence)
        
        # Tokenize
        encoding = self.tokenizer(
            unaccented_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Create labels
        word_ids = encoding.word_ids(batch_index=0)
        labels = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)  # Ignore special tokens
            elif word_idx != previous_word_idx:
                if word_idx < len(accented_syllables):
                    syllable_id = self.syllable_vocab.get_id(accented_syllables[word_idx])
                    labels.append(syllable_id)
                else:
                    labels.append(-100)
            else:
                labels.append(-100)  # Ignore sub-tokens
            previous_word_idx = word_idx
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

class StreamingTrainer:
    """
    Trainer class để train model với streaming data.
    """
    def __init__(self, model, tokenizer, syllable_vocab, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.syllable_vocab = syllable_vocab
        self.device = device
        self.model.to(device)
        
    def train_on_chunk(self, chunk_data: List[str], epochs_per_chunk: int = 10, 
                      batch_size: int = 32, learning_rate: float = 2e-5):
        """
        Train model trên một chunk cụ thể.
        """
        from transformers import AdamW, get_scheduler
        from torch.utils.data import DataLoader, random_split
        
        # Create dataset for this chunk
        dataset = ChunkDataset(chunk_data, self.tokenizer, self.syllable_vocab)
        
        # Split train/val
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        num_training_steps = epochs_per_chunk * len(train_dataloader)
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
        
        # Training loop
        self.model.train()
        for epoch in range(epochs_per_chunk):
            print(f"Epoch {epoch + 1}/{epochs_per_chunk}")
            
            total_loss = 0
            progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")
            
            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_loss = total_loss / len(train_dataloader)
            print(f"Average training loss: {avg_loss:.4f}")
            
            # Validation
            if val_dataloader:
                val_loss = self.validate(val_dataloader)
                print(f"Validation loss: {val_loss:.4f}")
    
    def validate(self, val_dataloader):
        """
        Validation trên validation set.
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
        
        self.model.train()
        return total_loss / len(val_dataloader)
    
    def save_checkpoint(self, checkpoint_path: str, chunk_index: int, metadata: dict = None):
        """
        Save model checkpoint.
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'chunk_index': chunk_index,
            'vocab_size': self.syllable_vocab.get_vocab_size(),
        }
        
        if metadata:
            checkpoint.update(metadata)
            
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint.get('chunk_index', 0)
