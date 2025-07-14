import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
import time
import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict
import logging
from datetime import datetime
import pickle
from model_architecture import EnhancedACausalTCN, ACausalTCN

class ProgressiveTrainer:
    """
    Progressive trainer cho Vietnamese Accent Restoration
    Train từ đầu với hệ thống checkpoint và resume
    """
    
    def __init__(self, corpus_dir="corpus_splitted", 
                 model_type="enhanced", 
                 save_dir="models/progressive",
                 embedding_dim=256,
                 hidden_dim=512,
                 num_layers=8,
                 dropout=0.15,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.corpus_dir = corpus_dir
        self.save_dir = save_dir
        self.device = device
        self.model_type = model_type
        
        # Tạo thư mục lưu trữ
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{save_dir}/logs", exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Build character vocabulary
        self.char_to_idx, self.idx_to_char = self._build_vocabulary()
        self.vocab_size = len(self.char_to_idx)
        
        self.logger.info(f"Vocabulary size: {self.vocab_size}")
        
        # Initialize model
        if model_type == "enhanced":
            self.model = EnhancedACausalTCN(
                vocab_size=self.vocab_size,
                output_size=self.vocab_size,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                use_attention=True
            )
        else:
            self.model = ACausalTCN(
                vocab_size=self.vocab_size,
                output_size=self.vocab_size,
                embedding_dim=embedding_dim // 2,
                hidden_dim=hidden_dim // 2,
                num_layers=num_layers,
                dropout=dropout
            )
        
        self.model.to(self.device)
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
        
        # Training state
        self.training_state = {
            'current_sample_idx': 0,
            'total_samples': 0,
            'global_step': 0,
            'best_val_loss': float('inf'),
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'start_time': None,
            'total_training_time': 0
        }
        
        # Load existing state if available
        self._load_training_state()
        
        # Get list of training files
        self.training_files = self._get_training_files()
        self.training_state['total_samples'] = len(self.training_files)
        
        self.logger.info(f"Found {len(self.training_files)} training files")
        self.logger.info(f"Resume from sample {self.training_state['current_sample_idx']}")
    
    def _setup_logging(self):
        """Setup logging system"""
        log_file = f"{self.save_dir}/logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Tạo logger
        self.logger = logging.getLogger('ProgressiveTrainer')
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def _build_vocabulary(self):
        """Build character vocabulary từ một vài files sample"""
        vocab_file = f"{self.save_dir}/vocabulary.json"
        
        if os.path.exists(vocab_file):
            self.logger.info("Loading existing vocabulary...")
            with open(vocab_file, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
                return vocab_data['char_to_idx'], vocab_data['idx_to_char']
        
        self.logger.info("Building vocabulary from corpus...")
        char_counter = Counter()
        
        # Sample một vài files để build vocabulary
        sample_files = glob.glob(f"{self.corpus_dir}/training_data_*.json")[:10]
        
        for file_path in tqdm(sample_files, desc="Building vocabulary"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                for item in data[:1000]:  # Chỉ lấy 1000 items đầu từ mỗi file
                    char_counter.update(item['input'])
                    char_counter.update(item['target'])
                    
            except Exception as e:
                self.logger.warning(f"Error processing {file_path}: {e}")
                continue
        
        # Build mappings
        special_chars = ['<PAD>', '<UNK>', '<START>', '<END>']
        all_chars = special_chars + [char for char, count in char_counter.most_common() if count >= 2]
        
        char_to_idx = {char: idx for idx, char in enumerate(all_chars)}
        idx_to_char = {idx: char for char, idx in char_to_idx.items()}
        
        # Save vocabulary
        vocab_data = {
            'char_to_idx': char_to_idx,
            'idx_to_char': idx_to_char,
            'vocab_size': len(char_to_idx)
        }
        
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Built vocabulary with {len(char_to_idx)} characters")
        return char_to_idx, idx_to_char
    
    def _get_training_files(self):
        """Get sorted list of training files"""
        pattern = f"{self.corpus_dir}/training_data_*.json"
        files = glob.glob(pattern)
        
        # Sort by number
        def extract_number(filename):
            try:
                return int(filename.split('_')[-1].split('.')[0])
            except:
                return 0
        
        return sorted(files, key=extract_number)
    
    def _load_training_state(self):
        """Load training state from checkpoint"""
        state_file = f"{self.save_dir}/training_state.json"
        
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    saved_state = json.load(f)
                    self.training_state.update(saved_state)
                    
                self.logger.info(f"Loaded training state from {state_file}")
                
                # Load model checkpoint if exists
                checkpoint_file = f"{self.save_dir}/checkpoints/latest_checkpoint.pth"
                if os.path.exists(checkpoint_file):
                    checkpoint = torch.load(checkpoint_file, map_location=self.device)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    
                    self.logger.info("Loaded model checkpoint")
                    
            except Exception as e:
                self.logger.warning(f"Failed to load training state: {e}")
    
    def _save_training_state(self):
        """Save current training state"""
        state_file = f"{self.save_dir}/training_state.json"
        
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(self.training_state, f, ensure_ascii=False, indent=2)
        
        # Save model checkpoint
        checkpoint_file = f"{self.save_dir}/checkpoints/latest_checkpoint.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_state': self.training_state
        }, checkpoint_file)
    
    def _save_best_model(self):
        """Save best model"""
        model_file = f"{self.save_dir}/best_model.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char,
            'model_config': {
                'vocab_size': self.vocab_size,
                'model_type': self.model_type
            },
            'training_state': self.training_state
        }, model_file)
        
        self.logger.info(f"Saved best model to {model_file}")

class SampleDataset(Dataset):
    """Dataset cho một sample file"""
    
    def __init__(self, file_path, char_to_idx, max_length=256):
        self.char_to_idx = char_to_idx
        self.max_length = max_length
        
        # Load data
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Filter valid samples
        self.samples = []
        for item in self.data:
            if isinstance(item, dict) and 'input' in item and 'target' in item:
                input_text = item['input'].strip()
                target_text = item['target'].strip()
                
                if len(input_text) > 0 and len(target_text) > 0:
                    self.samples.append({
                        'input': input_text,
                        'target': target_text
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        input_text = sample['input']
        target_text = sample['target']
        
        # Encode to indices
        input_ids = [self.char_to_idx.get(c, 1) for c in input_text[:self.max_length]]
        target_ids = [self.char_to_idx.get(c, 1) for c in target_text[:self.max_length]]
        
        # Padding
        length = len(input_ids)
        input_ids += [0] * (self.max_length - len(input_ids))
        target_ids += [0] * (self.max_length - len(target_ids))
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'length': length,
            'input_text': input_text,
            'target_text': target_text
        }

def train_on_sample(trainer, file_path, epochs=3, batch_size=32):
    """Train model trên một sample file"""
    trainer.logger.info(f"Training on sample: {os.path.basename(file_path)}")
    
    # Create dataset
    dataset = SampleDataset(file_path, trainer.char_to_idx)
    
    if len(dataset) == 0:
        trainer.logger.warning(f"No valid samples in {file_path}")
        return {'train_loss': 0, 'val_loss': 0, 'val_acc': 0}
    
    # Split train/val/test
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, temp_dataset = torch.utils.data.random_split(
        dataset, [train_size, len(dataset) - train_size]
    )
    val_dataset, test_dataset = torch.utils.data.random_split(
        temp_dataset, [val_size, test_size]
    )
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    trainer.logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Training loop
    trainer.model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(trainer.device)
            target_ids = batch['target_ids'].to(trainer.device)
            
            trainer.optimizer.zero_grad()
            
            outputs = trainer.model(input_ids)
            
            # Calculate loss
            outputs_flat = outputs.view(-1, outputs.size(-1))
            targets_flat = target_ids.view(-1)
            loss = trainer.criterion(outputs_flat, targets_flat)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=1.0)
            trainer.optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            trainer.training_state['global_step'] += 1
            
            progress_bar.set_postfix({'Loss': f"{loss.item():.4f}"})
        
        avg_train_loss = epoch_loss / max(num_batches, 1)
        trainer.training_state['train_losses'].append(avg_train_loss)
        
        # Validation
        val_loss, val_acc = validate_model(trainer, val_loader)
        trainer.training_state['val_losses'].append(val_loss)
        
        trainer.logger.info(
            f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )
        
        # Save best model
        if val_loss < trainer.training_state['best_val_loss']:
            trainer.training_state['best_val_loss'] = val_loss
            trainer._save_best_model()
    
    return {
        'train_loss': trainer.training_state['train_losses'][-1],
        'val_loss': trainer.training_state['val_losses'][-1],
        'val_acc': val_acc
    }

def validate_model(trainer, val_loader):
    """Validate model"""
    trainer.model.eval()
    total_loss = 0
    total_correct = 0
    total_chars = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(trainer.device)
            target_ids = batch['target_ids'].to(trainer.device)
            lengths = batch['length']
            
            outputs = trainer.model(input_ids)
            
            # Calculate loss
            outputs_flat = outputs.view(-1, outputs.size(-1))
            targets_flat = target_ids.view(-1)
            loss = trainer.criterion(outputs_flat, targets_flat)
            
            total_loss += loss.item()
            
            # Calculate accuracy
            predictions = outputs.argmax(dim=-1)
            for i, length in enumerate(lengths):
                pred = predictions[i][:length]
                target = target_ids[i][:length]
                mask = target != 0
                
                if mask.sum() > 0:
                    total_correct += ((pred == target) * mask).sum().item()
                    total_chars += mask.sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = total_correct / max(total_chars, 1)
    
    return avg_loss, accuracy

def main():
    """Main training function"""
    print("Starting Progressive Training for Vietnamese Accent Restoration")
    print("=" * 80)
    
    # Check corpus directory
    corpus_dir = "corpus_splitted"
    if not os.path.exists(corpus_dir):
        print(f"Corpus directory not found: {corpus_dir}")
        return
    
    # Initialize trainer
    trainer = ProgressiveTrainer(
        corpus_dir=corpus_dir,
        model_type="enhanced",  # hoặc "standard"
        embedding_dim=256,
        hidden_dim=512,
        num_layers=8,
        dropout=0.15
    )
    
    trainer.logger.info("Starting progressive training...")
    trainer.training_state['start_time'] = time.time()
    
    # Training loop qua các samples
    start_idx = trainer.training_state['current_sample_idx']
    
    try:
        for i in range(start_idx, len(trainer.training_files)):
            file_path = trainer.training_files[i]
            trainer.training_state['current_sample_idx'] = i
            
            trainer.logger.info(f"Processing sample {i+1}/{len(trainer.training_files)}")
            
            # Train on current sample
            sample_results = train_on_sample(
                trainer, file_path, 
                epochs=2 if i == 0 else 1,  # First sample gets more epochs
                batch_size=32
            )
            
            trainer.logger.info(
                f"Sample {i+1} completed - "
                f"Train Loss: {sample_results['train_loss']:.4f}, "
                f"Val Loss: {sample_results['val_loss']:.4f}, "
                f"Val Acc: {sample_results['val_acc']:.4f}"
            )
            
            # Save state after each sample
            trainer._save_training_state()
            
            # Log progress
            elapsed_time = time.time() - trainer.training_state['start_time']
            progress = (i + 1) / len(trainer.training_files)
            estimated_total = elapsed_time / progress
            remaining_time = estimated_total - elapsed_time
            
            trainer.logger.info(
                f"Progress: {progress*100:.1f}% - "
                f"Elapsed: {elapsed_time/3600:.1f}h - "
                f"Remaining: {remaining_time/3600:.1f}h"
            )
            
    except KeyboardInterrupt:
        trainer.logger.info("Training interrupted by user")
        trainer._save_training_state()
        
    except Exception as e:
        trainer.logger.error(f"Training error: {e}")
        trainer._save_training_state()
        raise
    
    trainer.logger.info("Progressive training completed!")
    
    # Final model save
    final_model_path = f"{trainer.save_dir}/final_model.pth"
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'char_to_idx': trainer.char_to_idx,
        'idx_to_char': trainer.idx_to_char,
        'model_config': {
            'vocab_size': trainer.vocab_size,
            'model_type': trainer.model_type
        },
        'training_state': trainer.training_state
    }, final_model_path)
    
    trainer.logger.info(f"Final model saved to {final_model_path}")

if __name__ == "__main__":
    main() 