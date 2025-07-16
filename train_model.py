import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
from model_architecture import VietnameseAccentRestorer

class AccentDataset(Dataset):
    """Dataset for Vietnamese accent restoration"""
    
    def __init__(self, input_texts, target_texts, char_to_idx, max_length=128):
        self.input_texts = input_texts
        self.target_texts = target_texts
        self.char_to_idx = char_to_idx
        self.max_length = max_length
    
    def __len__(self):
        return len(self.input_texts)
    
    def __getitem__(self, idx):
        input_text = self.input_texts[idx]
        target_text = self.target_texts[idx]
        
        # Truncate if too long
        if len(input_text) > self.max_length:
            input_text = input_text[:self.max_length]
            target_text = target_text[:self.max_length]
        
        # Encode
        input_ids = [self.char_to_idx.get(char, 0) for char in input_text]
        target_ids = [self.char_to_idx.get(char, 0) for char in target_text]
        
        # Padding
        input_ids += [0] * (self.max_length - len(input_ids))
        target_ids += [0] * (self.max_length - len(target_ids))
        
        return {
            'input_ids': torch.tensor(input_ids[:self.max_length], dtype=torch.long),
            'target_ids': torch.tensor(target_ids[:self.max_length], dtype=torch.long),
            'length': torch.tensor(len(input_text), dtype=torch.long)
        }

class AccentRestorationTrainer:
    """
    Simple trainer for Vietnamese accent restoration
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.model.to(device)
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        self.optimizer = optim.AdamW(
            model.model.parameters(),
            lr=2e-3,
            weight_decay=1e-4,
            betas=(0.9, 0.95)
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=15, eta_min=1e-5
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience = 3
        self.patience_counter = 0
    
    def calculate_accuracy(self, predictions, targets, lengths):
        """Calculate accuracy"""
        predictions = predictions.argmax(dim=-1)
        correct = 0
        total = 0
        
        for i, length in enumerate(lengths):
            pred = predictions[i][:length]
            target = targets[i][:length]
            mask = target != 0
            if mask.sum() > 0:
                correct += ((pred == target) * mask).sum().item()
                total += mask.sum().item()
        
        return correct / total if total > 0 else 0
    
    def train_epoch(self, train_loader):
        """Training epoch"""
        self.model.model.train()
        total_loss = 0
        total_acc = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            lengths = batch['length']
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model.model(input_ids)
            outputs_flat = outputs.view(-1, outputs.size(-1))
            targets_flat = target_ids.view(-1)
            
            # Loss
            loss = self.criterion(outputs_flat, targets_flat)
            
            # Accuracy
            acc = self.calculate_accuracy(outputs, target_ids, lengths)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_acc += acc
        
        return total_loss / len(train_loader), total_acc / len(train_loader)
    
    def validate_epoch(self, val_loader):
        """Validation epoch"""
        self.model.model.eval()
        total_loss = 0
        total_acc = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                lengths = batch['length']
                
                outputs = self.model.model(input_ids)
                outputs_flat = outputs.view(-1, outputs.size(-1))
                targets_flat = target_ids.view(-1)
                
                loss = self.criterion(outputs_flat, targets_flat)
                acc = self.calculate_accuracy(outputs, target_ids, lengths)
                
                total_loss += loss.item()
                total_acc += acc
        
        return total_loss / len(val_loader), total_acc / len(val_loader)
    
    def train(self, train_loader, val_loader, num_epochs, save_dir="models"):
        """Main training loop with early stopping"""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Training - {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Parameters: {sum(p.numel() for p in self.model.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
            print(f"Val: Loss={val_loss:.4f}, Acc={val_acc:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # Save best model
                best_path = os.path.join(save_dir, "best_model.pth")
                self.model.save_model(best_path)
                print(f"Saved best: {best_path}")
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
        
        print(f"Training complete! Best val loss: {self.best_val_loss:.4f}")

def load_viet74k_data(data_file):
    """Load data from Viet74K.txt"""
    from unidecode import unidecode
    input_texts = []
    target_texts = []
    
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and len(line) >= 2:
                # Create pair (no accent, with accent)
                with_accent = line
                no_accent = unidecode(with_accent)
                
                # Only add if different (has accent)
                if no_accent != with_accent and 2 <= len(with_accent) <= 50:
                    input_texts.append(no_accent)
                    target_texts.append(with_accent)
    
    return input_texts, target_texts

def main():
    """Basic training main function"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load Viet74K data
    viet74k_file = "data/Viet74K_clean.txt"
    if not os.path.exists(viet74k_file):
        print(f"Data file not found: {viet74k_file}")
        return
        
    print("Loading Viet74K data...")
    input_texts, target_texts = load_viet74k_data(viet74k_file)
    print(f"Loaded {len(input_texts)} samples")
    
    # Split train/val
    val_size = min(10000, len(input_texts) // 10)
    train_inputs = input_texts[val_size:]
    train_targets = target_texts[val_size:]
    val_inputs = input_texts[:val_size]
    val_targets = target_texts[:val_size]
    
    print(f"Train: {len(train_inputs)}, Val: {len(val_inputs)}")
    
    # Model
    model = VietnameseAccentRestorer(use_enhanced_model=False)
    
    # Dataset
    train_dataset = AccentDataset(train_inputs, train_targets, model.char_to_idx, 128)
    val_dataset = AccentDataset(val_inputs, val_targets, model.char_to_idx, 128)
    
    batch_size = 128 if device.type == 'cuda' else 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Trainer
    trainer = AccentRestorationTrainer(model, device)
    
    # Training
    trainer.train(train_loader, val_loader, num_epochs=15, save_dir="models")
    
    print("Training complete!")

if __name__ == "__main__":
    main() 