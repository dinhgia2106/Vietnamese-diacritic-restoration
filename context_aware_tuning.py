import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np
import json
import os
import re
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict
import unicodedata
import random
from model_architecture import VietnameseAccentRestorer

def remove_accents(text):
    """Remove Vietnamese accents from text"""
    accent_map = {
        'á': 'a', 'à': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
        'ă': 'a', 'ắ': 'a', 'ằ': 'a', 'ẳ': 'a', 'ẵ': 'a', 'ặ': 'a',
        'â': 'a', 'ấ': 'a', 'ầ': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ậ': 'a',
        'é': 'e', 'è': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
        'ê': 'e', 'ế': 'e', 'ề': 'e', 'ể': 'e', 'ễ': 'e', 'ệ': 'e',
        'í': 'i', 'ì': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
        'ó': 'o', 'ò': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
        'ô': 'o', 'ố': 'o', 'ồ': 'o', 'ổ': 'o', 'ỗ': 'o', 'ộ': 'o',
        'ơ': 'o', 'ớ': 'o', 'ờ': 'o', 'ở': 'o', 'ỡ': 'o', 'ợ': 'o',
        'ú': 'u', 'ù': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
        'ư': 'u', 'ứ': 'u', 'ừ': 'u', 'ử': 'u', 'ữ': 'u', 'ự': 'u',
        'ý': 'y', 'ỳ': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y',
        'đ': 'd'
    }
    result = ""
    for char in text:
        result += accent_map.get(char.lower(), char)
    return result

def clean_text(text):
    """Clean and normalize Vietnamese text"""
    text = unicodedata.normalize('NFC', text)
    # Remove URLs, emails, phone numbers
    text = re.sub(r'http[s]?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    text = re.sub(r'\b\d{10,11}\b', '', text)
    
    # Keep only valid Vietnamese characters
    valid_chars = r'[a-zA-Zàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ\s.,!?;:()\-\[\]{}"\'/'
    text = re.sub(f'[^{valid_chars}]', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

class StreamingDataset(IterableDataset):
    """Efficient streaming dataset for large text files"""
    
    def __init__(self, data_files, char_to_idx, max_length=256, context_window=5, 
                 is_test=False, test_ratio=0.01, shuffle_buffer_size=10000):
        self.data_files = data_files if isinstance(data_files, list) else [data_files]
        self.char_to_idx = char_to_idx
        self.max_length = max_length
        self.context_window = context_window
        self.is_test = is_test
        self.test_ratio = test_ratio
        self.shuffle_buffer_size = shuffle_buffer_size
        
        # Cache directories
        os.makedirs("models/preprocessed", exist_ok=True)
        self.preprocessed_stats_path = "models/preprocessed/dataset_stats.json"
        self.context_dict_path = "models/preprocessed/context_dict.json"
        
        # Load or build preprocessed data
        if self._check_preprocessed_data():
            self._load_preprocessed_data()
        else:
            self.total_lines, self.file_line_counts = self._count_total_lines()
            self.test_lines = int(self.total_lines * test_ratio)
            self.train_lines = self.total_lines - self.test_lines
            self.context_dict = self._build_context_dictionary()
            self._save_preprocessed_data()
    
    def _count_total_lines(self):
        """Count total valid lines in all files"""
        total_lines = 0
        file_line_counts = {}
        
        for file_path in self.data_files:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    total_file_lines = sum(1 for _ in f)
                
                valid_lines = 0
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in tqdm(f, total=total_file_lines, desc=f"Counting {os.path.basename(file_path)}"):
                        if line.strip() and len(line.strip()) >= 10:
                            valid_lines += 1
                
                file_line_counts[file_path] = valid_lines
                total_lines += valid_lines
            else:
                file_line_counts[file_path] = 0
        
        return total_lines, file_line_counts
    
    def _build_context_dictionary(self):
        """Build context dictionary for ranking"""
        context_dict = defaultdict(lambda: defaultdict(int))
        processed_lines = 0
        
        # Sample 10% or max 100k lines for context building
        sample_ratio = min(0.1, 100000 / self.total_lines)
        estimated_samples = int(self.total_lines * sample_ratio)
        
        pbar = tqdm(total=estimated_samples, desc="Building context")
        
        for file_path in self.data_files:
            if not os.path.exists(file_path):
                continue
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if random.random() > sample_ratio:
                        continue
                    
                    line = line.strip()
                    if not line or len(line) < 10 or len(line) > 300:
                        continue
                    
                    cleaned = clean_text(line)
                    if not cleaned:
                        continue
                    
                    words = cleaned.lower().split()
                    for i, word in enumerate(words):
                        if len(word) > 2:
                            start = max(0, i - self.context_window)
                            end = min(len(words), i + self.context_window + 1)
                            context_words = words[start:i] + words[i+1:end]
                            
                            for ctx_word in context_words:
                                if len(ctx_word) > 2:
                                    context_dict[word][ctx_word] += 1
                    
                    processed_lines += 1
                    pbar.update(1)
                    
                    if processed_lines >= estimated_samples:
                        break
                
                if processed_lines >= estimated_samples:
                    break
        
        pbar.close()
        
        # Normalize scores
        for word in tqdm(context_dict, desc="Normalizing"):
            total = sum(context_dict[word].values())
            if total > 0:
                for ctx_word in context_dict[word]:
                    context_dict[word][ctx_word] /= total
        
        return dict(context_dict)
    
    def _check_preprocessed_data(self):
        """Check if preprocessed data exists and is valid"""
        if not os.path.exists(self.preprocessed_stats_path) or not os.path.exists(self.context_dict_path):
            return False
        
        try:
            with open(self.preprocessed_stats_path, 'r', encoding='utf-8') as f:
                stats = json.load(f)
            
            if 'file_info' not in stats:
                return False
            
            # Check if source files changed
            for file_path in self.data_files:
                if file_path not in stats['file_info']:
                    return False
                
                if not os.path.exists(file_path):
                    return False
                
                current_mtime = os.path.getmtime(file_path)
                stored_mtime = stats['file_info'][file_path].get('mtime', 0)
                
                if abs(current_mtime - stored_mtime) > 1:
                    return False
            
            return True
        except Exception:
            return False
    
    def _load_preprocessed_data(self):
        """Load preprocessed data"""
        with open(self.preprocessed_stats_path, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        
        self.total_lines = stats['total_lines']
        self.file_line_counts = stats['file_line_counts']
        self.test_lines = int(self.total_lines * self.test_ratio)
        self.train_lines = self.total_lines - self.test_lines
        
        with open(self.context_dict_path, 'r', encoding='utf-8') as f:
            self.context_dict = json.load(f)
    
    def _save_preprocessed_data(self):
        """Save preprocessed data"""
        file_info = {}
        for file_path in self.data_files:
            if os.path.exists(file_path):
                file_info[file_path] = {
                    'mtime': os.path.getmtime(file_path),
                    'size': os.path.getsize(file_path),
                    'line_count': self.file_line_counts.get(file_path, 0)
                }
        
        stats = {
            'total_lines': self.total_lines,
            'file_line_counts': self.file_line_counts,
            'test_ratio': self.test_ratio,
            'context_window': self.context_window,
            'file_info': file_info,
            'created_time': time.time()
        }
        
        with open(self.preprocessed_stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        with open(self.context_dict_path, 'w', encoding='utf-8') as f:
            json.dump(self.context_dict, f, ensure_ascii=False, indent=2)
    
    def _get_line_assignment(self, global_line_idx):
        """Determine if line belongs to train or test"""
        return hash(str(global_line_idx)) % 100 < (self.test_ratio * 100)
    
    def _generate_samples(self):
        """Generate samples from files"""
        global_line_idx = 0
        
        for file_path in self.data_files:
            if not os.path.exists(file_path):
                continue
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or len(line) < 10 or len(line) > 300:
                        continue
                    
                    is_test_line = self._get_line_assignment(global_line_idx)
                    
                    if is_test_line != self.is_test:
                        global_line_idx += 1
                        continue
                    
                    cleaned = clean_text(line)
                    if not cleaned:
                        global_line_idx += 1
                        continue
                    
                    no_accent = remove_accents(cleaned)
                    
                    if no_accent != cleaned:
                        yield {
                            'input_text': no_accent,
                            'target_text': cleaned,
                            'line_id': global_line_idx,
                            'file_path': file_path
                        }
                    
                    global_line_idx += 1
    
    def __iter__(self):
        """Iterator with shuffle buffer"""
        sample_generator = self._generate_samples()
        
        if not self.is_test and self.shuffle_buffer_size > 0:
            buffer = []
            
            for sample in sample_generator:
                buffer.append(sample)
                
                if len(buffer) >= self.shuffle_buffer_size:
                    random.shuffle(buffer)
                    for item in buffer:
                        yield self._process_sample(item)
                    buffer = []
            
            if buffer:
                random.shuffle(buffer)
                for item in buffer:
                    yield self._process_sample(item)
        else:
            for sample in sample_generator:
                yield self._process_sample(sample)
    
    def _process_sample(self, sample):
        """Process sample for model input"""
        input_text = sample['input_text']
        target_text = sample['target_text']
        
        input_ids = [self.char_to_idx.get(c, 1) for c in input_text[:self.max_length]]
        target_ids = [self.char_to_idx.get(c, 1) for c in target_text[:self.max_length]]
        
        length = len(input_ids)
        input_ids += [0] * (self.max_length - len(input_ids))
        target_ids += [0] * (self.max_length - len(target_ids))
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'length': length,
            'original_input': input_text,
            'original_target': target_text
        }

class ContextAwareRanker:
    """Context-aware ranking system for predictions"""
    
    def __init__(self, context_dict, n_gram_weights=None):
        self.context_dict = context_dict
        self.n_gram_weights = n_gram_weights or {1: 0.1, 2: 0.3, 3: 0.6}
        self.word_freq = self._build_word_frequency()
    
    def _build_word_frequency(self):
        """Build word frequency dictionary"""
        word_freq = defaultdict(int)
        
        for word in self.context_dict:
            word_freq[word] += sum(self.context_dict[word].values())
        
        total = sum(word_freq.values())
        if total > 0:
            for word in word_freq:
                word_freq[word] /= total
        
        return dict(word_freq)
    
    def calculate_context_score(self, prediction, context_words):
        """Calculate context score for prediction"""
        words = prediction.lower().split()
        total_score = 0.0
        word_count = 0
        
        for word in words:
            if len(word) > 2:
                word_score = 0.0
                
                if word in self.context_dict:
                    for ctx_word in context_words:
                        if ctx_word in self.context_dict[word]:
                            word_score += self.context_dict[word][ctx_word]
                
                freq_score = self.word_freq.get(word, 0.0001)
                total_score += (word_score * 0.7 + freq_score * 0.3)
                word_count += 1
        
        return total_score / max(word_count, 1)
    
    def calculate_fluency_score(self, prediction):
        """Calculate n-gram fluency score"""
        words = prediction.lower().split()
        if len(words) < 2:
            return 0.5
        
        total_score = 0.0
        total_weight = 0.0
        
        for n in self.n_gram_weights:
            if len(words) >= n:
                n_gram_score = 0.0
                count = 0
                
                for i in range(len(words) - n + 1):
                    n_gram = ' '.join(words[i:i+n])
                    n_gram_score += self._score_pattern(n_gram)
                    count += 1
                
                if count > 0:
                    weight = self.n_gram_weights[n]
                    total_score += (n_gram_score / count) * weight
                    total_weight += weight
        
        return total_score / max(total_weight, 1)
    
    def _score_pattern(self, text):
        """Score text pattern"""
        score = 0.5
        
        # Bonus for common Vietnamese words
        if re.search(r'\b(là|của|và|với|trong|ngoài|trên|dưới)\b', text):
            score += 0.2
        
        # Penalty for repeated characters
        if re.search(r'(.)\1{2,}', text):
            score -= 0.3
        
        # Accent ratio check
        accented_chars = len(re.findall(r'[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]', text))
        total_chars = len(re.findall(r'[a-zA-Zàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]', text))
        
        if total_chars > 0:
            accent_ratio = accented_chars / total_chars
            if 0.2 <= accent_ratio <= 0.5:
                score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def rank_predictions(self, predictions, input_text):
        """Rank predictions by context and quality"""
        if not predictions:
            return predictions
        
        input_words = [w.lower() for w in input_text.split() if len(w) > 2]
        scored_predictions = []
        
        for pred in predictions:
            context_score = self.calculate_context_score(pred, input_words)
            fluency_score = self.calculate_fluency_score(pred)
            similarity_score = self._calculate_similarity_score(pred, input_text)
            
            final_score = (
                context_score * 0.4 +
                fluency_score * 0.3 +
                similarity_score * 0.3
            )
            
            scored_predictions.append((pred, final_score, {
                'context': context_score,
                'fluency': fluency_score,
                'similarity': similarity_score
            }))
        
        scored_predictions.sort(key=lambda x: x[1], reverse=True)
        return scored_predictions
    
    def _calculate_similarity_score(self, prediction, input_text):
        """Calculate similarity score"""
        pred_chars = set(prediction.lower())
        input_chars = set(input_text.lower())
        
        if not input_chars:
            return 0.0
        
        intersection = len(pred_chars & input_chars)
        union = len(pred_chars | input_chars)
        
        return intersection / max(union, 1)

class ContextAwareFinetuner:
    """Context-aware fine-tuner for accent restoration models"""
    
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu',
                 ranking_weight=0.3, ranking_margin=1.0, use_enhanced_model=False):
        self.device = device
        
        self.restorer = VietnameseAccentRestorer(model_path, use_enhanced_model=use_enhanced_model)
        self.restorer.model.to(self.device)
        
        self.optimizer = optim.AdamW(
            self.restorer.model.parameters(),
            lr=1e-4,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # Get label smoothing from model if available
        label_smoothing_val = getattr(self.restorer.model, 'label_smoothing', 0.1)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=label_smoothing_val)
        self.ranking_weight = ranking_weight
        self.ranking_margin = ranking_margin
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.restorer.model.train()
        total_loss = 0
        total_accuracy = 0
        total_ranking_loss = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            lengths = batch['length']
            
            self.optimizer.zero_grad()
            
            outputs = self.restorer.model(input_ids)
            
            outputs_flat = outputs.view(-1, outputs.size(-1))
            targets_flat = target_ids.view(-1)
            classification_loss = self.criterion(outputs_flat, targets_flat)
            
            # Calculate ranking loss
            ranking_loss = self._calculate_ranking_loss(outputs, target_ids, lengths)
            
            total_loss_value = classification_loss + self.ranking_weight * ranking_loss
            
            total_loss_value.backward()
            torch.nn.utils.clip_grad_norm_(self.restorer.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            accuracy = self._calculate_accuracy(outputs, target_ids, lengths)
            
            total_loss += classification_loss.item()
            total_ranking_loss += ranking_loss.item()
            total_accuracy += accuracy
            
            progress_bar.set_postfix({
                'Loss': f"{classification_loss.item():.4f}",
                'Rank Loss': f"{ranking_loss.item():.4f}",
                'Acc': f"{accuracy:.4f}"
            })
        
        return {
            'loss': total_loss / len(train_loader),
            'ranking_loss': total_ranking_loss / len(train_loader),
            'accuracy': total_accuracy / len(train_loader)
        }
    
    def _calculate_ranking_loss(self, outputs, target_ids, lengths):
        """Calculate ranking loss using pairwise comparison"""
        # Calculate log probabilities
        log_probs = F.log_softmax(outputs, dim=-1)
        
        # Get log-prob of target sequence
        target_log_probs = torch.gather(log_probs, 2, target_ids.unsqueeze(2)).squeeze(2)
        
        # Create negative samples (second-best predictions)
        second_best_ids = torch.topk(outputs, 2, dim=-1)[1][:, :, 1]
        negative_log_probs = torch.gather(log_probs, 2, second_best_ids.unsqueeze(2)).squeeze(2)
        
        # Create mask to ignore padding
        mask = (target_ids != 0).float()
        
        # Calculate sequence-level scores
        target_scores = (target_log_probs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        negative_scores = (negative_log_probs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
        # Pairwise hinge loss
        ranking_loss = F.relu(self.ranking_margin - (target_scores - negative_scores)).mean()
        
        return ranking_loss
    
    def _calculate_accuracy(self, outputs, targets, lengths):
        """Calculate accuracy"""
        predictions = outputs.argmax(dim=-1)
        correct = 0
        total = 0
        
        for i, length in enumerate(lengths):
            pred = predictions[i][:length]
            target = targets[i][:length]
            mask = target != 0
            
            if mask.sum() > 0:
                correct += ((pred == target) * mask).sum().item()
                total += mask.sum().item()
        
        return correct / max(total, 1)
    
    def validate_epoch(self, val_loader, max_batches=50):
        """Validate for one epoch"""
        self.restorer.model.eval()
        total_loss = 0
        total_accuracy = 0
        batch_count = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", total=max_batches):
                if batch_count >= max_batches:
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                lengths = batch['length']
                
                outputs = self.restorer.model(input_ids)
                
                outputs_flat = outputs.view(-1, outputs.size(-1))
                targets_flat = target_ids.view(-1)
                loss = self.criterion(outputs_flat, targets_flat)
                
                total_loss += loss.item()
                total_accuracy += self._calculate_accuracy(outputs, target_ids, lengths)
                
                batch_count += 1
        
        return {
            'loss': total_loss / max(batch_count, 1),
            'accuracy': total_accuracy / max(batch_count, 1)
        }
    
    def fine_tune(self, data_files, epochs=5, batch_size=256, save_dir="models"):
        """Fine-tune model"""
        available_files = [f for f in data_files if os.path.exists(f)]
        
        if not available_files:
            raise ValueError("No valid data files found")
        
        train_dataset = StreamingDataset(
            data_files=available_files,
            char_to_idx=self.restorer.char_to_idx,
            is_test=False
        )
        
        val_dataset = StreamingDataset(
            data_files=available_files,
            char_to_idx=self.restorer.char_to_idx,
            is_test=True
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
        
        ranker = ContextAwareRanker(train_dataset.context_dict)
        
        best_val_acc = 0.0
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=2
        )
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            train_stats = self.train_epoch(train_loader)
            val_stats = self.validate_epoch(val_loader)
            
            scheduler.step(val_stats['accuracy'])
            
            print(f"Train Loss: {train_stats['loss']:.4f}, Train Acc: {train_stats['accuracy']:.4f}")
            print(f"Val Loss: {val_stats['loss']:.4f}, Val Acc: {val_stats['accuracy']:.4f}")
            
            if val_stats['accuracy'] > best_val_acc:
                best_val_acc = val_stats['accuracy']
                
                os.makedirs(save_dir, exist_ok=True)
                model_path = os.path.join(save_dir, "context_aware_best_model.pth")
                self.restorer.save_model(model_path)
                
                ranker_path = os.path.join(save_dir, "context_ranker.json")
                with open(ranker_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'context_dict': train_dataset.context_dict,
                        'word_freq': ranker.word_freq,
                        'n_gram_weights': ranker.n_gram_weights
                    }, f, ensure_ascii=False, indent=2)
                
                print(f"Saved new best model: {model_path}")
        
        return best_val_acc

def main():
    """Main training function"""
    base_model_path = "models/best_model.pth"
    data_files = ["data/cleaned_comments.txt", "data/corpus-full.txt"]
    
    if not os.path.exists(base_model_path):
        print(f"Base model not found: {base_model_path}")
        return
    
    available_files = [f for f in data_files if os.path.exists(f)]
    if not available_files:
        print("No data files found")
        return
    
    total_size = sum(os.path.getsize(f) for f in available_files) / (1024 * 1024 * 1024)
    use_enhanced = total_size < 1.0  # Use enhanced for datasets < 1GB
    
    finetuner = ContextAwareFinetuner(
        base_model_path,
        ranking_weight=0.3,
        ranking_margin=1.0,
        use_enhanced_model=use_enhanced
    )
    
    print(f"Using {'Enhanced' if use_enhanced else 'Legacy'} model architecture")
    print(f"Data size: {total_size:.2f} GB")
    
    best_score = finetuner.fine_tune(
        data_files=available_files,
        epochs=3,
        batch_size=1024
    )
    
    print(f"Best accuracy: {best_score:.4f}")

if __name__ == "__main__":
    main()