import torch
from transformers import RobertaForTokenClassification, RobertaConfig, RobertaTokenizerFast
from tqdm import tqdm
import re
import os
import math
from collections import defaultdict, Counter

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

# --- Intelligent Ranking System ---
class IntelligentRankingSystem:
    """
    Hệ thống ranking thông minh cho multiple suggestions.
    """
    def __init__(self):
        # Frequency scores cho các từ tiếng Việt phổ biến
        self.word_frequencies = {
            'tôi': 0.95, 'tối': 0.3, 'tới': 0.1,
            'đi': 0.9, 'đỉ': 0.01, 'đì': 0.01,
            'học': 0.9, 'hoc': 0.01, 'hòc': 0.01,
            'làm': 0.9, 'lam': 0.05,
            'việc': 0.8, 'viêc': 0.01,
            'với': 0.9, 'voi': 0.05,
            'không': 0.95, 'khong': 0.05,
            'có': 0.95, 'co': 0.05,
            'và': 0.95, 'va': 0.1,
            'này': 0.9, 'nay': 0.2,
            'những': 0.8, 'nhung': 0.2,
            'thể': 0.8, 'the': 0.3,
            'được': 0.9, 'duoc': 0.1,
            'sẽ': 0.9, 'se': 0.1,
            'cho': 0.9, 'cho': 0.9,
            'về': 0.9, 've': 0.1,
            'như': 0.9, 'nhu': 0.1,
            'người': 0.9, 'nguoi': 0.1,
            'thời': 0.8, 'thoi': 0.3,
            'đây': 0.8, 'day': 0.3,
            'đó': 0.9, 'do': 0.2,
            'khi': 0.8, 'khi': 0.8,
            'cả': 0.8, 'ca': 0.3,
            'từ': 0.9, 'tu': 0.2,
            'sau': 0.9, 'sau': 0.9,
            'trước': 0.8, 'truoc': 0.2,
            'nhiều': 0.8, 'nhieu': 0.2,
            'lại': 0.8, 'lai': 0.3,
            'thì': 0.9, 'thi': 0.2,
            'bây': 0.7, 'bay': 0.4,
            'giờ': 0.8, 'gio': 0.3,
            'vào': 0.9, 'vao': 0.1,
            'ra': 0.9, 'ra': 0.9,
            'nếu': 0.8, 'neu': 0.2,
            'đến': 0.9, 'den': 0.1,
            'của': 0.95, 'cua': 0.05,
            'trong': 0.9, 'trong': 0.9,
            'chúng': 0.8, 'chung': 0.3,
            'mình': 0.8, 'minh': 0.3,
            'rất': 0.9, 'rat': 0.1,
            'cũng': 0.8, 'cung': 0.2,
            'phải': 0.8, 'phai': 0.2,
            'còn': 0.8, 'con': 0.3,
            'đã': 0.9, 'da': 0.2,
            'chỉ': 0.8, 'chi': 0.3,
            'một': 0.95, 'mot': 0.05,
            'hai': 0.9, 'hai': 0.9,
            'ba': 0.9, 'ba': 0.9,
            'bốn': 0.7, 'bon': 0.3,
            'năm': 0.9, 'nam': 0.2,
            'sáu': 0.8, 'sau': 0.9,  # conflict với 'sau' - cần context
            'bảy': 0.7, 'bay': 0.4,  # conflict với 'bay' - cần context
            'tám': 0.8, 'tam': 0.2,
            'chín': 0.8, 'chin': 0.2,
            'mười': 0.8, 'muoi': 0.2,
        }
        
        # Common bigrams cho context scoring
        self.common_bigrams = {
            ('tôi', 'đi'): 0.9,
            ('tôi', 'học'): 0.8,
            ('tôi', 'làm'): 0.9,
            ('tôi', 'có'): 0.9,
            ('tôi', 'sẽ'): 0.8,
            ('đi', 'học'): 0.9,
            ('đi', 'làm'): 0.9,
            ('đi', 'chơi'): 0.8,
            ('học', 'bài'): 0.8,
            ('học', 'tập'): 0.8,
            ('làm', 'việc'): 0.9,
            ('làm', 'bài'): 0.8,
            ('vào', 'lúc'): 0.8,
            ('lúc', 'nào'): 0.8,
            ('bây', 'giờ'): 0.9,
            ('lúc', 'này'): 0.8,
            ('thời', 'gian'): 0.9,
            ('không', 'biết'): 0.9,
            ('không', 'có'): 0.9,
            ('có', 'thể'): 0.9,
            ('có', 'được'): 0.8,
            ('được', 'không'): 0.8,
            ('rất', 'nhiều'): 0.8,
            ('rất', 'tốt'): 0.8,
            ('cảm', 'ơn'): 0.9,
            ('xin', 'chào'): 0.9,
            ('chúc', 'mừng'): 0.8,
        }
        
        # Semantic patterns
        self.semantic_patterns = {
            # Time-related patterns
            'time': ['buổi', 'sáng', 'trưa', 'chiều', 'tối', 'đêm', 'hôm', 'ngày', 'tháng', 'năm', 'giờ', 'phút'],
            # Action patterns  
            'action': ['đi', 'đến', 'về', 'làm', 'học', 'chơi', 'ăn', 'uống', 'ngủ', 'thức'],
            # Person patterns
            'person': ['tôi', 'bạn', 'anh', 'chị', 'em', 'cô', 'chú', 'bác'],
            # Question patterns
            'question': ['gì', 'ai', 'đâu', 'khi', 'nào', 'sao', 'thế', 'nào'],
        }
    
    def calculate_frequency_score(self, word):
        """Tính điểm frequency cho từ."""
        return self.word_frequencies.get(word.lower(), 0.1)  # Default low score
    
    def calculate_context_score(self, sentence_words):
        """Tính điểm context dựa trên bigrams."""
        if len(sentence_words) < 2:
            return 0.5
        
        total_score = 0
        count = 0
        
        for i in range(len(sentence_words) - 1):
            bigram = (sentence_words[i].lower(), sentence_words[i + 1].lower())
            if bigram in self.common_bigrams:
                total_score += self.common_bigrams[bigram]
                count += 1
        
        return total_score / max(count, 1) if count > 0 else 0.3
    
    def calculate_semantic_score(self, sentence_words):
        """Tính điểm semantic dựa trên patterns."""
        sentence_lower = [w.lower() for w in sentence_words]
        
        pattern_matches = 0
        total_patterns = len(self.semantic_patterns)
        
        for pattern_name, pattern_words in self.semantic_patterns.items():
            if any(word in sentence_lower for word in pattern_words):
                pattern_matches += 1
        
        # Bonus for grammatically sound patterns
        grammatical_bonus = 0
        if len(sentence_words) >= 2:
            # Subject + verb pattern
            if (sentence_words[0].lower() in ['tôi', 'bạn', 'anh', 'chị'] and
                sentence_words[1].lower() in ['đi', 'học', 'làm', 'có']):
                grammatical_bonus += 0.2
            
            # Time + action pattern  
            if (sentence_words[0].lower() in ['sáng', 'tối', 'trưa', 'chiều'] and
                'đi' in sentence_lower):
                grammatical_bonus += 0.1
        
        base_score = pattern_matches / total_patterns
        return min(base_score + grammatical_bonus, 1.0)
    
    def calculate_length_penalty(self, sentence, original_length):
        """Penalty cho câu quá dài hoặc quá ngắn so với original."""
        length_diff = abs(len(sentence.split()) - original_length)
        if length_diff == 0:
            return 0.0
        elif length_diff == 1:
            return -0.05
        else:
            return -0.1 * length_diff
    
    def rank_suggestions(self, suggestions, original_text, model_scores=None):
        """
        Rank suggestions dựa trên multiple metrics với adaptive scoring.
        
        Args:
            suggestions: List of suggestion strings
            original_text: Original input text
            model_scores: Optional model confidence scores
            
        Returns:
            List of (suggestion, final_score) tuples, sorted by score desc
        """
        original_words = split_to_syllables(original_text)
        original_length = len(original_words)
        
        scored_suggestions = []
        
        for i, suggestion in enumerate(suggestions):
            suggestion_words = split_to_syllables(suggestion)
            
            # Calculate individual scores
            freq_score = sum(self.calculate_frequency_score(word) for word in suggestion_words) / len(suggestion_words)
            context_score = self.calculate_context_score(suggestion_words)
            semantic_score = self.calculate_semantic_score(suggestion_words)
            length_penalty = self.calculate_length_penalty(suggestion, original_length)
            
            # Model confidence score (if available)
            model_score = model_scores[i] if model_scores and i < len(model_scores) else 0.5
            
            # Adaptive weighting dựa trên context
            if len(suggestion_words) <= 2:
                # Short phrases: prioritize frequency và model confidence
                final_score = (
                    0.4 * freq_score +        # 40% frequency for short phrases
                    0.15 * context_score +    # 15% context
                    0.15 * semantic_score +   # 15% semantic
                    0.25 * model_score +      # 25% model confidence
                    0.05 * (1.0 + length_penalty)  # 5% length
                )
            elif len(suggestion_words) <= 4:
                # Medium phrases: balanced approach
                final_score = (
                    0.25 * freq_score +       # 25% frequency
                    0.25 * context_score +    # 25% context
                    0.25 * semantic_score +   # 25% semantic
                    0.15 * model_score +      # 15% model confidence
                    0.10 * (1.0 + length_penalty)  # 10% length
                )
            else:
                # Long phrases: prioritize semantic và context
                final_score = (
                    0.15 * freq_score +       # 15% frequency
                    0.35 * context_score +    # 35% context
                    0.35 * semantic_score +   # 35% semantic
                    0.10 * model_score +      # 10% model confidence
                    0.05 * (1.0 + length_penalty)  # 5% length
                )
            
            # Bonus cho perfect grammatical patterns
            if self._is_perfect_pattern(suggestion_words):
                final_score += 0.1
            
            # Penalty cho unlikely combinations
            if self._has_unlikely_combinations(suggestion_words):
                final_score -= 0.15
            
            scored_suggestions.append((suggestion, final_score))
        
        # Sort by score descending
        scored_suggestions.sort(key=lambda x: x[1], reverse=True)
        return scored_suggestions
    
    def _is_perfect_pattern(self, words):
        """Check for perfect grammatical patterns."""
        if len(words) >= 2:
            # Perfect subject-verb patterns
            if (words[0].lower() in ['tôi', 'bạn', 'anh', 'chị', 'em'] and
                words[1].lower() in ['đi', 'học', 'làm', 'có', 'sẽ', 'đang']):
                return True
            
            # Perfect time-action patterns
            if (words[0].lower() in ['sáng', 'tối', 'trưa', 'chiều'] and
                len(words) >= 2 and words[1].lower() in ['này', 'qua', 'mai']):
                return True
                
        return False
    
    def _has_unlikely_combinations(self, words):
        """Check for unlikely word combinations."""
        unlikely_patterns = [
            # Patterns that are grammatically wrong
            ('tới', 'đi'),  # tới đi doesn't make sense
            ('sáng', 'tối'), # sáng tối in same sentence is weird
            ('học', 'tới'),  # học tới is weird order
        ]
        
        for i in range(len(words) - 1):
            bigram = (words[i].lower(), words[i + 1].lower())
            if bigram in unlikely_patterns:
                return True
                
        return False

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
        
        # Initialize ranking system
        self.ranking_system = IntelligentRankingSystem()

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
    
    def predict_with_ranking(self, unaccented_text, num_suggestions=3, beam_width=10):
        """
        Predict với intelligent ranking system.
        
        Args:
            unaccented_text: Input text without accents
            num_suggestions: Number of suggestions to return
            beam_width: Beam width for initial suggestions (higher = more diverse)
            
        Returns:
            List of (suggestion, score) tuples, ranked by intelligent metrics
        """
        # Get more initial suggestions để có nhiều options cho ranking
        initial_beam_width = max(beam_width, num_suggestions * 2)
        raw_suggestions = self.predict_beam_search(
            unaccented_text, 
            beam_width=initial_beam_width, 
            num_suggestions=initial_beam_width
        )
        
        if not raw_suggestions:
            # Fallback to single prediction
            single_pred = self.predict(unaccented_text)
            return [(single_pred, 1.0)]
        
        # Remove duplicates while preserving order
        unique_suggestions = []
        seen = set()
        for suggestion in raw_suggestions:
            if suggestion not in seen:
                unique_suggestions.append(suggestion)
                seen.add(suggestion)
        
        # Get model confidence scores
        model_scores = self._get_model_confidence_scores(unaccented_text, unique_suggestions)
        
        # Apply intelligent ranking
        ranked_suggestions = self.ranking_system.rank_suggestions(
            unique_suggestions, 
            unaccented_text, 
            model_scores
        )
        
        # Return top N suggestions
        return ranked_suggestions[:num_suggestions]
    
    def _get_model_confidence_scores(self, unaccented_text, suggestions):
        """
        Tính model confidence scores cho các suggestions.
        """
        scores = []
        inputs = self.tokenizer(unaccented_text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=2)
            word_ids = inputs.word_ids(batch_index=0)
            
            for suggestion in suggestions:
                suggestion_words = split_to_syllables(suggestion)
                total_prob = 0.0
                word_count = 0
                previous_word_idx = None
                
                for token_idx, word_idx in enumerate(word_ids):
                    if word_idx is None or word_idx == previous_word_idx:
                        continue
                    if word_idx < len(suggestion_words):
                        syllable = suggestion_words[word_idx]
                        syllable_id = self.syllable_vocab.get_id(syllable)
                        prob = probs[0, token_idx, syllable_id].item()
                        total_prob += prob
                        word_count += 1
                    previous_word_idx = word_idx
                
                avg_prob = total_prob / max(word_count, 1)
                scores.append(avg_prob)
        
        return scores
    
    def predict_with_adaptive_ranking(self, unaccented_text, max_suggestions=5, quality_threshold=0.3, diversity_threshold=0.1):
        """
        Predict với adaptive ranking - mô hình tự quyết định số lượng suggestions.
        
        Args:
            unaccented_text: Input text without accents
            max_suggestions: Maximum number of suggestions to consider
            quality_threshold: Minimum quality score để include suggestion
            diversity_threshold: Minimum score difference để avoid duplicates
            
        Returns:
            List of (suggestion, score) tuples, auto-determined quantity
        """
        # Get initial suggestions với beam search rộng
        initial_suggestions = self.predict_beam_search(
            unaccented_text, 
            beam_width=max_suggestions * 2, 
            num_suggestions=max_suggestions * 2
        )
        
        if not initial_suggestions:
            # Fallback to single prediction
            single_pred = self.predict(unaccented_text)
            return [(single_pred, 1.0)]
        
        # Remove duplicates while preserving order
        unique_suggestions = []
        seen = set()
        for suggestion in initial_suggestions:
            if suggestion not in seen:
                unique_suggestions.append(suggestion)
                seen.add(suggestion)
        
        # Get model confidence scores và apply intelligent ranking
        model_scores = self._get_model_confidence_scores(unaccented_text, unique_suggestions)
        ranked_suggestions = self.ranking_system.rank_suggestions(
            unique_suggestions, 
            unaccented_text, 
            model_scores
        )
        
        # Adaptive filtering dựa trên quality và diversity
        final_suggestions = []
        
        if not ranked_suggestions:
            return [(self.predict(unaccented_text), 1.0)]
        
        # Luôn include suggestion tốt nhất
        best_suggestion = ranked_suggestions[0]
        final_suggestions.append(best_suggestion)
        
        # Chỉ include thêm suggestions nếu chúng đủ quality và diversity
        for suggestion, score in ranked_suggestions[1:]:
            # Check quality threshold
            if score < quality_threshold:
                break
                
            # Check diversity (score difference với previous suggestions)
            if len(final_suggestions) > 0:
                score_diff = abs(score - final_suggestions[-1][1])
                if score_diff < diversity_threshold:
                    continue
            
            # Check if this suggestion adds meaningful value
            if self._is_meaningful_suggestion(suggestion, [s[0] for s in final_suggestions]):
                final_suggestions.append((suggestion, score))
                
                # Stop nếu đã có đủ suggestions (usually 2-3 is good)
                if len(final_suggestions) >= 3:
                    break
        
        # Nếu chỉ có 1 suggestion và score cao, chỉ trả về 1
        if len(final_suggestions) == 1 and final_suggestions[0][1] > 0.8:
            return final_suggestions
        
        # Nếu có multiple suggestions nhưng best score quá cao so với others, chỉ trả về best
        if len(final_suggestions) > 1:
            best_score = final_suggestions[0][1]
            second_score = final_suggestions[1][1]
            
            # Nếu best suggestion quá vượt trội (gap > 0.3), chỉ trả về best
            if best_score - second_score > 0.3:
                return [final_suggestions[0]]
        
        return final_suggestions
    
    def _is_meaningful_suggestion(self, new_suggestion, existing_suggestions):
        """
        Check xem suggestion mới có meaningful không so với existing ones.
        """
        new_words = split_to_syllables(new_suggestion.lower())
        
        for existing in existing_suggestions:
            existing_words = split_to_syllables(existing.lower())
            
            # Check nếu chỉ khác 1 từ và từ đó very similar
            if len(new_words) == len(existing_words):
                diff_count = sum(1 for i in range(len(new_words)) if new_words[i] != existing_words[i])
                
                # Nếu chỉ khác 1 từ
                if diff_count == 1:
                    # Tìm từ khác nhau
                    for i in range(len(new_words)):
                        if new_words[i] != existing_words[i]:
                            new_word = new_words[i]
                            existing_word = existing_words[i]
                            
                            # Check nếu base form giống nhau (remove accents)
                            if remove_vietnamese_accents(new_word) == remove_vietnamese_accents(existing_word):
                                # Chỉ khác accent thì meaningful
                                return True
                            
                            # Check nếu words quá similar (edit distance = 1)
                            if self._edit_distance(new_word, existing_word) <= 1:
                                return False
                
                # Nếu khác quá nhiều từ thì không meaningful trong context này
                elif diff_count > 2:
                    return False
        
        return True
    
    def _edit_distance(self, s1, s2):
        """Simple edit distance calculation."""
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        
        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        return distances[-1]

# --- Dataset Class ---
class AccentSyllableDataset:
    """
    Dataset class cho training data.
    """
    def __init__(self, data_corpus, tokenizer, syllable_vocab, max_length=128):
        self.data_corpus = data_corpus
        self.tokenizer = tokenizer
        self.syllable_vocab = syllable_vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.data_corpus)

    def __getitem__(self, idx):
        sentence = self.data_corpus[idx]
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