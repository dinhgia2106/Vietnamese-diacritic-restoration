import torch
import time
import os
import json
from model_architecture import VietnameseAccentRestorer

# Import context ranker từ context_aware_tuning
try:
    from context_aware_tuning import ContextAwareRanker
    CONTEXT_RANKER_AVAILABLE = True
except ImportError:
    CONTEXT_RANKER_AVAILABLE = False
    ContextAwareRanker = None

class VietnameseAccentDemo:
    """
    Demo cho mô hình phục hồi dấu tiếng Việt
    """
    
    def __init__(self, model_path=None, ranker_path=None, word_dict_path=None):
        """
        Khởi tạo demo
        """
        print("Đang tải mô hình...")
        self.restorer = VietnameseAccentRestorer(model_path, use_enhanced_model=False)
        
        if model_path and os.path.exists(model_path):
            print(f"Đã tải mô hình từ: {model_path}")
        else:
            print("Sử dụng mô hình chưa được huấn luyện (chỉ để test kiến trúc)")
        
        # Load word dictionary nếu có
        self.word_dictionary = {}
        if word_dict_path and os.path.exists(word_dict_path):
            try:
                print(f"Đang tải word dictionary từ: {word_dict_path}")
                with open(word_dict_path, 'r', encoding='utf-8') as f:
                    self.word_dictionary = json.load(f)
                print(f"Đã tải {len(self.word_dictionary)} entries từ word dictionary")
            except Exception as e:
                print(f"Lỗi khi tải word dictionary: {e}")
                self.word_dictionary = {}
        else:
            print("Không sử dụng word dictionary - chỉ sử dụng model")
        
        # Load context ranker nếu có
        self.ranker = None
        if CONTEXT_RANKER_AVAILABLE and ranker_path and os.path.exists(ranker_path):
            try:
                print(f"Đang tải context ranker từ: {ranker_path}")
                with open(ranker_path, 'r', encoding='utf-8') as f:
                    ranker_data = json.load(f)
                
                # Convert string keys thành int keys cho n_gram_weights
                n_gram_weights = ranker_data.get('n_gram_weights', {1: 0.1, 2: 0.3, 3: 0.6})
                if n_gram_weights and isinstance(list(n_gram_weights.keys())[0], str):
                    n_gram_weights = {int(k): v for k, v in n_gram_weights.items()}
                
                self.ranker = ContextAwareRanker(
                    context_dict=ranker_data['context_dict'],
                    n_gram_weights=n_gram_weights
                )
                self.ranker.word_freq = ranker_data.get('word_freq', {})
                print("Đã tải context ranker thành công!")
            except Exception as e:
                print(f"Lỗi khi tải context ranker: {e}")
                self.ranker = None
        else:
            print("Không sử dụng context ranker - chỉ sử dụng model cơ bản")
        
        # Các câu test mẫu
        self.test_sentences = [
            "toi di hoc",
            "chung ta se thanh cong",
            "viet nam la dat nuoc xinh dep",
            "hom nay troi dep",
            "cam on ban rat nhieu",
            "xin chao moi nguoi",
            "chuc ban ngay tot lanh",
            "toi thich hoc tieng viet",
            "ha noi la thu do cua viet nam",
            "pho la mon an truyen thong"
        ]
        
        # Các từ đơn test cho dictionary lookup
        self.test_single_words = [
            "toi", "an", "di", "hoc", "viet", "nam", "ha", "noi",
            "cam", "on", "xin", "chao", "dep", "tot", "lam"
        ]
        
        # Kết quả mong đợi (để so sánh)
        self.expected_results = [
            "tôi đi học",
            "chúng ta sẽ thành công",
            "việt nam là đất nước xinh đẹp",
            "hôm nay trời đẹp",
            "cảm ơn bạn rất nhiều",
            "xin chào mọi người",
            "chúc bạn ngày tốt lành",
            "tôi thích học tiếng việt",
            "hà nội là thủ đô của việt nam",
            "phở là món ăn truyền thống"
        ]
    
    def _remove_accents(self, text):
        """Remove accents for dictionary lookup"""
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
            result += accent_map.get(char.lower(), char.lower())
        return result
    
    def _is_single_word(self, text):
        """Check if input is a single word (no spaces)"""
        return len(text.strip().split()) == 1
    
    def lookup_word_variants(self, word, verbose=True):
        """Lookup all variants of a single word from dictionary"""
        if not self.word_dictionary:
            if verbose:
                print("Word dictionary không khả dụng")
            return []
        
        # Normalize input
        normalized_word = self._remove_accents(word.strip().lower())
        
        variants = self.word_dictionary.get(normalized_word, [])
        
        if verbose:
            if variants:
                print(f"Từ '{word}' có {len(variants)} variants:")
                for i, variant in enumerate(variants, 1):
                    print(f"  {i}. {variant}")
            else:
                print(f"Không tìm thấy variants cho từ '{word}' trong dictionary")
        
        return variants
    
    def predict_single(self, text, verbose=True):
        """
        Dự đoán cho một câu hoặc từ đơn
        """
        start_time = time.time()
        
        try:
            # Kiểm tra nếu là từ đơn và có word dictionary
            if self._is_single_word(text) and self.word_dictionary:
                variants = self.lookup_word_variants(text, verbose=False)
                
                if variants:
                    inference_time = time.time() - start_time
                    
                    if verbose:
                        print(f"Input: {text} (từ đơn)")
                        print(f"Dictionary variants ({len(variants)}):")
                        for i, variant in enumerate(variants, 1):
                            print(f"  {i}. {variant}")
                        print(f"Thời gian: {inference_time*1000:.2f}ms")
                        print("-" * 50)
                    
                    # Trả về variant đầu tiên làm kết quả chính
                    return variants[0], inference_time
            
            # Sử dụng model cho câu hoặc từ không có trong dictionary
            result = self.restorer.predict(text)
            inference_time = time.time() - start_time
            
            if verbose:
                print(f"Input:  {text}")
                print(f"Output: {result}")
                print(f"Thời gian: {inference_time*1000:.2f}ms")
                print("-" * 50)
            
            return result, inference_time
            
        except Exception as e:
            print(f"Lỗi khi xử lý: {text}")
            print(f"Chi tiết lỗi: {e}")
            return text, 0
    
    def predict_multiple_single(self, text, num_suggestions=3, verbose=True):
        """
        Dự đoán nhiều gợi ý cho một câu hoặc từ đơn
        """
        start_time = time.time()
        
        try:
            # Kiểm tra nếu là từ đơn và có word dictionary
            if self._is_single_word(text) and self.word_dictionary:
                variants = self.lookup_word_variants(text, verbose=False)
                
                if variants:
                    inference_time = time.time() - start_time
                    
                    if verbose:
                        print(f"Input: {text} (từ đơn)")
                        print(f"Tìm thấy {len(variants)} variants từ dictionary:")
                        for i, variant in enumerate(variants, 1):
                            print(f"Gợi ý {i}: {variant}")
                        print(f"Thời gian: {inference_time*1000:.2f}ms")
                        print("-" * 50)
                    
                    return variants, inference_time
            
            # Sử dụng model cho câu hoặc từ không có trong dictionary
            results = self.restorer.predict_multiple(text, num_suggestions)
            inference_time = time.time() - start_time
            
            if verbose:
                print(f"Input:  {text}")
                for i, result in enumerate(results, 1):
                    print(f"Gợi ý {i}: {result}")
                print(f"Thời gian: {inference_time*1000:.2f}ms")
                print("-" * 50)
            
            return results, inference_time
            
        except Exception as e:
            print(f"Lỗi khi xử lý: {text}")
            print(f"Chi tiết lỗi: {e}")
            return [text], 0
    
    def predict_with_ranking(self, text, num_suggestions=5, verbose=True):
        """
        Dự đoán với context-aware ranking
        """
        start_time = time.time()
        
        try:
            # Generate multiple predictions
            base_results = self.restorer.predict_multiple(text, num_suggestions)
            
            # Rank predictions nếu có ranker
            if self.ranker and len(base_results) > 1:
                ranked_results = self.ranker.rank_predictions(base_results, text)
                inference_time = time.time() - start_time
                
                if verbose:
                    print(f"Input:  {text}")
                    print("Kết quả với ranking (từ tốt nhất đến kém nhất):")
                    for i, (pred, score, details) in enumerate(ranked_results, 1):
                        print(f"  {i}. {pred}")
                        print(f"     Score: {score:.4f} (Context: {details['context']:.3f}, "
                              f"Fluency: {details['fluency']:.3f}, Similarity: {details['similarity']:.3f})")
                    print(f"Thời gian: {inference_time*1000:.2f}ms")
                    print("-" * 80)
                
                return ranked_results, inference_time
            else:
                # Fallback về predict_multiple thông thường
                inference_time = time.time() - start_time
                simple_results = [(pred, 0.0, {'context': 0.0, 'fluency': 0.0, 'similarity': 0.0}) 
                                for pred in base_results]
                
                if verbose:
                    print(f"Input:  {text}")
                    print("Kết quả (không có ranking):")
                    for i, (pred, _, _) in enumerate(simple_results, 1):
                        print(f"  {i}. {pred}")
                    print(f"Thời gian: {inference_time*1000:.2f}ms")
                    print("-" * 80)
                
                return simple_results, inference_time
            
        except Exception as e:
            print(f"Lỗi khi xử lý: {text}")
            print(f"Chi tiết lỗi: {e}")
            return [(text, 0.0, {'context': 0.0, 'fluency': 0.0, 'similarity': 0.0})], 0
    
    def run_batch_test(self):
        """
        Chạy test trên nhiều câu
        """
        print("DEMO PHỤC HỒI DẤU TIẾNG VIỆT")
        print("=" * 80)
        print(f"Mô hình A-TCN - Vocabulary size: {self.restorer.vocab_size}")
        print(f"Số tham số: {sum(p.numel() for p in self.restorer.model.parameters()):,}")
        print("=" * 80)
        
        total_time = 0
        total_chars = 0
        correct_predictions = 0
        
        for i, text in enumerate(self.test_sentences):
            print(f"\nTest {i+1}:")
            result, inference_time = self.predict_single(text)
            
            total_time += inference_time
            total_chars += len(text)
            
            # So sánh với kết quả mong đợi (nếu có)
            if i < len(self.expected_results):
                expected = self.expected_results[i]
                if result == expected:
                    correct_predictions += 1
                    print(f"Kết quả: ĐÚNG")
                else:
                    print(f"Kết quả: SAI")
                    print(f"Mong đợi: {expected}")
        
        # Thống kê
        print("\n" + "=" * 80)
        print("THỐNG KÊ HIỆU SUẤT")
        print("=" * 80)
        print(f"Tổng thời gian: {total_time*1000:.2f}ms")
        print(f"Thời gian trung bình: {total_time*1000/len(self.test_sentences):.2f}ms/câu")
        print(f"Tốc độ xử lý: {total_chars/total_time:.0f} ký tự/giây")
        
        if self.expected_results:
            accuracy = correct_predictions / min(len(self.test_sentences), len(self.expected_results))
            print(f"Độ chính xác: {accuracy*100:.1f}% ({correct_predictions}/{min(len(self.test_sentences), len(self.expected_results))})")
    
    def run_ranking_batch_test(self):
        """
        Chạy test với context-aware ranking
        """
        print("DEMO PHỤC HỒI DẤU TIẾNG VIỆT - CONTEXT-AWARE RANKING")
        print("=" * 80)
        print(f"Mô hình A-TCN - Vocabulary size: {self.restorer.vocab_size}")
        print(f"Số tham số: {sum(p.numel() for p in self.restorer.model.parameters()):,}")
        
        if self.ranker:
            print("Context ranker: ĐƯỢC KÍCH HOẠT")
        else:
            print("Context ranker: KHÔNG KHẢ DỤNG")
        
        print("=" * 80)
        
        total_time = 0
        total_chars = 0
        best_predictions = 0  # Số lần prediction tốt nhất trùng với expected
        top3_predictions = 0  # Số lần expected nằm trong top 3
        
        for i, text in enumerate(self.test_sentences):
            print(f"\nTest {i+1}: {text}")
            
            results, inference_time = self.predict_with_ranking(
                text, num_suggestions=5, verbose=False
            )
            
            total_time += inference_time
            total_chars += len(text)
            
            # Hiển thị top 3 kết quả
            print("Top 3 gợi ý:")
            for j, (pred, score, details) in enumerate(results[:3], 1):
                if score > 0:
                    print(f"  {j}. {pred} (Score: {score:.3f})")
                else:
                    print(f"  {j}. {pred}")
            
            # So sánh với kết quả mong đợi
            if i < len(self.expected_results):
                expected = self.expected_results[i]
                
                # Kiểm tra prediction tốt nhất
                best_pred = results[0][0] if results else text
                if best_pred == expected:
                    best_predictions += 1
                    print(f"  Kết quả tốt nhất: ĐÚNG")
                else:
                    print(f"  Kết quả tốt nhất: SAI (mong đợi: {expected})")
                
                # Kiểm tra top 3
                top3_preds = [pred for pred, _, _ in results[:3]]
                if expected in top3_preds:
                    top3_predictions += 1
                    pos = top3_preds.index(expected) + 1
                    print(f"  Expected trong top 3: ĐÚNG (vị trí {pos})")
                else:
                    print(f"  Expected trong top 3: SAI")
        
        # Thống kê
        print("\n" + "=" * 80)
        print("THỐNG KÊ HIỆU SUẤT RANKING")
        print("=" * 80)
        print(f"Tổng thời gian: {total_time*1000:.2f}ms")
        print(f"Thời gian trung bình: {total_time*1000/len(self.test_sentences):.2f}ms/câu")
        print(f"Tốc độ xử lý: {total_chars/total_time:.0f} ký tự/giây")
        
        if self.expected_results:
            num_tests = min(len(self.test_sentences), len(self.expected_results))
            best_accuracy = best_predictions / num_tests
            top3_accuracy = top3_predictions / num_tests
            
            print(f"Độ chính xác prediction tốt nhất: {best_accuracy*100:.1f}% ({best_predictions}/{num_tests})")
            print(f"Tỷ lệ expected trong top 3: {top3_accuracy*100:.1f}% ({top3_predictions}/{num_tests})")
            
            if self.ranker:
                improvement = top3_accuracy - best_accuracy
                print(f"Cải thiện nhờ ranking: {improvement*100:.1f}%")
    

    def interactive_mode(self):
        """
        Chế độ tương tác
        """
        print("\n" + "=" * 80)
        print("CHẾ ĐỘ TƯƠNG TÁC")
        print("=" * 80)
        print("Nhập văn bản không dấu để phục hồi dấu.")
        
        if self.word_dictionary:
            print("Hỗ trợ dictionary lookup cho từ đơn.")
        
        print("Gõ 'quit' hoặc 'exit' để thoát.")
        print("-" * 80)
        
        while True:
            try:
                text = input("\nNhập văn bản: ").strip()
                
                if text.lower() in ['quit', 'exit', 'q']:
                    print("Tạm biệt!")
                    break
                
                if not text:
                    continue
                
                result, inference_time = self.predict_single(text, verbose=False)
                print(f"Kết quả: {result}")
                print(f"Thời gian: {inference_time*1000:.2f}ms")
                
            except KeyboardInterrupt:
                print("\nTạm biệt!")
                break
            except Exception as e:
                print(f"Lỗi: {e}")
    
    def interactive_multi_suggest_mode(self):
        """
        Chế độ tương tác với nhiều gợi ý
        """
        print("\n" + "=" * 80)
        print("CHẾ ĐỘ TƯƠNG TÁC - NHIỀU GỢI Ý")
        print("=" * 80)
        print("Nhập văn bản không dấu để phục hồi dấu với nhiều gợi ý.")
        
        if self.word_dictionary:
            print("Tự động detect từ đơn và hiển thị tất cả variants từ dictionary.")
        
        print("Gõ 'quit' hoặc 'exit' để thoát.")
        print("-" * 80)
        
        while True:
            try:
                text = input("\nNhập văn bản: ").strip()
                
                if text.lower() in ['quit', 'exit', 'q']:
                    print("Tạm biệt!")
                    break
                
                if not text:
                    continue
                
                # Tự động điều chỉnh num_suggestions cho từ đơn
                if self._is_single_word(text) and self.word_dictionary:
                    # Cho từ đơn, hiển thị tất cả variants
                    results, inference_time = self.predict_multiple_single(text, num_suggestions=999, verbose=False)
                else:
                    # Cho câu, hiển thị 3 gợi ý
                    results, inference_time = self.predict_multiple_single(text, num_suggestions=3, verbose=False)
                if len(results) == 1:
                    print(f"Kết quả: {results[0]}")
                else:
                    print("Các gợi ý:")
                    for i, result in enumerate(results, 1):
                        print(f"  {i}. {result}")
                print(f"Thời gian: {inference_time*1000:.2f}ms")
                
            except KeyboardInterrupt:
                print("\nTạm biệt!")
                break
            except Exception as e:
                print(f"Lỗi: {e}")
    
    def interactive_ranking_mode(self):
        """
        Chế độ tương tác với context-aware ranking
        """
        print("\n" + "=" * 80)
        print("CHẾ ĐỘ TƯƠNG TÁC - CONTEXT-AWARE RANKING")
        print("=" * 80)
        
        if self.ranker:
            print("Sử dụng context-aware ranking để sắp xếp kết quả theo mức độ phù hợp.")
        else:
            print("Context ranker không khả dụng - sử dụng multiple suggestions cơ bản.")
        
        print("Nhập văn bản không dấu để phục hồi dấu với ranking thông minh.")
        print("Gõ 'quit' hoặc 'exit' để thoát.")
        print("Gõ 'select <số>' để chọn một gợi ý cụ thể từ kết quả trước.")
        print("-" * 80)
        
        last_results = []
        
        while True:
            try:
                user_input = input("\nNhập văn bản: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Tạm biệt!")
                    break
                
                if not user_input:
                    continue
                
                # Kiểm tra lệnh select
                if user_input.lower().startswith('select '):
                    try:
                        selection_num = int(user_input.split()[1])
                        if 1 <= selection_num <= len(last_results):
                            selected = last_results[selection_num - 1]
                            print(f"\nBạn đã chọn: {selected[0]}")
                            print(f"Score: {selected[1]:.4f}")
                            print(f"Details: Context={selected[2]['context']:.3f}, "
                                  f"Fluency={selected[2]['fluency']:.3f}, "
                                  f"Similarity={selected[2]['similarity']:.3f}")
                        else:
                            print(f"Số thứ tự không hợp lệ. Vui lòng chọn từ 1 đến {len(last_results)}")
                    except (ValueError, IndexError):
                        print("Lệnh select không hợp lệ. Sử dụng: select <số>")
                    continue
                
                # Thực hiện prediction với ranking
                results, inference_time = self.predict_with_ranking(
                    user_input, num_suggestions=5, verbose=False
                )
                
                last_results = results
                
                if len(results) == 1:
                    pred, score, details = results[0]
                    print(f"\nKết quả: {pred}")
                    if score > 0:
                        print(f"Score: {score:.4f}")
                else:
                    print(f"\nCác gợi ý được sắp xếp theo mức độ phù hợp:")
                    for i, (pred, score, details) in enumerate(results, 1):
                        if score > 0:
                            print(f"  {i}. {pred} (Score: {score:.3f})")
                        else:
                            print(f"  {i}. {pred}")
                    
                    print(f"\nGõ 'select <số>' để chọn một gợi ý, ví dụ: select 2")
                
                print(f"Thời gian: {inference_time*1000:.2f}ms")
                
            except KeyboardInterrupt:
                print("\nTạm biệt!")
                break
            except Exception as e:
                print(f"Lỗi: {e}")
    
    def run_single_word_test(self):
        """
        Test word dictionary với từ đơn
        """
        print("\n" + "=" * 80)
        print("TEST WORD DICTIONARY - TỪ ĐƠN")
        print("=" * 80)
        
        if not self.word_dictionary:
            print("Word dictionary không khả dụng!")
            return
        
        print(f"Dictionary có {len(self.word_dictionary)} entries")
        print("-" * 80)
        
        total_time = 0
        found_count = 0
        
        for i, word in enumerate(self.test_single_words):
            print(f"\nTest {i+1}: {word}")
            start_time = time.time()
            
            variants = self.lookup_word_variants(word, verbose=False)
            
            inference_time = time.time() - start_time
            total_time += inference_time
            
            if variants:
                found_count += 1
                print(f"  Tìm thấy {len(variants)} variants:")
                for j, variant in enumerate(variants[:5], 1):  # Chỉ hiển thị 5 variants đầu
                    print(f"    {j}. {variant}")
                if len(variants) > 5:
                    print(f"    ... và {len(variants) - 5} variants khác")
            else:
                # Fallback sử dụng model
                result = self.restorer.predict(word)
                print(f"  Dictionary không có, model predict: {result}")
            
            print(f"  Thời gian: {inference_time*1000:.2f}ms")
        
        # Thống kê
        print("\n" + "=" * 80)
        print("THỐNG KÊ WORD DICTIONARY TEST")
        print("=" * 80)
        print(f"Tổng từ test: {len(self.test_single_words)}")
        print(f"Tìm thấy trong dictionary: {found_count}")
        print(f"Tỷ lệ tìm thấy: {found_count/len(self.test_single_words)*100:.1f}%")
        print(f"Thời gian trung bình: {total_time*1000/len(self.test_single_words):.2f}ms/từ")

    def benchmark_speed(self, num_iterations=100):
        """
        Benchmark tốc độ xử lý
        """
        print("\n" + "=" * 80)
        print("BENCHMARK TỐC ĐỘ")
        print("=" * 80)
        
        test_text = "toi di hoc va lam viec tai ha noi"
        print(f"Text test: {test_text}")
        print(f"Số lần lặp: {num_iterations}")
        print("-" * 80)
        
        # Warm up
        for _ in range(5):
            self.restorer.predict(test_text)
        
        # Benchmark
        start_time = time.time()
        for i in range(num_iterations):
            if i % 10 == 0:
                print(f"Progress: {i}/{num_iterations}", end="\r")
            self.restorer.predict(test_text)
        
        total_time = time.time() - start_time
        avg_time = total_time / num_iterations
        chars_per_sec = len(test_text) * num_iterations / total_time
        
        print(f"\nKết quả benchmark:")
        print(f"- Tổng thời gian: {total_time:.2f}s")
        print(f"- Thời gian trung bình: {avg_time*1000:.2f}ms/câu")
        print(f"- Tốc độ xử lý: {chars_per_sec:.0f} ký tự/giây")
        print(f"- Throughput: {num_iterations/total_time:.1f} câu/giây")

def main():
    """
    Hàm chính
    """
    # Kiểm tra model đã được huấn luyện
    model_path = "models/best_model.pth"
    context_ranker_path = "models/context_ranker.json"
    word_dict_path = "models/word_dictionary.json"
    
    if not os.path.exists(model_path):
        print(f"Cảnh báo: Không tìm thấy model tại {model_path}")
        print("Sử dụng mô hình chưa huấn luyện để demo kiến trúc.")
        model_path = None
    
    # Kiểm tra context ranker
    if not os.path.exists(context_ranker_path):
        print(f"Cảnh báo: Không tìm thấy context ranker tại {context_ranker_path}")
        print("Các tính năng ranking sẽ bị giới hạn.")
        context_ranker_path = None
    
    # Kiểm tra word dictionary
    if not os.path.exists(word_dict_path):
        print(f"Cảnh báo: Không tìm thấy word dictionary tại {word_dict_path}")
        print("Các tính năng từ đơn sẽ bị giới hạn.")
        word_dict_path = None
    
    # Khởi tạo demo
    demo = VietnameseAccentDemo(model_path, context_ranker_path, word_dict_path)
    
    while True:
        print("\n" + "=" * 80)
        print("DEMO PHỤC HỒI DẤU TIẾNG VIỆT")
        print("=" * 80)
        
        if demo.ranker:
            print("Context-Aware Ranking: ĐƯỢC KÍCH HOẠT")
        else:
            print("Context-Aware Ranking: KHÔNG KHẢ DỤNG")
        
        if demo.word_dictionary:
            print(f"Word Dictionary: ĐƯỢC KÍCH HOẠT ({len(demo.word_dictionary)} entries)")
        else:
            print("Word Dictionary: KHÔNG KHẢ DỤNG")
        
        print("-" * 80)
        print("Chọn chức năng:")
        print("1. Chạy test batch cơ bản")
        print("2. Chạy test batch với ranking")
        print("3. Test word dictionary - từ đơn")
        print("4. Chế độ tương tác đơn giản")
        print("5. Chế độ tương tác - Nhiều gợi ý (tự detect từ đơn)")
        print("6. Chế độ tương tác - Context-aware ranking")
        print("7. Benchmark tốc độ")
        print("8. Thoát")
        print("-" * 80)
        
        try:
            choice = input("Nhập lựa chọn (1-8): ").strip()
            
            if choice == '1':
                demo.run_batch_test()
            elif choice == '2':
                demo.run_ranking_batch_test()
            elif choice == '3':
                demo.run_single_word_test()
            elif choice == '4':
                demo.interactive_mode()
            elif choice == '5':
                demo.interactive_multi_suggest_mode()
            elif choice == '6':
                demo.interactive_ranking_mode()
            elif choice == '7':
                demo.benchmark_speed()
            elif choice == '8':
                print("Tạm biệt!")
                break
            else:
                print("Lựa chọn không hợp lệ. Vui lòng chọn 1-8.")
                
        except KeyboardInterrupt:
            print("\nTạm biệt!")
            break
        except Exception as e:
            print(f"Lỗi: {e}")

if __name__ == "__main__":
    main() 