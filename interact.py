# file: interactive.py (đã cập nhật)
import torch
import pickle
import os
import time # Thêm thư viện time

from phoBERT import PhobertAccentRestorer 

def interactive_session():
    MODEL_PATH = "./vietnamese_accent_restorer" # Hoặc đường dẫn model đã fine-tune
    VOCAB_PATH = os.path.join(MODEL_PATH, "syllable_vocab.pkl")

    if not os.path.exists(MODEL_PATH) or not os.path.exists(VOCAB_PATH):
        print(f"Lỗi: Không tìm thấy mô hình hoặc từ điển tại '{MODEL_PATH}'.")
        return

    print("Đang tải từ điển âm tiết...")
    with open(VOCAB_PATH, 'rb') as f:
        syllable_vocab = pickle.load(f)
    
    print("Đang tải mô hình đã huấn luyện...")
    try:
        restorer = PhobertAccentRestorer(model_path=MODEL_PATH, syllable_vocab=syllable_vocab)
        print("Mô hình đã sẵn sàng!")
    except Exception as e:
        print(f"Đã xảy ra lỗi khi tải mô hình: {e}")
        return

    print("\n" + "="*50)
    print("      CHƯƠNG TRÌNH PHỤC HỒI DẤU TIẾNG VIỆT")
    print("="*50)
    print("Nhập một câu không dấu và nhấn Enter.")
    print("Gõ 'quit' hoặc 'exit' để thoát.")
    
    while True:
        try:
            user_input = input("\nNhập câu không dấu > ")

            if user_input.lower() in ['quit', 'exit']:
                print("Tạm biệt!")
                break
            
            if not user_input.strip():
                continue

            # Đo thời gian và thực hiện dự đoán với nhiều gợi ý
            start_time = time.time()
            suggestions = restorer.predict_beam_search(user_input, num_suggestions=3)
            end_time = time.time()
            
            duration_ms = (end_time - start_time) * 1000
            
            # In kết quả
            print("--- Các gợi ý khả thi: ---")
            if not suggestions:
                print("Không tìm thấy gợi ý nào.")
            else:
                for i, sentence in enumerate(suggestions):
                    print(f"{i+1}. {sentence}")
            
            print(f"(Thời gian xử lý: {duration_ms:.2f} ms)")

        except KeyboardInterrupt:
            print("\nĐã nhận tín hiệu thoát. Tạm biệt!")
            break
        except Exception as e:
            print(f"Đã xảy ra lỗi: {e}")

if __name__ == "__main__":
    interactive_session()