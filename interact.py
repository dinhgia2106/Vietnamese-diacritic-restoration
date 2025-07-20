import torch
import pickle
import os
import time
from datetime import datetime

# Import lớp PhoverAccentRestorer từ file phoBERT.py của bạn
from phoBERT import PhobertAccentRestorer 

def interactive_session():
    """
    Tải mô hình đã huấn luyện và bắt đầu một phiên tương tác.
    """
    # --- Cấu hình ---
    # Đường dẫn này phải khớp với MODEL_SAVE_PATH trong file train.py
    MODEL_PATH = "./vietnamese_accent_restorer"
    VOCAB_PATH = os.path.join(MODEL_PATH, "syllable_vocab.pkl")

    # --- Kiểm tra xem mô hình đã tồn tại chưa ---
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VOCAB_PATH):
        print(f"Lỗi: Không tìm thấy mô hình hoặc từ điển tại '{MODEL_PATH}'.")
        print("Vui lòng chạy file train.py để huấn luyện và lưu mô hình trước.")
        return

    # --- Tải các thành phần cần thiết ---
    print("Đang tải từ điển âm tiết...")
    with open(VOCAB_PATH, 'rb') as f:
        syllable_vocab = pickle.load(f)
    
    print("Đang tải mô hình đã huấn luyện...")
    # Khởi tạo restorer với mô hình đã huấn luyện và từ điển
    try:
        restorer = PhobertAccentRestorer(
            model_path=MODEL_PATH,
            syllable_vocab=syllable_vocab
        )
        print("Mô hình đã sẵn sàng!")
    except Exception as e:
        print(f"Đã xảy ra lỗi khi tải mô hình: {e}")
        return

    # --- Bắt đầu vòng lặp tương tác ---
    print("\n" + "="*50)
    print("      CHƯƠNG TRÌNH PHỤC HỒI DẤU TIẾNG VIỆT")
    print("="*50)
    print("Nhập một câu không dấu và nhấn Enter.")
    print("Gõ 'quit' hoặc 'exit' để thoát.")
    
    while True:
        try:
            # Nhận đầu vào từ người dùng
            user_input = input("\nNhập câu không dấu > ")

            # Kiểm tra điều kiện thoát
            if user_input.lower() in ['quit', 'exit']:
                print("Tạm biệt!")
                break
            
            if not user_input.strip():
                continue

            # Log thời gian bắt đầu
            start_time = time.time()
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"[{current_time}] Bắt đầu xử lý...")

            # Thực hiện prediction với intelligent ranking (mô hình tự quyết định số lượng)
            suggestions = restorer.predict_with_adaptive_ranking(user_input.strip())
            
            print(f"Input: {user_input.strip()}")
            if len(suggestions) == 1:
                print(f"Kết quả: {suggestions[0][0]} (độ tin cậy: {suggestions[0][1]:.3f})")
            else:
                print(f"Các gợi ý (được xếp hạng thông minh):")
                for i, (suggestion, score) in enumerate(suggestions, 1):
                    print(f"  {i}. {suggestion} (điểm: {score:.3f})")
            
            # Log thời gian kết thúc và tính toán thời gian chạy
            end_time = time.time()
            execution_time = end_time - start_time
            end_current_time = datetime.now().strftime("%H:%M:%S")
            
            print(f"[{end_current_time}] Hoàn thành trong {execution_time:.3f} giây")

        except KeyboardInterrupt:
            print("\nĐã nhận tín hiệu thoát. Tạm biệt!")
            break
        except Exception as e:
            print(f"Đã xảy ra lỗi: {e}")


if __name__ == "__main__":
    interactive_session()