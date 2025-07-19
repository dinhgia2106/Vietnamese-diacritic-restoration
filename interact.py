import torch
import pickle
import os

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

            # Thực hiện dự đoán
            restored_sentence = restorer.predict(user_input)
            
            # In kết quả
            print(f"Kết quả: {restored_sentence}")

        except KeyboardInterrupt:
            print("\nĐã nhận tín hiệu thoát. Tạm biệt!")
            break
        except Exception as e:
            print(f"Đã xảy ra lỗi: {e}")


if __name__ == "__main__":
    interactive_session()