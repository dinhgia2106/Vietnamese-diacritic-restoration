import torch
from torch.utils.data import DataLoader, random_split
from transformers import RobertaForTokenClassification, RobertaTokenizerFast, AdamW, get_scheduler
from tqdm.auto import tqdm
import os
import pickle

# Import các lớp đã tạo
from phoBERT import SyllableVocabulary # Vẫn cần để xây dựng vocab
from streaming_dataset import StreamingAccentSyllableDataset # SỬ DỤNG DATASET MỚI

def train():
    # --- 1. Cấu hình ---
    MODEL_NAME = "vinai/phobert-base"
    DATA_PATH = "data/cleaned_comments.txt" # Đảm bảo đường dẫn đúng
    MODEL_SAVE_PATH = "./vietnamese_accent_restorer"
    
    NUM_EPOCHS = 3 # Số lần lặp qua TOÀN BỘ file dữ liệu
    BATCH_SIZE = 16 
    LEARNING_RATE = 2e-5
    VOCAB_SAMPLE_SIZE = 50000 # Lấy 50k dòng đầu để xây dựng từ điển

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 2. Chuẩn bị dữ liệu ---
    print("Building vocabulary from a sample of the data...")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        vocab_corpus = [line.strip() for _, line in zip(range(VOCAB_SAMPLE_SIZE), f) if line.strip()]
    
    syllable_vocab = SyllableVocabulary()
    syllable_vocab.build_from_data(vocab_corpus)

    print("Creating tokenizer...")
    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)

    print("Initializing streaming dataset...")
    # Không cần chia train/val với streaming, ta sẽ train trên toàn bộ
    # và có thể dùng một file validation riêng nếu muốn
    train_dataset = StreamingAccentSyllableDataset(DATA_PATH, tokenizer, syllable_vocab)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    
    # --- 3. Khởi tạo Mô hình và Optimizer ---
    print("Initializing model...")
    model = RobertaForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=syllable_vocab.get_vocab_size(), ignore_mismatched_sizes=True
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Ước tính số bước huấn luyện để tạo scheduler
    # Đây là cách làm gần đúng vì ta không biết chính xác số dòng
    # Bạn có thể đếm số dòng trước nếu muốn chính xác hơn
    estimated_num_lines = sum(1 for line in open(DATA_PATH, 'r', encoding='utf-8'))
    num_training_steps = NUM_EPOCHS * (estimated_num_lines // BATCH_SIZE)

    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # --- 4. Vòng lặp Huấn luyện ---
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        model.train()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training")
        
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            
            # Cập nhật thanh tiến trình với loss hiện tại
            progress_bar.set_postfix(loss=loss.item())
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        
        # Lưu model sau mỗi epoch
        print(f"\nEnd of Epoch {epoch+1}. Saving model...")
        if not os.path.exists(MODEL_SAVE_PATH):
            os.makedirs(MODEL_SAVE_PATH)
        
        model.save_pretrained(MODEL_SAVE_PATH)
        tokenizer.save_pretrained(MODEL_SAVE_PATH)
        
        vocab_save_path = os.path.join(MODEL_SAVE_PATH, "syllable_vocab.pkl")
        with open(vocab_save_path, 'wb') as f:
            pickle.dump(syllable_vocab, f)
        print(f"Model for epoch {epoch+1} saved to {MODEL_SAVE_PATH}")

    print("Training complete!")

if __name__ == "__main__":
    train()