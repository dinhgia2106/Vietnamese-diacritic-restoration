import torch
from torch.utils.data import DataLoader, random_split
from transformers import RobertaForTokenClassification, RobertaTokenizerFast, AdamW, get_scheduler
from tqdm.auto import tqdm
import os
import pickle

# Import các lớp đã tạo từ file phoBERT.py
from phoBERT import SyllableVocabulary, AccentSyllableDataset

def train():
    # --- 1. Cấu hình ---
    MODEL_NAME = "vinai/phobert-base"
    DATA_PATH = "data\cleaned_comments.txt" 
    MODEL_SAVE_PATH = "./vietnamese_accent_restorer"
    
    NUM_EPOCHS = 10
    BATCH_SIZE = 32 # Giảm xuống 8 hoặc 4 nếu bị lỗi hết bộ nhớ (CUDA out of memory)
    LEARNING_RATE = 2e-5

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 2. Chuẩn bị dữ liệu ---
    print("Loading data...")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        # Giới hạn số lượng câu để chạy thử nhanh hơn, xóa dòng [:10000] khi huấn luyện thật
        data_corpus = [line.strip() for line in f if line.strip()][:10000] 

    print("Building vocabulary...")
    syllable_vocab = SyllableVocabulary()
    syllable_vocab.build_from_data(data_corpus)

    print("Creating tokenizer...")
    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)

    print("Creating dataset...")
    full_dataset = AccentSyllableDataset(data_corpus, tokenizer, syllable_vocab)

    # Chia dataset thành train và validation
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # --- 3. Khởi tạo Mô hình và Optimizer ---
    print("Initializing model...")
    model = RobertaForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=syllable_vocab.get_vocab_size(),
        ignore_mismatched_sizes=True
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    num_training_steps = NUM_EPOCHS * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # --- 4. Vòng lặp Huấn luyện ---
    progress_bar = tqdm(range(num_training_steps))
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}, Validation Loss: {avg_val_loss:.4f}")

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"\nNew best model found! Val Loss: {best_val_loss:.4f}. Saving...")
            
            if not os.path.exists(MODEL_SAVE_PATH):
                os.makedirs(MODEL_SAVE_PATH)
                
            # Lưu mô hình và tokenizer
            model.save_pretrained(MODEL_SAVE_PATH)
            tokenizer.save_pretrained(MODEL_SAVE_PATH)
            
            # LƯU TỪ ĐIỂN ÂM TIẾT - BƯỚC QUAN TRỌNG
            vocab_save_path = os.path.join(MODEL_SAVE_PATH, "syllable_vocab.pkl")
            with open(vocab_save_path, 'wb') as f:
                pickle.dump(syllable_vocab, f)
            print(f"Model, tokenizer, and vocabulary saved to {MODEL_SAVE_PATH}")

    print("Training complete!")

if __name__ == "__main__":
    train()