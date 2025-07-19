# file: finetune.py

import torch
from torch.utils.data import DataLoader
from transformers import RobertaForTokenClassification, RobertaTokenizerFast, AdamW, get_scheduler
from tqdm.auto import tqdm
import os
import pickle

from phoBERT import SyllableVocabulary
from streaming_dataset import StreamingAccentSyllableDataset

def fine_tune():
    # --- 1. Cấu hình ---
    # ĐƯỜNG DẪN TỚI MODEL ĐÃ HUẤN LUYỆN CỦA BẠN
    PRETRAINED_MODEL_PATH = "./vietnamese_accent_restorer"
    
    # ĐƯỜNG DẪN TỚI BỘ DỮ LIỆU MỚI ĐỂ FINE-TUNE
    NEW_DATA_PATH = "data/corpus-full.txt" # << THAY ĐỔI FILE NÀY
    
    FINETUNED_MODEL_SAVE_PATH = "./vietnamese_accent_restorer_finetuned"
    
    NUM_EPOCHS = 2 # Thường thì fine-tune cần ít epoch hơn
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-5 # Fine-tune thường dùng learning rate thấp hơn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # --- 2. Tải các thành phần đã có ---
    print(f"Loading components from {PRETRAINED_MODEL_PATH}")
    
    # Tải lại từ điển đã dùng để huấn luyện model gốc
    vocab_path = os.path.join(PRETRAINED_MODEL_PATH, "syllable_vocab.pkl")
    with open(vocab_path, 'rb') as f:
        syllable_vocab = pickle.load(f)

    # Tải tokenizer đã lưu
    tokenizer = RobertaTokenizerFast.from_pretrained(PRETRAINED_MODEL_PATH)
    
    # Tải mô hình đã huấn luyện của bạn
    model = RobertaForTokenClassification.from_pretrained(PRETRAINED_MODEL_PATH).to(device)

    # --- 3. Chuẩn bị dữ liệu mới ---
    print(f"Initializing streaming dataset for new data at {NEW_DATA_PATH}")
    finetune_dataset = StreamingAccentSyllableDataset(NEW_DATA_PATH, tokenizer, syllable_vocab)
    finetune_dataloader = DataLoader(finetune_dataset, batch_size=BATCH_SIZE)

    # --- 4. Khởi tạo Optimizer và bắt đầu Fine-tuning ---
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    estimated_num_lines = sum(1 for line in open(NEW_DATA_PATH, 'r', encoding='utf-8'))
    num_training_steps = NUM_EPOCHS * (estimated_num_lines // BATCH_SIZE)
    
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    
    # Vòng lặp fine-tuning
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Fine-tuning Epoch {epoch+1}/{NUM_EPOCHS} ---")
        model.train()
        progress_bar = tqdm(finetune_dataloader, desc=f"Epoch {epoch+1} Fine-tuning")
        
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            progress_bar.set_postfix(loss=loss.item())
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
        print(f"\nEnd of Fine-tuning Epoch {epoch+1}. Saving fine-tuned model...")
        if not os.path.exists(FINETUNED_MODEL_SAVE_PATH):
            os.makedirs(FINETUNED_MODEL_SAVE_PATH)
            
        model.save_pretrained(FINETUNED_MODEL_SAVE_PATH)
        tokenizer.save_pretrained(FINETUNED_MODEL_SAVE_PATH)
        
        vocab_save_path = os.path.join(FINETUNED_MODEL_SAVE_PATH, "syllable_vocab.pkl")
        with open(vocab_save_path, 'wb') as f:
            pickle.dump(syllable_vocab, f)
        print(f"Fine-tuned model saved to {FINETUNED_MODEL_SAVE_PATH}")

    print("Fine-tuning complete!")

if __name__ == "__main__":
    fine_tune()