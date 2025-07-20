# Vietnamese Accent Restoration - Advanced Training System

## Tổng quan

Hệ thống này bao gồm 3 thành phần chính:

1. **Multiple Suggestions**: Đưa ra nhiều gợi ý phục hồi dấu
2. **Streaming Training**: Training với dataset lớn bằng cách chia thành chunks
3. **Fine-tuning**: Fine-tune model trên domain-specific data

## 1. Multiple Suggestions

### Sử dụng trong Interactive Mode

```bash
python interact.py
```

Sau khi chạy, bạn có thể:

- `toi di hoc` → Trả về 1 kết quả (default)
- `multi toi di hoc` → Trả về 3 gợi ý: "tôi đi học", "tối đi học", v.v.
- `single toi di hoc` → Trả về 1 kết quả

### API Usage

```python
from phoBERT import PhobertAccentRestorer

# Load model
restorer = PhobertAccentRestorer(model_path="./vietnamese_accent_restorer", syllable_vocab=vocab)

# Single prediction
result = restorer.predict("toi di hoc")
print(result)  # "tôi đi học"

# Multiple suggestions
suggestions = restorer.predict_beam_search("toi di hoc", beam_width=5, num_suggestions=3)
for i, suggestion in enumerate(suggestions, 1):
    print(f"{i}. {suggestion}")
# 1. tôi đi học
# 2. tối đi học
# 3. tói đi học
```

## 2. Streaming Training

### Step 1: Chia dataset lớn thành chunks

```bash
python train_streaming.py --split-data --input-file data/large_dataset.txt --chunk-size 1000000
```

Điều này sẽ:

- Chia file lớn thành các chunks 1M samples
- Lưu vào thư mục `corpus_splitted/`
- Mỗi chunk là 1 file JSON

### Step 2: Training với streaming

```bash
python train_streaming.py --train
```

Hoặc đơn giản:

```bash
python train_streaming.py
```

### Tính năng của Streaming Training:

- **Checkpoint System**: Tự động save checkpoint sau mỗi chunk
- **Resume Training**: Có thể tiếp tục training từ chunk đã dừng
- **Memory Efficient**: Chỉ load 1 chunk vào memory tại 1 thời điểm
- **Progress Tracking**: Log chi tiết thời gian training cho mỗi chunk

### Configuration trong `train_streaming.py`:

```python
CHUNK_SIZE = 1000000        # 1M samples per chunk
EPOCHS_PER_CHUNK = 10       # 10 epochs cho mỗi chunk
BATCH_SIZE = 32             # Batch size
LEARNING_RATE = 2e-5        # Learning rate
```

## 3. Fine-tuning

### Chuẩn bị data cho fine-tuning

Data có thể là:

- File text thường (mỗi dòng 1 câu)
- File JSON (array of strings)

Ví dụ file `domain_data.txt`:

```
Tôi muốn đặt bàn cho 4 người.
Xin chào, tôi cần hỗ trợ về sản phẩm.
Cảm ơn bạn đã sử dụng dịch vụ.
```

### Chạy fine-tuning

#### Basic fine-tuning:

```bash
python fine_tune.py \
    --pretrained-model ./vietnamese_accent_restorer \
    --data-file ./domain_data.txt \
    --save-path ./fine_tuned_model \
    --epochs 5 \
    --batch-size 16 \
    --learning-rate 1e-5
```

#### Advanced fine-tuning với layer freezing:

```bash
python fine_tune.py \
    --pretrained-model ./vietnamese_accent_restorer \
    --data-file ./domain_data.txt \
    --save-path ./fine_tuned_model \
    --epochs 5 \
    --batch-size 16 \
    --learning-rate 1e-5 \
    --freeze-embeddings \
    --freeze-encoder-layers 6
```

### Fine-tuning Options:

- `--freeze-embeddings`: Freeze embedding layers
- `--freeze-encoder-layers N`: Freeze N encoder layers đầu
- `--val-split 0.1`: 10% data cho validation
- `--learning-rate 1e-5`: Learning rate thấp hơn training

### Output của Fine-tuning:

```
fine_tuned_model/
├── best_model/           # Model tốt nhất (theo val loss)
├── final_model/          # Model cuối cùng
└── fine_tuning_history.json  # Lịch sử training
```

## 4. Workflow hoàn chỉnh

### Scenario 1: Training từ đầu với large dataset

```bash
# 1. Chia data
python train_streaming.py --split-data --input-file data/huge_dataset.txt

# 2. Training
python train_streaming.py --train

# 3. Test model
python interact.py
```

### Scenario 2: Fine-tune model có sẵn

```bash
# 1. Fine-tune trên domain data
python fine_tune.py \
    --pretrained-model ./vietnamese_accent_restorer \
    --data-file ./restaurant_data.txt \
    --save-path ./restaurant_model

# 2. Test fine-tuned model
# (Cần update interact.py để point đến restaurant_model)
```

### Scenario 3: Continuous learning

```bash
# 1. Train với chunk 1-50
python train_streaming.py

# 2. Có thêm data mới, fine-tune
python fine_tune.py \
    --pretrained-model ./vietnamese_accent_restorer_streaming \
    --data-file ./new_data.txt \
    --save-path ./updated_model

# 3. Tiếp tục training với chunks mới
# (Copy model từ updated_model về streaming path)
```

## 5. Monitoring và Debugging

### Check training progress:

```bash
# Xem log files
ls -la checkpoints/
cat checkpoints/latest_checkpoint.pt

# Xem model size
du -sh vietnamese_accent_restorer*/
```

### Memory optimization:

- Giảm `BATCH_SIZE` trong config
- Giảm `CHUNK_SIZE`
- Sử dụng `torch.cuda.empty_cache()` trong code

### Error handling:

- Checkpoint system tự động resume training
- Validation loss để detect overfitting
- Log chi tiết để debug

## 6. Best Practices

### Training:

1. Bắt đầu với chunk size nhỏ để test
2. Monitor GPU memory usage
3. Save intermediate checkpoints
4. Use validation set để tránh overfitting

### Fine-tuning:

1. Sử dụng learning rate thấp (1e-5 hoặc 5e-6)
2. Freeze một số layers để tránh catastrophic forgetting
3. Sử dụng domain-specific validation set
4. Monitor validation loss để early stopping

### Production:

1. Test model với diverse test cases
2. Benchmark inference time
3. Monitor model performance over time
4. Keep original model as fallback
