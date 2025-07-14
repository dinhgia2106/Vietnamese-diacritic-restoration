# Vietnamese Accent Restoration - Progressive Training

Hệ thống training từ đầu cho Vietnamese Accent Restoration model với khả năng checkpoint và resume.

## Cấu trúc Files

- `progressive_trainer.py` - Main training script
- `training_utils.py` - Utilities để monitor và test
- `model_architecture.py` - Kiến trúc model
- `context_aware_tuning.py` - Fine-tuning system (legacy)

## Tính năng chính

### Progressive Training
- Train từ đầu trên toàn bộ corpus_splitted
- Mỗi file JSON được treat như một training sample
- Chia train/val/test cho từng sample
- Lưu checkpoint sau mỗi sample

### Resume Capability
- Tự động lưu trạng thái training
- Có thể resume từ bất kỳ điểm nào
- Lưu best model dựa trên validation loss

### Monitoring System
- Log chi tiết cho mỗi bước training
- Test model trong quá trình training
- Vẽ training curves
- Backup và cleanup utilities

## Cách sử dụng

### 1. Bắt đầu training từ đầu

```bash
python progressive_trainer.py
```

### 2. Monitor quá trình training

Kiểm tra trạng thái:
```bash
python training_utils.py status
```

Test model hiện tại:
```bash
python training_utils.py test
```

Vẽ training curves:
```bash
python training_utils.py plot
```

### 3. Resume training

Nếu training bị dừng, resume từ checkpoint:
```bash
python training_utils.py resume
```

### 4. Utilities khác

Backup model:
```bash
python training_utils.py backup
```

Dọn dẹp checkpoints cũ:
```bash
python training_utils.py clean
```

## Cấu trúc thư mục output

```
models/progressive/
├── best_model.pth              # Model tốt nhất
├── final_model.pth             # Model cuối cùng
├── vocabulary.json             # Character vocabulary
├── training_state.json         # Trạng thái training
├── training_curves.png         # Đồ thị training
├── checkpoints/
│   └── latest_checkpoint.pth   # Checkpoint mới nhất
└── logs/
    └── training_*.log          # Log files
```

## Training State

File `training_state.json` chứa:
- `current_sample_idx`: Sample hiện tại đang train
- `total_samples`: Tổng số samples
- `global_step`: Số bước training tổng cộng
- `best_val_loss`: Validation loss tốt nhất
- `train_losses`: Lịch sử train loss
- `val_losses`: Lịch sử validation loss
- `start_time`: Thời gian bắt đầu training

## Model Configuration

Mặc định sử dụng Enhanced model với:
- `embedding_dim`: 256
- `hidden_dim`: 512
- `num_layers`: 8
- `dropout`: 0.15

Có thể thay đổi trong `progressive_trainer.py`:
```python
trainer = ProgressiveTrainer(
    model_type="enhanced",  # hoặc "standard"
    embedding_dim=256,
    hidden_dim=512,
    num_layers=8,
    dropout=0.15
)
```

## Xử lý lỗi

### Nếu training bị crash:
1. Kiểm tra log files trong `models/progressive/logs/`
2. Chạy `python training_utils.py status` để xem trạng thái
3. Resume training với `python training_utils.py resume`

### Nếu hết memory:
1. Giảm `batch_size` trong `train_on_sample()` function
2. Hoặc sử dụng model "standard" thay vì "enhanced"

### Nếu muốn train lại từ đầu:
1. Xóa thư mục `models/progressive/`
2. Chạy `python progressive_trainer.py`

## Performance Monitoring

Model sẽ được test với các câu mẫu:
- "toi di hoc"
- "hom nay troi dep"
- "ban co khoe khong"
- ...

Xem kết quả với `python training_utils.py test`

## Disk Space Management

- Mỗi checkpoint ~100-500MB tùy model size
- Log files tăng dần theo thời gian
- Sử dụng `clean` command để dọn dẹp định kỳ
- Backup model quan trọng với `backup` command 