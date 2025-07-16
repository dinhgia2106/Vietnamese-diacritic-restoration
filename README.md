# Vietnamese Accent Restoration

A deep learning system for restoring Vietnamese diacritics using Acausal Temporal Convolutional Networks (A-TCN).

## Project Structure

```
VietnameseAnccentRestore/
├── model_architecture.py          # Core model definitions (TCN, Enhanced TCN, VietnameseAccentRestorer)
├── train_model.py                 # Basic training pipeline with simple trainer
├── context_aware_tuning.py        # Advanced fine-tuning with context-aware ranking
├── VietnameseAccentRestore.py     # Demo application with interactive modes
└── data/                          # Training data directory
    ├── Viet74K_clean.txt         # Main training corpus
    ├── cleaned_comments.txt      # Additional comments data
    └── corpus-full.txt           # Full corpus data
```

## Components

### 1. Model Architecture (`model_architecture.py`)

- **VietnameseAccentRestorer**: Main wrapper class for the model
- **ACausalTCN**: Basic Temporal Convolutional Network
- **EnhancedACausalTCN**: Advanced TCN with multi-head attention
- **Supporting layers**: PositionalEncoding, MultiHeadAttention, TCNBlock

### 2. Training Pipeline (`train_model.py`)

- **AccentRestorationTrainer**: Simple trainer for basic training
- **AccentDataset**: Dataset class for loading training pairs
- **load_viet74k_data()**: Data loading utility
- Features: Early stopping, learning rate scheduling, gradient clipping

### 3. Context-Aware Fine-tuning (`context_aware_tuning.py`)

- **ContextAwareFinetuner**: Advanced trainer with ranking loss
- **StreamingDataset**: Efficient dataset for large files
- **ContextAwareRanker**: Context-aware prediction ranking
- Features: Ranking loss, context dictionary, streaming data processing

### 4. Demo Application (`VietnameseAccentRestore.py`)

- **VietnameseAccentDemo**: Interactive demo with multiple modes
- Supports word dictionary lookup
- Context-aware ranking integration
- Performance benchmarking

## Usage

### Basic Training

```bash
python train_model.py
```

This will train a basic model using Viet74K data.

### Context-Aware Fine-tuning

```bash
python context_aware_tuning.py
```

This requires a pre-trained base model and additional corpus data.

### Demo Application

```bash
python VietnameseAccentRestore.py
```

Interactive demo with multiple modes:

1. Batch testing
2. Context-aware ranking
3. Word dictionary lookup
4. Interactive modes
5. Performance benchmarking

## Data Requirements

- **Viet74K_clean.txt**: Main training corpus (Vietnamese text with diacritics)
- **cleaned_comments.txt**: Additional training data (optional)
- **corpus-full.txt**: Full corpus for fine-tuning (optional)

## Model Features

- Character-level sequence-to-sequence prediction
- Temporal convolutions with dilated kernels
- Multi-head self-attention (Enhanced model)
- Context-aware prediction ranking
- Support for multiple prediction suggestions

## Training Features

- Automatic train/validation split
- Early stopping with patience
- Learning rate scheduling
- Gradient clipping
- Mixed training with ranking loss
- Streaming data processing for large datasets

## Requirements

- torch
- numpy
- tqdm
- sklearn
- unidecode
- unicodedata

## Model Outputs

- **models/best_model.pth**: Best trained model
- **models/context_aware_best_model.pth**: Fine-tuned model
- **models/context_ranker.json**: Context ranking data
- **models/word_dictionary.json**: Word dictionary (if available)
