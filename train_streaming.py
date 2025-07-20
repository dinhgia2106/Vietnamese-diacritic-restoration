import torch
from transformers import RobertaForTokenClassification, RobertaTokenizerFast, RobertaConfig
import os
import pickle
import argparse
from datetime import datetime

# Import custom classes
from phoBERT import SyllableVocabulary
from streaming_dataset import StreamingDatasetManager, StreamingTrainer

def streaming_train():
    """
    Main function để train model với streaming dataset.
    """
    # --- Configuration ---
    MODEL_NAME = "vinai/phobert-base"
    DATA_DIR = "./corpus_splitted"  # Directory chứa các chunk files
    MODEL_SAVE_PATH = "./vietnamese_accent_restorer_streaming"
    CHECKPOINT_DIR = "./checkpoints"
    
    # Training parameters
    CHUNK_SIZE = 1000000  # 1M samples per chunk
    EPOCHS_PER_CHUNK = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-5
    
    # Ensure directories exist
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # --- Initialize Dataset Manager ---
    dataset_manager = StreamingDatasetManager(DATA_DIR, CHUNK_SIZE)
    
    # Check if we already have chunk files
    chunk_files = dataset_manager.get_chunk_files(DATA_DIR)
    if not chunk_files:
        print("No chunk files found. Please run data splitting first.")
        print("Example: python streaming_train.py --split-data --input-file data/cleaned_comments.txt")
        return
    
    print(f"Found {len(chunk_files)} chunk files")
    
    # --- Build or Load Vocabulary ---
    vocab_path = os.path.join(MODEL_SAVE_PATH, "syllable_vocab.pkl")
    
    if os.path.exists(vocab_path):
        print("Loading existing vocabulary...")
        with open(vocab_path, 'rb') as f:
            syllable_vocab = pickle.load(f)
    else:
        print("Building vocabulary from all chunks...")
        syllable_vocab = build_vocabulary_from_chunks(chunk_files)
        # Save vocabulary
        with open(vocab_path, 'wb') as f:
            pickle.dump(syllable_vocab, f)
        print(f"Vocabulary saved to {vocab_path}")
    
    print(f"Vocabulary size: {syllable_vocab.get_vocab_size()}")
    
    # --- Initialize Model and Tokenizer ---
    print("Initializing tokenizer and model...")
    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)
    
    # Check for existing model or create new one
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pt")
    
    if os.path.exists(checkpoint_path):
        print("Found existing checkpoint. Loading...")
        config = RobertaConfig.from_pretrained(MODEL_NAME, num_labels=syllable_vocab.get_vocab_size())
        model = RobertaForTokenClassification.from_pretrained(MODEL_NAME, config=config)
        trainer = StreamingTrainer(model, tokenizer, syllable_vocab, device)
        start_chunk = trainer.load_checkpoint(checkpoint_path)
        print(f"Resuming from chunk {start_chunk}")
    else:
        print("Creating new model...")
        config = RobertaConfig.from_pretrained(MODEL_NAME, num_labels=syllable_vocab.get_vocab_size())
        model = RobertaForTokenClassification.from_pretrained(MODEL_NAME, config=config)
        trainer = StreamingTrainer(model, tokenizer, syllable_vocab, device)
        start_chunk = 0
    
    # --- Training Loop ---
    total_chunks = dataset_manager.get_total_chunks()
    print(f"Starting training from chunk {start_chunk} to {total_chunks - 1}")
    
    for chunk_idx in range(start_chunk, total_chunks):
        print(f"\n{'='*60}")
        print(f"TRAINING CHUNK {chunk_idx + 1}/{total_chunks}")
        print(f"{'='*60}")
        
        # Load chunk data
        chunk_data = dataset_manager.load_chunk(chunk_idx)
        if chunk_data is None:
            print(f"Failed to load chunk {chunk_idx}. Skipping...")
            continue
        
        print(f"Chunk {chunk_idx} contains {len(chunk_data)} samples")
        
        # Train on this chunk
        start_time = datetime.now()
        trainer.train_on_chunk(
            chunk_data, 
            epochs_per_chunk=EPOCHS_PER_CHUNK,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE
        )
        end_time = datetime.now()
        
        training_time = (end_time - start_time).total_seconds()
        print(f"Chunk {chunk_idx} training completed in {training_time:.2f} seconds")
        
        # Save checkpoint after each chunk
        checkpoint_metadata = {
            'chunk_idx': chunk_idx,
            'total_chunks': total_chunks,
            'training_time': training_time,
            'timestamp': datetime.now().isoformat()
        }
        
        trainer.save_checkpoint(checkpoint_path, chunk_idx + 1, checkpoint_metadata)
        
        # Save intermediate model every 5 chunks
        if (chunk_idx + 1) % 5 == 0:
            intermediate_path = os.path.join(MODEL_SAVE_PATH, f"model_checkpoint_{chunk_idx + 1}")
            trainer.model.save_pretrained(intermediate_path)
            print(f"Intermediate model saved: {intermediate_path}")
    
    # --- Save Final Model ---
    print(f"\n{'='*60}")
    print("TRAINING COMPLETED! Saving final model...")
    print(f"{'='*60}")
    
    trainer.model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    
    print(f"Final model saved to: {MODEL_SAVE_PATH}")
    print("Training completed successfully!")

def build_vocabulary_from_chunks(chunk_files):
    """
    Build vocabulary từ tất cả chunk files.
    """
    import json
    from phoBERT import split_to_syllables
    
    syllable_vocab = SyllableVocabulary()
    all_syllables = set()
    
    for chunk_file in chunk_files:
        print(f"Processing {chunk_file} for vocabulary...")
        with open(chunk_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for sentence in data:
            syllables = split_to_syllables(sentence)
            all_syllables.update(syllables)
    
    # Build vocabulary
    for syllable in sorted(list(all_syllables)):
        if syllable not in syllable_vocab.accented_syllable_to_id:
            new_id = len(syllable_vocab.accented_syllable_to_id)
            syllable_vocab.accented_syllable_to_id[syllable] = new_id
            syllable_vocab.id_to_accented_syllable[new_id] = syllable
    
    print(f"Vocabulary built with {syllable_vocab.get_vocab_size()} unique syllables")
    return syllable_vocab

def split_dataset(input_file, output_dir, chunk_size):
    """
    Utility function để split large dataset.
    """
    dataset_manager = StreamingDatasetManager(output_dir, chunk_size)
    dataset_manager.split_large_dataset(input_file, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming Training for Vietnamese Accent Restoration")
    parser.add_argument("--split-data", action="store_true", help="Split large dataset into chunks")
    parser.add_argument("--input-file", type=str, help="Input file to split")
    parser.add_argument("--output-dir", type=str, default="./corpus_splitted", help="Output directory for chunks")
    parser.add_argument("--chunk-size", type=int, default=1000000, help="Size of each chunk")
    parser.add_argument("--train", action="store_true", help="Start training")
    
    args = parser.parse_args()
    
    if args.split_data:
        if not args.input_file:
            print("Please provide --input-file when using --split-data")
            exit(1)
        split_dataset(args.input_file, args.output_dir, args.chunk_size)
    elif args.train:
        streaming_train()
    else:
        # Default behavior: start training
        streaming_train()
