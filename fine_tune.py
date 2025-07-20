import torch
from transformers import RobertaForTokenClassification, RobertaTokenizerFast, AdamW, get_scheduler
from torch.utils.data import DataLoader, random_split
import os
import pickle
import argparse
from datetime import datetime
from tqdm import tqdm

# Import custom classes
from phoBERT import SyllableVocabulary
from streaming_dataset import ChunkDataset

class FineTuner:
    """
    Class để fine-tune model đã train sẵn trên domain-specific data.
    """
    def __init__(self, pretrained_model_path, device='cuda'):
        self.device = device
        self.pretrained_model_path = pretrained_model_path
        
        # Load vocabulary
        vocab_path = os.path.join(pretrained_model_path, "syllable_vocab.pkl")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
            
        with open(vocab_path, 'rb') as f:
            self.syllable_vocab = pickle.load(f)
        
        print(f"Loaded vocabulary with {self.syllable_vocab.get_vocab_size()} syllables")
        
        # Load tokenizer
        self.tokenizer = RobertaTokenizerFast.from_pretrained(pretrained_model_path)
        
        # Load model
        self.model = RobertaForTokenClassification.from_pretrained(pretrained_model_path)
        self.model.to(device)
        
        print(f"Loaded pretrained model from: {pretrained_model_path}")
        print(f"Model has {self.model.config.num_labels} output labels")
    
    def prepare_fine_tuning_data(self, data_file):
        """
        Chuẩn bị data cho fine-tuning.
        """
        print(f"Loading fine-tuning data from: {data_file}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            if data_file.endswith('.json'):
                import json
                data = json.load(f)
            else:
                data = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(data)} samples for fine-tuning")
        return data
    
    def fine_tune(self, fine_tune_data, epochs=5, batch_size=16, learning_rate=1e-5, 
                  val_split=0.1, save_path=None, freeze_layers=None):
        """
        Fine-tune model trên domain-specific data.
        
        Args:
            fine_tune_data: List of sentences for fine-tuning
            epochs: Number of epochs
            batch_size: Batch size (smaller than training)
            learning_rate: Learning rate (lower than training)
            val_split: Validation split ratio
            save_path: Path to save fine-tuned model
            freeze_layers: List of layer names to freeze, None to train all
        """
        
        # Freeze specified layers
        if freeze_layers:
            self.freeze_model_layers(freeze_layers)
        
        # Create dataset
        dataset = ChunkDataset(fine_tune_data, self.tokenizer, self.syllable_vocab)
        
        # Split train/val
        if val_split > 0:
            val_size = int(val_split * len(dataset))
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        else:
            train_dataset = dataset
            val_dataset = None
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
        
        # Setup optimizer with lower learning rate
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        num_training_steps = epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=int(0.1 * num_training_steps),  # 10% warmup
            num_training_steps=num_training_steps,
        )
        
        print(f"Starting fine-tuning...")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset) if val_dataset else 0}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        
        self.model.train()
        best_val_loss = float('inf')
        training_history = []
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 40)
            
            # Training
            total_train_loss = 0
            progress_bar = tqdm(train_dataloader, desc=f"Fine-tuning Epoch {epoch + 1}")
            
            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                total_train_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item(), 'lr': lr_scheduler.get_last_lr()[0]})
            
            avg_train_loss = total_train_loss / len(train_dataloader)
            
            # Validation
            val_loss = None
            if val_dataloader:
                val_loss = self.validate(val_dataloader)
                print(f"Validation loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if save_path:
                        best_model_path = os.path.join(save_path, "best_model")
                        os.makedirs(best_model_path, exist_ok=True)
                        self.model.save_pretrained(best_model_path)
                        self.tokenizer.save_pretrained(best_model_path)
                        # Save vocabulary
                        with open(os.path.join(best_model_path, "syllable_vocab.pkl"), 'wb') as f:
                            pickle.dump(self.syllable_vocab, f)
                        print(f"Best model saved to: {best_model_path}")
            
            # Record training history
            epoch_info = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'learning_rate': lr_scheduler.get_last_lr()[0]
            }
            training_history.append(epoch_info)
            
            print(f"Average training loss: {avg_train_loss:.4f}")
        
        # Save final model
        if save_path:
            final_model_path = os.path.join(save_path, "final_model")
            os.makedirs(final_model_path, exist_ok=True)
            self.model.save_pretrained(final_model_path)
            self.tokenizer.save_pretrained(final_model_path)
            # Save vocabulary
            with open(os.path.join(final_model_path, "syllable_vocab.pkl"), 'wb') as f:
                pickle.dump(self.syllable_vocab, f)
            
            # Save training history
            import json
            with open(os.path.join(save_path, "fine_tuning_history.json"), 'w') as f:
                json.dump(training_history, f, indent=2)
            
            print(f"Final model saved to: {final_model_path}")
        
        return training_history
    
    def freeze_model_layers(self, layer_patterns):
        """
        Freeze specified layers của model.
        
        Args:
            layer_patterns: List of layer name patterns to freeze
                          e.g., ['roberta.embeddings', 'roberta.encoder.layer.0']
        """
        frozen_params = 0
        total_params = 0
        
        for name, param in self.model.named_parameters():
            total_params += 1
            should_freeze = any(pattern in name for pattern in layer_patterns)
            
            if should_freeze:
                param.requires_grad = False
                frozen_params += 1
                print(f"Frozen: {name}")
            else:
                param.requires_grad = True
        
        print(f"Frozen {frozen_params}/{total_params} parameters")
        print(f"Trainable parameters: {total_params - frozen_params}")
    
    def validate(self, val_dataloader):
        """
        Validation function.
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
        
        self.model.train()
        return total_loss / len(val_dataloader)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Vietnamese Accent Restoration Model")
    parser.add_argument("--pretrained-model", type=str, required=True,
                       help="Path to pretrained model directory")
    parser.add_argument("--data-file", type=str, required=True,
                       help="Path to fine-tuning data file")
    parser.add_argument("--save-path", type=str, required=True,
                       help="Path to save fine-tuned model")
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of epochs for fine-tuning")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size for fine-tuning")
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                       help="Learning rate for fine-tuning")
    parser.add_argument("--val-split", type=float, default=0.1,
                       help="Validation split ratio")
    parser.add_argument("--freeze-embeddings", action="store_true",
                       help="Freeze embedding layers")
    parser.add_argument("--freeze-encoder-layers", type=int, default=0,
                       help="Number of encoder layers to freeze (from bottom)")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize fine-tuner
    fine_tuner = FineTuner(args.pretrained_model, device)
    
    # Prepare data
    fine_tune_data = fine_tuner.prepare_fine_tuning_data(args.data_file)
    
    # Setup layer freezing
    freeze_layers = []
    if args.freeze_embeddings:
        freeze_layers.append("roberta.embeddings")
    
    if args.freeze_encoder_layers > 0:
        for i in range(args.freeze_encoder_layers):
            freeze_layers.append(f"roberta.encoder.layer.{i}")
    
    # Start fine-tuning
    start_time = datetime.now()
    
    training_history = fine_tuner.fine_tune(
        fine_tune_data=fine_tune_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_split=args.val_split,
        save_path=args.save_path,
        freeze_layers=freeze_layers if freeze_layers else None
    )
    
    end_time = datetime.now()
    training_duration = (end_time - start_time).total_seconds()
    
    print(f"\nFine-tuning completed!")
    print(f"Total time: {training_duration:.2f} seconds")
    print(f"Models saved to: {args.save_path}")

if __name__ == "__main__":
    main()
