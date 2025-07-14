import torch
import json
import os
import glob
from datetime import datetime
import matplotlib.pyplot as plt
from model_architecture import EnhancedACausalTCN, ACausalTCN

class TrainingMonitor:
    """Monitor và test model trong quá trình training"""
    
    def __init__(self, save_dir="models/progressive"):
        self.save_dir = save_dir
        self.test_samples = [
            "toi di hoc",
            "hom nay troi dep",
            "ban co khoe khong",
            "chung ta cung nhau lam viec",
            "viet nam la dat nuoc xinh dep",
            "em yeu anh rat nhieu",
            "hoc sinh can phai co y thuc hoc tap",
            "cong ty chung toi dang phat trien manh",
            "gia dinh la noi quan trong nhat",
            "moi nguoi can giu gin suc khoe"
        ]
    
    def load_latest_model(self):
        """Load model mới nhất"""
        try:
            # Try best model first
            best_model_path = f"{self.save_dir}/best_model.pth"
            if os.path.exists(best_model_path):
                checkpoint = torch.load(best_model_path, map_location='cpu')
                return self._create_model_from_checkpoint(checkpoint)
            
            # Try latest checkpoint
            checkpoint_path = f"{self.save_dir}/checkpoints/latest_checkpoint.pth"
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                
                # Load vocabulary
                vocab_path = f"{self.save_dir}/vocabulary.json"
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    vocab_data = json.load(f)
                
                checkpoint['char_to_idx'] = vocab_data['char_to_idx']
                checkpoint['idx_to_char'] = vocab_data['idx_to_char']
                checkpoint['model_config'] = {
                    'vocab_size': vocab_data['vocab_size'],
                    'model_type': 'enhanced'  # default
                }
                
                return self._create_model_from_checkpoint(checkpoint)
            
            return None, None, None
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, None, None
    
    def _create_model_from_checkpoint(self, checkpoint):
        """Tạo model từ checkpoint"""
        char_to_idx = checkpoint['char_to_idx']
        idx_to_char = checkpoint['idx_to_char']
        model_config = checkpoint.get('model_config', {})
        
        vocab_size = model_config.get('vocab_size', len(char_to_idx))
        model_type = model_config.get('model_type', 'enhanced')
        
        if model_type == "enhanced":
            model = EnhancedACausalTCN(
                vocab_size=vocab_size,
                output_size=vocab_size,
                embedding_dim=256,
                hidden_dim=512,
                num_layers=8,
                dropout=0.15,
                use_attention=True
            )
        else:
            model = ACausalTCN(
                vocab_size=vocab_size,
                output_size=vocab_size,
                embedding_dim=128,
                hidden_dim=256,
                num_layers=4,
                dropout=0.1
            )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, char_to_idx, idx_to_char
    
    def test_model_performance(self):
        """Test performance của model hiện tại"""
        model, char_to_idx, idx_to_char = self.load_latest_model()
        
        if model is None:
            print("No model found to test!")
            return
        
        print("Testing current model performance:")
        print("=" * 50)
        
        for test_input in self.test_samples:
            prediction = self.predict_text(model, test_input, char_to_idx, idx_to_char)
            print(f"Input:  {test_input}")
            print(f"Output: {prediction}")
            print("-" * 30)
    
    def predict_text(self, model, text, char_to_idx, idx_to_char, max_length=256):
        """Predict text với model"""
        with torch.no_grad():
            # Encode input
            input_ids = [char_to_idx.get(c, 1) for c in text[:max_length]]
            input_ids += [0] * (max_length - len(input_ids))
            
            input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
            
            # Predict
            outputs = model(input_tensor)
            predictions = outputs.argmax(dim=-1)[0]
            
            # Decode output
            result = ""
            for i in range(len(text)):
                pred_idx = predictions[i].item()
                if pred_idx in idx_to_char:
                    result += idx_to_char[pred_idx]
                else:
                    result += text[i]  # fallback to original
            
            return result
    
    def get_training_status(self):
        """Lấy trạng thái training hiện tại"""
        state_file = f"{self.save_dir}/training_state.json"
        
        if not os.path.exists(state_file):
            print("No training state found!")
            return
        
        with open(state_file, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        print("Training Status:")
        print("=" * 50)
        print(f"Current Sample: {state.get('current_sample_idx', 0) + 1}/{state.get('total_samples', 0)}")
        print(f"Global Steps: {state.get('global_step', 0)}")
        print(f"Best Val Loss: {state.get('best_val_loss', 'N/A'):.4f}")
        
        if state.get('start_time'):
            import time
            elapsed = time.time() - state['start_time']
            print(f"Training Time: {elapsed/3600:.1f} hours")
            
            progress = (state.get('current_sample_idx', 0) + 1) / max(state.get('total_samples', 1), 1)
            if progress > 0:
                estimated_total = elapsed / progress
                remaining = estimated_total - elapsed
                print(f"Estimated Remaining: {remaining/3600:.1f} hours")
        
        # Show recent losses
        train_losses = state.get('train_losses', [])
        val_losses = state.get('val_losses', [])
        
        if train_losses:
            print(f"Recent Train Loss: {train_losses[-1]:.4f}")
        if val_losses:
            print(f"Recent Val Loss: {val_losses[-1]:.4f}")
    
    def plot_training_curves(self):
        """Vẽ đồ thị training curves"""
        state_file = f"{self.save_dir}/training_state.json"
        
        if not os.path.exists(state_file):
            print("No training state found!")
            return
        
        with open(state_file, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        train_losses = state.get('train_losses', [])
        val_losses = state.get('val_losses', [])
        
        if not train_losses and not val_losses:
            print("No loss data found!")
            return
        
        plt.figure(figsize=(12, 4))
        
        # Loss curves
        plt.subplot(1, 2, 1)
        if train_losses:
            plt.plot(train_losses, label='Train Loss', alpha=0.7)
        if val_losses:
            plt.plot(val_losses, label='Val Loss', alpha=0.7)
        plt.xlabel('Sample')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Progress info
        plt.subplot(1, 2, 2)
        current_sample = state.get('current_sample_idx', 0)
        total_samples = state.get('total_samples', 0)
        
        if total_samples > 0:
            progress = [i / total_samples * 100 for i in range(current_sample + 1)]
            plt.plot(progress, label='Training Progress (%)')
            plt.xlabel('Sample')
            plt.ylabel('Progress (%)')
            plt.title('Training Progress')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/training_curves.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Training curves saved to {self.save_dir}/training_curves.png")

def resume_training():
    """Resume training từ checkpoint"""
    print("Resuming training from checkpoint...")
    
    # Import và chạy main function
    from progressive_trainer import main
    main()

def quick_test():
    """Quick test model hiện tại"""
    monitor = TrainingMonitor()
    monitor.test_model_performance()

def show_status():
    """Hiển thị trạng thái training"""
    monitor = TrainingMonitor()
    monitor.get_training_status()

def plot_curves():
    """Vẽ training curves"""
    monitor = TrainingMonitor()
    monitor.plot_training_curves()

def clean_checkpoints(keep_latest=True):
    """Dọn dẹp checkpoints cũ"""
    save_dir = "models/progressive"
    checkpoint_dir = f"{save_dir}/checkpoints"
    
    if not os.path.exists(checkpoint_dir):
        print("No checkpoint directory found!")
        return
    
    checkpoints = glob.glob(f"{checkpoint_dir}/*.pth")
    
    if keep_latest and len(checkpoints) > 1:
        # Keep only latest
        latest_checkpoint = max(checkpoints, key=os.path.getmtime)
        
        for checkpoint in checkpoints:
            if checkpoint != latest_checkpoint:
                os.remove(checkpoint)
                print(f"Removed {checkpoint}")
        
        print(f"Kept latest checkpoint: {latest_checkpoint}")
    else:
        for checkpoint in checkpoints:
            os.remove(checkpoint)
            print(f"Removed {checkpoint}")

def backup_model():
    """Backup model và training state"""
    save_dir = "models/progressive"
    backup_dir = f"models/backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if not os.path.exists(save_dir):
        print("No model directory found!")
        return
    
    import shutil
    shutil.copytree(save_dir, backup_dir)
    print(f"Model backed up to {backup_dir}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python training_utils.py test     - Test current model")
        print("  python training_utils.py status   - Show training status")
        print("  python training_utils.py plot     - Plot training curves")
        print("  python training_utils.py resume   - Resume training")
        print("  python training_utils.py clean    - Clean old checkpoints")
        print("  python training_utils.py backup   - Backup current model")
    else:
        command = sys.argv[1].lower()
        
        if command == "test":
            quick_test()
        elif command == "status":
            show_status()
        elif command == "plot":
            plot_curves()
        elif command == "resume":
            resume_training()
        elif command == "clean":
            clean_checkpoints()
        elif command == "backup":
            backup_model()
        else:
            print(f"Unknown command: {command}") 