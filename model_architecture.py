import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """
    Positional Encoding cho sequence modeling
    """
    def __init__(self, embedding_dim, max_length=1000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * 
                           (-math.log(10000.0) / embedding_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention cho context modeling
    """
    def __init__(self, embedding_dim, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert embedding_dim % num_heads == 0
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        
        self.fc_out = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, embedding_dim = x.shape
        
        # Linear transformations và reshape cho multi-head
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        out = torch.matmul(attention_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embedding_dim)
        
        return self.fc_out(out)

class EnhancedTCNBlock(nn.Module):
    """
    Enhanced Temporal Convolutional Block với attention và improved residual connections
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout, use_attention=True):
        super(EnhancedTCNBlock, self).__init__()
        
        self.use_attention = use_attention
        
        # Padding để đảm bảo output length = input length
        self.padding = (kernel_size - 1) * dilation // 2
        
        # Depthwise separable convolution cho efficiency
        self.depthwise_conv1 = nn.Conv1d(in_channels, in_channels, kernel_size,
                                       dilation=dilation, padding=self.padding, groups=in_channels)
        self.pointwise_conv1 = nn.Conv1d(in_channels, out_channels, 1)
        
        self.depthwise_conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                                       dilation=dilation, padding=self.padding, groups=out_channels)
        self.pointwise_conv2 = nn.Conv1d(out_channels, out_channels, 1)
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.norm3 = nn.LayerNorm(out_channels)
        
        # Attention layer
        if use_attention:
            self.attention = MultiHeadAttention(out_channels, num_heads=8, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
        # Gating mechanism
        self.gate = nn.Linear(out_channels, out_channels)
        
    def forward(self, x):
        # x shape: (batch_size, channels, seq_len)
        residual = x
        batch_size, channels, seq_len = x.shape
        
        # First depthwise separable conv block
        out = self.depthwise_conv1(x)
        out = self.pointwise_conv1(out)
        out = out.transpose(1, 2)  # (batch_size, seq_len, channels)
        out = self.norm1(out)
        out = F.gelu(out)
        out = self.dropout(out)
        
        # Self-attention (optional)
        if self.use_attention:
            attn_out = self.attention(out)
            out = out + attn_out
            out = self.norm2(out)
        
        out = out.transpose(1, 2)  # Back to (batch_size, channels, seq_len)
        
        # Second depthwise separable conv block
        out = self.depthwise_conv2(out)
        out = self.pointwise_conv2(out)
        out = out.transpose(1, 2)
        out = self.norm3(out)
        
        # Gating mechanism
        gate_values = torch.sigmoid(self.gate(out))
        out = out * gate_values
        
        out = out.transpose(1, 2)
        
        # Residual connection
        if self.residual is not None:
            residual = self.residual(residual)
        
        out = out + residual
        return F.gelu(out)

class EnhancedACausalTCN(nn.Module):
    """
    Enhanced Acausal Temporal Convolutional Network với attention và modern techniques
    """
    
    def __init__(self, vocab_size, output_size, embedding_dim=256, hidden_dim=512, 
                 num_layers=8, kernel_size=3, dropout=0.15, use_attention=True):
        super(EnhancedACausalTCN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embedding_dim)
        
        # Layer scaling cho embedding
        self.embedding_scale = math.sqrt(embedding_dim)
        
        # TCN layers với progressive dilation
        self.tcn_layers = nn.ModuleList()
        
        # Input projection layer
        self.input_projection = nn.Linear(embedding_dim, hidden_dim)
        
        # Lớp đầu tiên
        self.tcn_layers.append(
            EnhancedTCNBlock(hidden_dim, hidden_dim, kernel_size, dilation=1, 
                           dropout=dropout, use_attention=use_attention)
        )
        
        # Các lớp tiếp theo với dilation pattern cải tiến
        dilation_pattern = [2, 4, 8, 16, 2, 4, 8]  # Repeat pattern cho receptive field tốt hơn
        for i in range(1, num_layers):
            dilation = dilation_pattern[(i-1) % len(dilation_pattern)]
            self.tcn_layers.append(
                EnhancedTCNBlock(hidden_dim, hidden_dim, kernel_size, dilation=dilation, 
                               dropout=dropout, use_attention=use_attention and i % 2 == 0)
            )
        
        # Output layers với highway connection
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.highway_gate = nn.Linear(hidden_dim, hidden_dim)
        self.highway_transform = nn.Linear(hidden_dim, hidden_dim)
        
        # Final classifier với dropout
        self.pre_classifier_dropout = nn.Dropout(dropout * 1.5)
        self.classifier = nn.Linear(hidden_dim, output_size)
        
        # Label smoothing weight
        self.label_smoothing = 0.1
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x) * self.embedding_scale  # (batch_size, seq_len, embedding_dim)
        
        # Add positional encoding
        embedded = embedded.transpose(0, 1)  # (seq_len, batch_size, embedding_dim)
        embedded = self.pos_encoding(embedded)
        embedded = embedded.transpose(0, 1)  # (batch_size, seq_len, embedding_dim)
        
        # Project to hidden dimension
        out = self.input_projection(embedded)  # (batch_size, seq_len, hidden_dim)
        out = out.transpose(1, 2)  # (batch_size, hidden_dim, seq_len)
        
        # Qua các TCN layers
        for tcn_layer in self.tcn_layers:
            out = tcn_layer(out)
        
        out = out.transpose(1, 2)  # (batch_size, seq_len, hidden_dim)
        out = self.output_norm(out)
        
        # Highway connection
        gate = torch.sigmoid(self.highway_gate(out))
        transform = F.gelu(self.highway_transform(out))
        out = gate * transform + (1 - gate) * out
        
        # Final classification
        out = self.pre_classifier_dropout(out)
        logits = self.classifier(out)  # (batch_size, seq_len, output_size)
        
        return logits

class ACausalTCN(nn.Module):
    """
    Acausal Temporal Convolutional Network cho phục hồi dấu tiếng Việt
    Dựa trên nghiên cứu của Alqahtani et al. (EMNLP 2019)
    """
    
    def __init__(self, vocab_size, output_size, embedding_dim=128, hidden_dim=256, num_layers=4, kernel_size=3, dropout=0.1):
        super(ACausalTCN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # TCN layers với dilated convolution
        self.tcn_layers = nn.ModuleList()
        
        # Lớp đầu tiên
        self.tcn_layers.append(
            TCNBlock(embedding_dim, hidden_dim, kernel_size, dilation=1, dropout=dropout)
        )
        
        # Các lớp tiếp theo với dilation tăng dần
        for i in range(1, num_layers):
            dilation = 2 ** i
            self.tcn_layers.append(
                TCNBlock(hidden_dim, hidden_dim, kernel_size, dilation=dilation, dropout=dropout)
            )
        
        # Layer classification cuối
        self.classifier = nn.Linear(hidden_dim, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        embedded = embedded.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)
        
        # Qua các TCN layers
        out = embedded
        for tcn_layer in self.tcn_layers:
            out = tcn_layer(out)
        
        out = out.transpose(1, 2)  # (batch_size, seq_len, hidden_dim)
        out = self.dropout(out)
        
        # Classification cho từng ký tự
        logits = self.classifier(out)  # (batch_size, seq_len, output_size)
        
        return logits

class TCNBlock(nn.Module):
    """
    Temporal Convolutional Block với dilated convolution và residual connection
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(TCNBlock, self).__init__()
        
        # Correct padding to maintain sequence length
        self.padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              dilation=dilation, padding=self.padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding)
        
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x):
        # x shape: (batch_size, channels, seq_len)
        residual = x
        
        # First conv block
        out = self.conv1(x)
        out = out.transpose(1, 2)  # (batch_size, seq_len, channels)
        out = self.norm1(out)
        out = out.transpose(1, 2)  # (batch_size, channels, seq_len)
        out = F.relu(out)
        out = self.dropout(out)
        
        # Second conv block  
        out = self.conv2(out)
        out = out.transpose(1, 2)
        out = self.norm2(out)
        out = out.transpose(1, 2)
        
        # Residual connection
        if self.residual is not None:
            residual = self.residual(residual)
            
        out = out + residual
        return F.relu(out)

class VietnameseAccentRestorer:
    """
    Wrapper class cho mô hình phục hồi dấu tiếng Việt
    """
    
    def __init__(self, model_path=None, use_enhanced_model=True):
        # Bảng ký tự tiếng Việt
        self.no_accent_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:-()[]{}\"'/"
        self.accent_chars = "aăâeêioôơuưyAĂÂEÊIOÔƠUƯYáắấéếíóốớúứýÁẮẤÉẾÍÓỐỚÚỨÝàằầèềìòồờùừỳÀẰẦÈỀÌÒỒỜÙỪỲảẳẩẻểỉỏổởủửỷẢẲẨẺỂỈỎỔỞỦỬỶãẵẫẽễĩõỗỡũữỹÃẴẪẼỄĨÕỖỠŨỮỸạặậẹệịọộợụựỵẠẶẬẸỆỊỌỘỢỤỰỴđĐ"
        
        # Tạo vocabulary
        all_chars = set(self.no_accent_chars + self.accent_chars)
        self.char_to_idx = {char: idx for idx, char in enumerate(sorted(all_chars))}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        
        self.vocab_size = len(self.char_to_idx)
        self.output_size = self.vocab_size
        
        # Khởi tạo mô hình - sử dụng enhanced model mặc định
        if use_enhanced_model:
            self.model = EnhancedACausalTCN(
                vocab_size=self.vocab_size,
                output_size=self.output_size,
                embedding_dim=256,
                hidden_dim=512,
                num_layers=8,
                kernel_size=3,
                dropout=0.15,
                use_attention=True
            )
        else:
            self.model = ACausalTCN(
                vocab_size=self.vocab_size,
                output_size=self.output_size,
                embedding_dim=128,
                hidden_dim=256,
                num_layers=4,
                kernel_size=3,
                dropout=0.1
            )
        
        if model_path:
            self.load_model(model_path)
    
    def encode_text(self, text):
        """Chuyển text thành sequence of indices"""
        return [self.char_to_idx.get(char, 0) for char in text]
    
    def decode_text(self, indices):
        """Chuyển sequence of indices về text"""
        return ''.join([self.idx_to_char.get(idx, '') for idx in indices])
    
    def predict(self, text):
        """Dự đoán text có dấu từ text không dấu"""
        self.model.eval()
        
        # Encode input
        input_ids = self.encode_text(text)
        input_tensor = torch.tensor(input_ids).unsqueeze(0)  # Add batch dimension
        
        # Check device và move tensor đến cùng device với model
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        with torch.no_grad():
            logits = self.model(input_tensor)
            predictions = torch.argmax(logits, dim=-1)
        
        # Decode output
        predicted_indices = predictions.squeeze().tolist()
        predicted_text = self.decode_text(predicted_indices)
        
        # Post-processing: Giữ nguyên khoảng trắng và ký tự đặc biệt
        corrected_text = self.preserve_non_alphabetic(text, predicted_text)
        
        return corrected_text
    
    def predict_multiple(self, text, num_suggestions=3, temperature=0.7):
        """Dự đoán và trả về nhiều gợi ý cho text có dấu từ text không dấu"""
        self.model.eval()
        
        # Encode input
        input_ids = self.encode_text(text)
        input_tensor = torch.tensor(input_ids).unsqueeze(0)  # Add batch dimension
        
        # Check device và move tensor đến cùng device với model
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        with torch.no_grad():
            logits = self.model(input_tensor)  # (batch_size, seq_len, vocab_size)
        
        suggestions = []
        
        # Suggestion đầu tiên: greedy (tốt nhất)
        predictions = torch.argmax(logits, dim=-1)
        predicted_indices = predictions.squeeze().tolist()
        predicted_text = self.decode_text(predicted_indices)
        corrected_text = self.preserve_non_alphabetic(text, predicted_text)
        suggestions.append(corrected_text)
        
        # Thử tạo thêm gợi ý bằng beam search với ngưỡng confidence cao
        for suggestion_idx in range(1, num_suggestions):
            # Sử dụng sampling với temperature thấp để có gợi ý có nghĩa
            predicted_indices = []
            
            for pos in range(logits.size(1)):
                char_logits = logits[0, pos, :] / temperature
                char_probs = torch.softmax(char_logits, dim=0)
                
                # Chỉ xem xét top-3 candidates có confidence cao
                top_probs, top_indices = torch.topk(char_probs, k=3)
                
                # Kiểm tra nếu có candidate thứ 2 với confidence đủ cao (>10% của top-1)
                if len(top_probs) > suggestion_idx and top_probs[suggestion_idx] > top_probs[0] * 0.1:
                    predicted_char_idx = top_indices[suggestion_idx].item()
                else:
                    # Không có alternative có confidence đủ cao, dùng greedy
                    predicted_char_idx = top_indices[0].item()
                
                predicted_indices.append(predicted_char_idx)
            
            # Decode và post-process
            predicted_text = self.decode_text(predicted_indices)
            corrected_text = self.preserve_non_alphabetic(text, predicted_text)
            
            # Chỉ thêm nếu khác biệt có nghĩa và không duplicate
            if (corrected_text != suggestions[0] and 
                corrected_text not in suggestions and
                self._has_meaningful_difference(corrected_text, suggestions[0])):
                suggestions.append(corrected_text)
            else:
                # Không tìm được gợi ý có nghĩa khác, dừng lại
                break
        
        # Nếu chỉ có 1 gợi ý duy nhất có nghĩa, trả về nó
        return suggestions
    
    def _has_meaningful_difference(self, text1, text2):
        """Kiểm tra xem 2 text có sự khác biệt có nghĩa không"""
        if len(text1) != len(text2):
            return False
        
        # Đếm số ký tự khác biệt
        diff_count = sum(1 for c1, c2 in zip(text1, text2) if c1 != c2)
        
        # Chỉ chấp nhận nếu có ít nhất 1 nhưng không quá 3 ký tự khác biệt
        if diff_count < 1 or diff_count > 3:
            return False
        
        # Kiểm tra xem những ký tự khác biệt có phải là variants hợp lý không
        for i, (c1, c2) in enumerate(zip(text1, text2)):
            if c1 != c2:
                if not self._is_valid_variant(c1, c2):
                    return False
        
        return True
    
    def _is_valid_variant(self, original_char, candidate_char):
        """Kiểm tra xem candidate_char có phải là variant hợp lý của original_char không"""
        if not candidate_char:
            return False
            
        original_lower = original_char.lower()
        candidate_lower = candidate_char.lower()
        
        # Mapping các ký tự cơ bản với các variant có dấu
        variant_groups = {
            'a': ['a', 'ă', 'â', 'á', 'à', 'ả', 'ã', 'ạ', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ'],
            'e': ['e', 'ê', 'é', 'è', 'ẻ', 'ẽ', 'ẹ', 'ế', 'ề', 'ể', 'ễ', 'ệ'],
            'i': ['i', 'í', 'ì', 'ỉ', 'ĩ', 'ị'],
            'o': ['o', 'ô', 'ơ', 'ó', 'ò', 'ỏ', 'õ', 'ọ', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ'],
            'u': ['u', 'ư', 'ú', 'ù', 'ủ', 'ũ', 'ụ', 'ứ', 'ừ', 'ử', 'ữ', 'ự'],
            'y': ['y', 'ý', 'ỳ', 'ỷ', 'ỹ', 'ỵ'],
            'd': ['d', 'đ']
        }
        
        # Tìm group của original character
        for base_char, variants in variant_groups.items():
            if original_lower in variants:
                return candidate_lower in variants
        
        # Nếu không phải ký tự có thể có dấu, chỉ chấp nhận chính nó
        return original_lower == candidate_lower
    
    def _is_reasonable_text(self, output_text, input_text):
        """Kiểm tra xem output text có hợp lý không"""
        # Kiểm tra độ dài
        if len(output_text) != len(input_text):
            return False
        
        # Kiểm tra tỷ lệ ký tự hợp lệ
        valid_chars = 0
        for char in output_text:
            if char.isalpha() or char.isspace() or char in ".,!?;:-()[]{}\"'/":
                valid_chars += 1
        
        valid_ratio = valid_chars / len(output_text) if len(output_text) > 0 else 0
        
        # Ít nhất 80% ký tự phải hợp lệ
        return valid_ratio >= 0.8
    
    def preserve_non_alphabetic(self, original, predicted):
        """
        Giữ nguyên khoảng trắng và ký tự không phải chữ cái từ text gốc
        """
        if len(original) != len(predicted):
            return predicted  # Fallback nếu length không khớp
        
        result = []
        for i, (orig_char, pred_char) in enumerate(zip(original, predicted)):
            if orig_char.isalpha():
                # Nếu là chữ cái, dùng kết quả dự đoán
                result.append(pred_char)
            else:
                # Nếu không phải chữ cái (space, số, dấu câu), giữ nguyên
                result.append(orig_char)
        
        return ''.join(result)
    
    def save_model(self, path):
        """Lưu mô hình"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char,
            'vocab_size': self.vocab_size,
            'output_size': self.output_size
        }, path)
    
    def load_model(self, path):
        """Tải mô hình"""
        checkpoint = torch.load(path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.char_to_idx = checkpoint['char_to_idx']
        self.idx_to_char = checkpoint['idx_to_char']
        self.vocab_size = checkpoint['vocab_size']
        self.output_size = checkpoint['output_size']
    
    def export_to_onnx(self, onnx_path, max_seq_len=512):
        """Xuất mô hình sang ONNX format"""
        self.model.eval()
        
        # Kiểm tra device của model
        device = next(self.model.parameters()).device
        
        # Tạo dummy input trên cùng device với model
        dummy_input = torch.randint(0, self.vocab_size, (1, max_seq_len)).to(device)
        
        # Export
        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {1: 'sequence'},
                'output': {1: 'sequence'}
            }
        )
        print(f"Đã xuất mô hình ONNX tại: {onnx_path}")

if __name__ == "__main__":
    # Test mô hình
    restorer = VietnameseAccentRestorer()
    
    # In thông tin mô hình
    total_params = sum(p.numel() for p in restorer.model.parameters())
    print(f"Tổng số tham số: {total_params:,}")
    print(f"Kích thước vocabulary: {restorer.vocab_size}")
    
    # Test forward pass
    test_text = "toi di hoc"
    encoded = restorer.encode_text(test_text)
    print(f"Text: {test_text}")
    print(f"Encoded: {encoded}") 