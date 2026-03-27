import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Positional Encoding
# =========================
def get_temporal_positional_encoding(seq_len, d_model, device):
    """Create temporal positional encoding for sequences."""
    pos = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device) * 
        -(torch.log(torch.tensor(10000.0, device=device)) / d_model)
    )
    
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(pos * div_term)
    if d_model % 2 == 1:
        pe[:, 1::2] = torch.cos(pos * div_term[:-1])
    else:
        pe[:, 1::2] = torch.cos(pos * div_term)
    
    return pe.unsqueeze(0)


# =========================
# Sequence Encoder
# =========================
class SequenceEncoder(nn.Module):
    """
    Encodes time-series sequences (OHLCV data) into feature representations.
    
    Similar to GridEncoder from ARC-Meta but for temporal sequences.
    Uses LSTM + attention to create rich feature representations.
    """
    
    def __init__(self, input_size=5, dim=128):
        super().__init__()
        self.dim = dim
        
        # Initial projection
        self.input_proj = nn.Linear(input_size, dim)
        self.proj_norm = nn.LayerNorm(dim)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=dim,
            hidden_size=dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Self-attention over sequence
        self.attention = nn.MultiheadAttention(
            embed_dim=dim * 2,
            num_heads=4,
            batch_first=True,
            dropout=0.2
        )
        
        # Post-attention processing
        self.post_attn = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, T, input_size) - Batch of sequences
        
        Returns:
            (B, T, dim) - Encoded sequence representations
        """
        # Project input
        x = self.input_proj(x)  # (B, T, dim)
        x = self.proj_norm(x)
        
        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # (B, T, 2*dim)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)  # (B, T, 2*dim)
        
        # Post-attention processing
        feat = self.post_attn(attn_out + lstm_out)  # (B, T, dim) + residual
        
        return feat


# =========================
# Task Encoder
# =========================
class TaskEncoder(nn.Module):
    """
    Extracts task-specific representations from support set (OHLCV sequences).
    
    Learns what patterns matter for predicting returns by comparing
    input sequences and their corresponding return values.
    """
    
    def __init__(self, dim=128, num_heads=4):
        super().__init__()
        self.dim = dim
        
        # Project input and returns to same dim
        self.input_proj = nn.Linear(dim, dim)
        self.returns_proj = nn.Linear(1, dim)
        
        # Combine input and return information
        self.combiner = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        
        # Cross-temporal attention (across support samples)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.2
        )
        
        self.norm = nn.LayerNorm(dim)
        
        # Global aggregation
        self.aggregate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
    
    def forward(self, support_seqs, support_returns):
        """
        Args:
            support_seqs: (B, K, T, dim) - B batch tasks, K support samples, T timesteps
            support_returns: (B, K, output_size) - K target returns per task
        
        Returns:
            task_tokens: (B*K, dim) - Flattened task-specific tokens
        """
        B, K, T, D = support_seqs.shape
        
        # Global average pool over time dimension
        input_pooled = support_seqs.mean(dim=2)  # (B, K, D)
        input_proj = self.input_proj(input_pooled)  # (B, K, D)
        
        # Average returns across output_size (flatten to 1) then project
        returns_avg = support_returns.mean(dim=-1, keepdim=True)  # (B, K, 1)
        returns_proj = self.returns_proj(returns_avg)  # (B, K, D)
        
        # Combine input and return info
        combined = torch.cat([input_proj, returns_proj], dim=-1)  # (B, K, 2*D)
        combined = self.combiner(combined)  # (B, K, D)
        
        # Cross-attention between samples to share patterns
        combined_flat = combined.view(B * K, D).unsqueeze(1)  # (B*K, 1, D)
        combined_all = combined.view(B, K, D)  # (B, K, D)
        
        # Use support samples as context
        task_tokens, _ = self.cross_attn(
            combined_flat,
            combined_all.repeat_interleave(K, dim=0),
            combined_all.repeat_interleave(K, dim=0)
        )  # (B*K, 1, D)
        
        # Aggregate and normalize
        task_tokens = task_tokens.squeeze(1)  # (B*K, D)
        task_tokens = self.norm(task_tokens)
        
        return task_tokens


# =========================
# Predictor Head
# =========================
class PredictorHead(nn.Module):
    """
    Predicts returns given query sequences and task representations.
    Uses cross-attention between query and task context.
    """
    
    def __init__(self, dim=128, output_size=1, num_heads=4):
        super().__init__()
        self.dim = dim
        
        # Cross-attention: query attends to task tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.2
        )
        
        # Local processing
        self.local_conv = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        
        # Regression head
        self.head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim // 2, output_size)
        )
    
    def forward(self, query_feat, task_tokens):
        """
        Args:
            query_feat: (B, T, dim) - Query sequence features
            task_tokens: (B, dim) - Task-specific context
        
        Returns:
            predictions: (B, output_size) - Predicted returns
        """
        B, T, D = query_feat.shape
        
        # Pool query over time
        query_pooled = query_feat.mean(dim=1, keepdim=True)  # (B, 1, D)
        
        # Cross-attention: use task tokens as context
        task_context = task_tokens.unsqueeze(1)  # (B, 1, D)
        
        attn_out, _ = self.cross_attn(
            query_pooled,
            task_context,
            task_context
        )  # (B, 1, D)
        
        # Local processing
        local = self.local_conv(attn_out.squeeze(1))  # (B, D)
        
        # Combine with query info
        combined = local + query_pooled.squeeze(1)  # (B, D)
        
        # Predict
        predictions = self.head(combined)  # (B, output_size)
        
        return predictions


# =========================
# Meta-Learning Model
# =========================
class MetaOHLCVPredictor(nn.Module):
    """
    Meta-learning model for OHLCV-to-Returns prediction.
    
    Architecture:
    1. SequenceEncoder: Encodes all sequences (support + query)
    2. TaskEncoder: Learns task representation from support set
    3. PredictorHead: Predicts returns using task context
    
    Supports both standard training and meta-learning training.
    """
    
    def __init__(self, input_size=5, dim=128, num_heads=4, output_size=1):
        super().__init__()
        
        self.encoder = SequenceEncoder(input_size=input_size, dim=dim)
        self.task_encoder = TaskEncoder(dim=dim, num_heads=num_heads)
        self.predictor = PredictorHead(dim=dim, output_size=output_size, num_heads=num_heads)
    
    def forward(self, support_x, support_y, query_x):
        """
        Forward pass for meta-learning.
        
        Args:
            support_x: (B, K, T, input_size) - K support sequences per task
            support_y: (B, K, output_size) - K target returns
            query_x: (B, Q, T, input_size) - Q query sequences per task
        
        Returns:
            predictions: (B, Q, output_size) - Q predictions per task
        """
        B, K, T, I = support_x.shape
        Q = query_x.shape[1]
        
        # Encode all sequences at once
        all_x = torch.cat([support_x, query_x], dim=1)  # (B, K+Q, T, I)
        all_x_flat = all_x.view(B * (K + Q), T, I)  # (B*(K+Q), T, I)
        
        all_features = self.encoder(all_x_flat)  # (B*(K+Q), T, dim)
        
        # Reshape back
        all_features = all_features.view(B, K + Q, T, -1)  # (B, K+Q, T, dim)
        support_feat = all_features[:, :K]  # (B, K, T, dim)
        query_feat = all_features[:, K:]  # (B, Q, T, dim)
        
        # Learn task representation from support
        task_tokens = self.task_encoder(support_feat, support_y)  # (B*K, dim)
        task_tokens = task_tokens.view(B, K, -1).mean(dim=1)  # (B, dim) - average over K
        
        # Predict on query
        query_feat_flat = query_feat.view(B * Q, T, -1)  # (B*Q, T, dim)
        task_tokens_rep = task_tokens.repeat_interleave(Q, dim=0)  # (B*Q, dim)
        
        predictions = self.predictor(query_feat_flat, task_tokens_rep)  # (B*Q, output_size)
        predictions = predictions.view(B, Q, -1)  # (B, Q, output_size)
        
        return predictions
    
    def forward_standard(self, x):
        """
        Forward pass for standard supervised learning (no meta-learning).
        
        Args:
            x: (B, T, input_size) - Batch of sequences
        
        Returns:
            predictions: (B, output_size)
        """
        # For standard mode, we don't have support/task info
        # Just encode and do a simple prediction
        feat = self.encoder(x)  # (B, T, dim)
        
        # Pool over time
        pooled = feat.mean(dim=1)  # (B, dim)
        
        # Simple prediction head
        predictions = self.predictor.head(pooled)  # (B, output_size)
        
        return predictions


if __name__ == "__main__":
    # Test meta-learning mode
    B, K, Q, T, input_size = 2, 5, 1, 10, 5
    
    model = MetaOHLCVPredictor(input_size=input_size, dim=128, output_size=1)
    
    support_x = torch.randn(B, K, T, input_size)
    support_y = torch.randn(B, K, 1)
    query_x = torch.randn(B, Q, T, input_size)
    
    output = model(support_x, support_y, query_x)
    print(f"Meta-learning output shape: {output.shape}")  # Should be (B, Q, 1)
    
    # Test standard mode
    x = torch.randn(B * K, T, input_size)
    output_std = model.forward_standard(x)
    print(f"Standard mode output shape: {output_std.shape}")  # Should be (B*K, 1)
