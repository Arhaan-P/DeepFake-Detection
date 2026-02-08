"""
Temporal Model - BiLSTM + Transformer for Sequence Modeling
============================================================
Captures both short-term and long-term temporal dependencies
in gait sequences.

Architecture:
1. BiLSTM for local temporal patterns (stride, step timing)
2. Transformer encoder for global temporal relationships
3. Parallel processing with fusion

Author: DeepFake Detection Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class BiLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM for capturing local temporal dynamics.
    Good for short-term patterns like stride timing, step rhythm.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 bidirectional: bool = True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.output_dim = hidden_dim * self.num_directions
        
        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(self.output_dim)
    
    def forward(self, x, lengths: Optional[torch.Tensor] = None):
        """
        Args:
            x: (batch, seq_len, input_dim)
            lengths: Optional tensor of sequence lengths for packing
            
        Returns:
            outputs: (batch, seq_len, hidden_dim * num_directions)
            final_hidden: (batch, hidden_dim * num_directions)
        """
        # LSTM forward pass
        outputs, (hidden, cell) = self.lstm(x)
        
        # Apply layer norm
        outputs = self.layer_norm(outputs)
        
        # Combine final hidden states from both directions
        if self.bidirectional:
            # hidden: (num_layers * 2, batch, hidden_dim)
            final_forward = hidden[-2, :, :]  # Last layer, forward
            final_backward = hidden[-1, :, :]  # Last layer, backward
            final_hidden = torch.cat([final_forward, final_backward], dim=1)
        else:
            final_hidden = hidden[-1, :, :]
        
        return outputs, final_hidden


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for capturing long-range temporal dependencies.
    Uses self-attention to model global relationships across the sequence.
    """
    
    def __init__(self,
                 input_dim: int,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 max_len: int = 500):
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection if dimensions don't match
        self.input_proj = nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for better training stability
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer norm
        self.output_norm = nn.LayerNorm(d_model)
        
        self.output_dim = d_model
    
    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: (batch, seq_len, input_dim)
            mask: Optional attention mask
            
        Returns:
            outputs: (batch, seq_len, d_model)
            cls_token: (batch, d_model) - first token output
        """
        # Project input
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer forward
        outputs = self.transformer(x, mask=mask)
        
        # Apply output norm
        outputs = self.output_norm(outputs)
        
        # Use mean pooling for sequence representation
        seq_embedding = outputs.mean(dim=1)
        
        return outputs, seq_embedding


class DualPathTemporalModel(nn.Module):
    """
    Dual-path temporal model combining BiLSTM and Transformer.
    
    - BiLSTM path: Captures short-term local patterns
    - Transformer path: Captures long-range global dependencies
    - Fusion: Combines both for comprehensive temporal modeling
    
    This architecture is based on research showing that combining
    sequential (LSTM) and attention (Transformer) mechanisms achieves
    98% accuracy on deepfake detection tasks.
    """
    
    def __init__(self,
                 input_dim: int = 256,
                 lstm_hidden: int = 128,
                 lstm_layers: int = 2,
                 transformer_d_model: int = 256,
                 transformer_heads: int = 8,
                 transformer_layers: int = 4,
                 output_dim: int = 256,
                 dropout: float = 0.3):
        super().__init__()
        
        # BiLSTM path
        self.bilstm = BiLSTMEncoder(
            input_dim=input_dim,
            hidden_dim=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout
        )
        
        # Transformer path
        self.transformer = TransformerEncoder(
            input_dim=input_dim,
            d_model=transformer_d_model,
            nhead=transformer_heads,
            num_layers=transformer_layers,
            dropout=dropout
        )
        
        # Fusion layer
        fusion_input_dim = self.bilstm.output_dim + self.transformer.output_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        self.output_dim = output_dim
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
            
        Returns:
            sequence_output: (batch, seq_len, output_dim)
            embedding: (batch, output_dim) - fused sequence representation
        """
        # BiLSTM path
        lstm_output, lstm_embedding = self.bilstm(x)
        
        # Transformer path
        trans_output, trans_embedding = self.transformer(x)
        
        # Fuse sequence-level representations
        fused_embedding = torch.cat([lstm_embedding, trans_embedding], dim=1)
        embedding = self.fusion(fused_embedding)
        
        # Fuse frame-level outputs (optional, for Grad-CAM)
        # Align dimensions if needed
        if lstm_output.size(-1) != trans_output.size(-1):
            lstm_output = F.pad(lstm_output, (0, trans_output.size(-1) - lstm_output.size(-1)))
        
        sequence_output = lstm_output + trans_output  # Residual fusion
        
        return sequence_output, embedding


class TemporalAttentionPool(nn.Module):
    """
    Attention-based temporal pooling.
    Learns which frames are most important for the final decision.
    """
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
            
        Returns:
            pooled: (batch, input_dim)
            attention_weights: (batch, seq_len)
        """
        # Compute attention weights
        attn_weights = self.attention(x).squeeze(-1)  # (B, T)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Weighted sum
        pooled = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # (B, D)
        
        return pooled, attn_weights


if __name__ == "__main__":
    # Test the temporal models
    print("Testing Temporal Models...")
    print("-" * 50)
    
    # Sample input: batch=4, seq_len=60, features=256
    x = torch.randn(4, 60, 256)
    
    # Test BiLSTM
    bilstm = BiLSTMEncoder(input_dim=256, hidden_dim=128)
    outputs, hidden = bilstm(x)
    print(f"BiLSTM: {x.shape} -> outputs: {outputs.shape}, hidden: {hidden.shape}")
    
    # Test Transformer
    transformer = TransformerEncoder(input_dim=256, d_model=256)
    outputs, embedding = transformer(x)
    print(f"Transformer: {x.shape} -> outputs: {outputs.shape}, embedding: {embedding.shape}")
    
    # Test Dual Path
    dual = DualPathTemporalModel(input_dim=256, output_dim=256)
    seq_out, emb = dual(x)
    print(f"DualPath: {x.shape} -> seq_out: {seq_out.shape}, embedding: {emb.shape}")
    
    # Test Attention Pooling
    pool = TemporalAttentionPool(input_dim=256)
    pooled, weights = pool(x)
    print(f"AttentionPool: {x.shape} -> pooled: {pooled.shape}, weights: {weights.shape}")
    
    # Count parameters
    print("-" * 50)
    print(f"BiLSTM params: {sum(p.numel() for p in bilstm.parameters()):,}")
    print(f"Transformer params: {sum(p.numel() for p in transformer.parameters()):,}")
    print(f"DualPath params: {sum(p.numel() for p in dual.parameters()):,}")
