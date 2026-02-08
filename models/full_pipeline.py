"""
Full Gait-Based Deepfake Detection Pipeline
============================================
Complete model combining:
1. GaitEncoder (CNN for spatial features)
2. DualPathTemporalModel (BiLSTM + Transformer)
3. IdentityVerifier (comparison network)

Input: Video gait features + Claimed identity features
Output: AUTHENTIC or DEEPFAKE prediction

Author: DeepFake Detection Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from models.gait_encoder import GaitEncoder, MultiScaleGaitEncoder
from models.temporal_model import DualPathTemporalModel
from models.identity_verifier import IdentityVerifier, TripletLossNetwork


class GaitDeepfakeDetector(nn.Module):
    """
    Complete gait-based deepfake detection model.
    
    Architecture:
    ┌──────────────────────────────────────────────────────────────┐
    │                    Video Gait Features                        │
    │                    (T × input_dim)                            │
    └──────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                    Gait Encoder (CNN)                         │
    │                    Spatial feature extraction                 │
    └──────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌──────────────────────────────────────────────────────────────┐
    │              Dual-Path Temporal Model                         │
    │              BiLSTM (local) + Transformer (global)            │
    └──────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                    Gait Embedding                             │
    │                    (embedding_dim)                            │
    └──────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                 Identity Verification                         │
    │       Compare with claimed identity → AUTHENTIC/DEEPFAKE      │
    └──────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self,
                 input_dim: int = 78,  # 36 coords + 6 angles + 36 velocities
                 encoder_hidden_dims: Tuple[int, ...] = (64, 128, 256),
                 encoder_output_dim: int = 256,
                 lstm_hidden: int = 128,
                 lstm_layers: int = 2,
                 transformer_d_model: int = 256,
                 transformer_heads: int = 8,
                 transformer_layers: int = 4,
                 embedding_dim: int = 256,
                 verification_hidden: int = 128,
                 dropout: float = 0.3,
                 use_multi_scale_encoder: bool = False):
        super().__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # 1. Gait Encoder (CNN)
        if use_multi_scale_encoder:
            self.gait_encoder = MultiScaleGaitEncoder(
                input_dim=input_dim,
                hidden_dim=encoder_hidden_dims[0],
                output_dim=encoder_output_dim,
                dropout=dropout
            )
        else:
            self.gait_encoder = GaitEncoder(
                input_dim=input_dim,
                hidden_dims=encoder_hidden_dims,
                output_dim=encoder_output_dim,
                dropout=dropout
            )
        
        # 2. Temporal Model (BiLSTM + Transformer)
        self.temporal_model = DualPathTemporalModel(
            input_dim=encoder_output_dim,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            transformer_d_model=transformer_d_model,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
            output_dim=embedding_dim,
            dropout=dropout
        )
        
        # 3. Identity Verifier (kept for inference API compatibility)
        self.identity_verifier = IdentityVerifier(
            embedding_dim=embedding_dim,
            hidden_dim=verification_hidden,
            dropout=dropout
        )
        
        # 4. Difference-based verification classifier
        # Instead of encoding both sequences → comparing embeddings (causes collapse),
        # directly compute per-timestep differences and classify with a small CNN.
        # This gives the model direct access to gait differences.
        comparison_dim = input_dim * 3  # diff, abs_diff, product
        diff_hidden = verification_hidden
        self.diff_conv = nn.Sequential(
            nn.Conv1d(comparison_dim, diff_hidden, kernel_size=7, padding=3),
            nn.BatchNorm1d(diff_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(diff_hidden, diff_hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(diff_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(diff_hidden, diff_hidden // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(diff_hidden // 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.diff_classifier = nn.Sequential(
            nn.Linear(diff_hidden // 2, diff_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(diff_hidden // 2, 2)  # [deepfake, authentic]
        )
        
        # 5. Binary classification head (optional standalone classifier)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 2)  # [fake, real]
        )
        
        # For Grad-CAM: store intermediate features
        self.intermediate_features = {}
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights. Skip LSTM (uses PyTorch default init)."""
        for module in self.modules():
            if isinstance(module, nn.LSTM):
                continue  # PyTorch default LSTM init is better than Xavier
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def encode_gait(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode gait sequence into embedding.
        
        Args:
            x: (batch, seq_len, input_dim) - gait features
            
        Returns:
            embedding: (batch, embedding_dim)
            sequence_features: (batch, seq_len, hidden_dim) - for Grad-CAM
        """
        # CNN encoding
        encoded = self.gait_encoder(x)  # (B, T, encoder_output)
        
        # Store for Grad-CAM
        self.intermediate_features['encoder_output'] = encoded
        
        # Temporal modeling
        sequence_features, embedding = self.temporal_model(encoded)
        
        # Store for Grad-CAM
        self.intermediate_features['temporal_output'] = sequence_features
        
        return embedding, sequence_features
    
    def forward(self, 
                video_features: torch.Tensor,
                claimed_features: Optional[torch.Tensor] = None,
                mode: str = 'verification') -> Dict[str, torch.Tensor]:
        """
        Forward pass for deepfake detection.
        
        Args:
            video_features: (batch, seq_len, input_dim) - extracted gait features
            claimed_features: (batch, seq_len, input_dim) - enrolled identity features
            mode: 'verification' (with identity), 'classification' (standalone), or 'embedding'
            
        Returns:
            Dictionary with predictions, embeddings, and intermediate features
        """
        # Encode video gait
        video_embedding, video_seq_features = self.encode_gait(video_features)
        
        output = {
            'video_embedding': video_embedding,
            'video_sequence_features': video_seq_features
        }
        
        if mode == 'embedding':
            # Just return embeddings (for enrollment or feature extraction)
            return output
        
        if mode == 'classification':
            # Standalone classification without identity verification
            logits = self.classifier(video_embedding)
            probs = F.softmax(logits, dim=1)
            prediction = probs.argmax(dim=1)
            
            output.update({
                'logits': logits,
                'probs': probs,
                'prediction': prediction  # 1 = real, 0 = fake
            })
            
        elif mode == 'verification':
            # Identity verification mode — difference-based approach
            if claimed_features is None:
                raise ValueError("claimed_features required for verification mode")
            
            # Compute per-timestep comparison features from raw inputs
            diff = video_features - claimed_features          # (B, T, D)
            abs_diff = torch.abs(diff)                         # (B, T, D)
            product = video_features * claimed_features        # (B, T, D)
            combined = torch.cat([diff, abs_diff, product], dim=2)  # (B, T, D*3)
            
            # CNN expects (B, C, T)
            x = combined.permute(0, 2, 1)
            x = self.diff_conv(x)          # (B, hidden//2, 1)
            x = x.squeeze(-1)              # (B, hidden//2)
            logits = self.diff_classifier(x)  # (B, 2)
            
            probs = F.softmax(logits, dim=1)
            prediction = probs.argmax(dim=1)
            similarity = probs[:, 1]  # P(authentic) as similarity proxy
            confidence = probs.max(dim=1).values
            
            # Also compute claimed embedding for diagnostics
            claimed_avg = claimed_features.mean(dim=1)
            output['claimed_embedding'] = claimed_avg  # raw mean for diagnostics
            
            output.update({
                'verification': {
                    'logits': logits,
                    'prediction': prediction,
                    'probs': probs,
                    'similarity': similarity,
                    'confidence': confidence
                },
                'is_authentic': prediction,
                'similarity': similarity,
                'confidence': confidence
            })
        
        return output
    
    def verify_identity(self, 
                        video_features: torch.Tensor,
                        claimed_features: torch.Tensor) -> Tuple[bool, float, float]:
        """
        Simple API for identity verification inference.
        
        Args:
            video_features: (1, seq_len, input_dim)
            claimed_features: (1, seq_len, input_dim)
            
        Returns:
            is_authentic: bool
            similarity: float (0-1)
            confidence: float (0-1)
        """
        self.eval()
        with torch.no_grad():
            result = self.forward(video_features, claimed_features, mode='verification')
            
            is_authentic = result['is_authentic'].item() == 1
            similarity = result['similarity'].item()
            confidence = result['confidence'].item()
            
        return is_authentic, similarity, confidence
    
    def get_embedding(self, features: torch.Tensor) -> torch.Tensor:
        """Get gait embedding for a sequence (for enrollment)."""
        self.eval()
        with torch.no_grad():
            embedding, _ = self.encode_gait(features)
        return embedding


class GaitDeepfakeDetectorWithTriplet(GaitDeepfakeDetector):
    """
    Extended model with triplet loss support for better embedding learning.
    """
    
    def __init__(self, *args, margin: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.triplet_network = TripletLossNetwork(
            embedding_dim=self.embedding_dim,
            margin=margin
        )
    
    def forward_triplet(self,
                        anchor: torch.Tensor,
                        positive: torch.Tensor,
                        negative: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass for triplet loss training.
        
        Args:
            anchor: (batch, seq_len, input_dim)
            positive: (batch, seq_len, input_dim) - same person
            negative: (batch, seq_len, input_dim) - different person
            
        Returns:
            loss: scalar triplet loss
            metrics: dict with distance metrics
        """
        # Encode all three
        anchor_emb, _ = self.encode_gait(anchor)
        positive_emb, _ = self.encode_gait(positive)
        negative_emb, _ = self.encode_gait(negative)
        
        # Compute triplet loss
        loss, metrics = self.triplet_network(anchor_emb, positive_emb, negative_emb)
        
        return loss, metrics


def create_model(config: Optional[Dict] = None) -> GaitDeepfakeDetector:
    """
    Factory function to create model with config.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Initialized model
    """
    default_config = {
        'input_dim': 78,
        'encoder_hidden_dims': (64, 128, 256),
        'encoder_output_dim': 256,
        'lstm_hidden': 128,
        'lstm_layers': 2,
        'transformer_d_model': 256,
        'transformer_heads': 8,
        'transformer_layers': 4,
        'embedding_dim': 256,
        'verification_hidden': 128,
        'dropout': 0.3,
        'use_multi_scale_encoder': False
    }
    
    if config:
        default_config.update(config)
    
    return GaitDeepfakeDetector(**default_config)


if __name__ == "__main__":
    import torch
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print("\n" + "=" * 60)
    print("Testing GaitDeepfakeDetector")
    print("=" * 60)
    
    # Create model
    model = create_model()
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 60
    input_dim = 78
    
    video_features = torch.randn(batch_size, seq_len, input_dim).to(device)
    claimed_features = torch.randn(batch_size, seq_len, input_dim).to(device)
    
    print(f"\nInput shapes:")
    print(f"  Video features: {video_features.shape}")
    print(f"  Claimed features: {claimed_features.shape}")
    
    # Test verification mode
    print("\n--- Verification Mode ---")
    output = model(video_features, claimed_features, mode='verification')
    print(f"  Video embedding: {output['video_embedding'].shape}")
    print(f"  Is authentic: {output['is_authentic']}")
    print(f"  Similarity: {output['similarity']}")
    print(f"  Confidence: {output['confidence']}")
    
    # Test classification mode
    print("\n--- Classification Mode ---")
    output = model(video_features, mode='classification')
    print(f"  Logits: {output['logits'].shape}")
    print(f"  Predictions: {output['prediction']}")
    
    # Test embedding mode
    print("\n--- Embedding Mode ---")
    output = model(video_features, mode='embedding')
    print(f"  Embedding: {output['video_embedding'].shape}")
    
    # Test triplet model
    print("\n--- Triplet Loss Model ---")
    triplet_model = GaitDeepfakeDetectorWithTriplet(input_dim=input_dim).to(device)
    anchor = torch.randn(batch_size, seq_len, input_dim).to(device)
    positive = torch.randn(batch_size, seq_len, input_dim).to(device)
    negative = torch.randn(batch_size, seq_len, input_dim).to(device)
    
    loss, metrics = triplet_model.forward_triplet(anchor, positive, negative)
    print(f"  Triplet loss: {loss.item():.4f}")
    print(f"  Metrics: {metrics}")
    
    print("\n✓ All tests passed!")
