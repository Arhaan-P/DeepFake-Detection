"""
Identity Verification Network
==============================
Compares extracted gait embedding with claimed identity embedding
to determine if the video is authentic or a deepfake.

Approach:
1. Siamese-style comparison network
2. Learns a similarity metric between gait embeddings
3. Threshold-based decision for authentication

Author: DeepFake Detection Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class GaitComparisonNetwork(nn.Module):
    """
    Network that compares two gait embeddings and outputs a similarity score.
    
    Uses a Siamese-style architecture where both inputs go through
    the same processing, then are compared.
    """
    
    def __init__(self,
                 embedding_dim: int = 256,
                 hidden_dim: int = 128,
                 dropout: float = 0.3):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Shared projection for both embeddings
        self.proj = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Comparison network
        # Takes concatenated features + element-wise difference + product
        comparison_input_dim = hidden_dim * 4  # concat(a, b, |a-b|, a*b)
        
        self.comparator = nn.Sequential(
            nn.Linear(comparison_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, video_embedding: torch.Tensor, 
                claimed_embedding: torch.Tensor) -> torch.Tensor:
        """
        Compare video embedding with claimed identity embedding.
        
        Args:
            video_embedding: (batch, embedding_dim) - extracted from video
            claimed_embedding: (batch, embedding_dim) - enrolled identity
            
        Returns:
            similarity: (batch,) - similarity scores (higher = more similar)
        """
        # Project both embeddings
        video_proj = self.proj(video_embedding)
        claimed_proj = self.proj(claimed_embedding)
        
        # Compute comparison features
        diff = torch.abs(video_proj - claimed_proj)
        product = video_proj * claimed_proj
        
        # Concatenate all comparison features
        combined = torch.cat([video_proj, claimed_proj, diff, product], dim=1)
        
        # Output similarity score
        similarity = self.comparator(combined).squeeze(-1)
        
        return similarity
    
    def get_cosine_similarity(self, video_embedding: torch.Tensor,
                               claimed_embedding: torch.Tensor) -> torch.Tensor:
        """
        Compute pure cosine similarity (useful for baseline comparison).
        """
        video_proj = self.proj(video_embedding)
        claimed_proj = self.proj(claimed_embedding)
        
        return F.cosine_similarity(video_proj, claimed_proj, dim=1)


class TripletLossNetwork(nn.Module):
    """
    Network for learning gait embeddings using triplet loss.
    
    Objective: Make embeddings of the same person close,
    and embeddings of different people far apart.
    """
    
    def __init__(self, embedding_dim: int = 256, margin: float = 1.0):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.margin = margin
        
        # Embedding projection
        self.proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # L2 normalize for cosine-based triplet loss
        self.normalize = True
    
    def forward(self, anchor: torch.Tensor, 
                positive: torch.Tensor, 
                negative: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute triplet loss.
        
        Args:
            anchor: (batch, embedding_dim) - anchor embedding
            positive: (batch, embedding_dim) - same person as anchor
            negative: (batch, embedding_dim) - different person
            
        Returns:
            loss: scalar triplet loss
            metrics: dict with distances
        """
        # Project embeddings
        anchor_proj = self.proj(anchor)
        positive_proj = self.proj(positive)
        negative_proj = self.proj(negative)
        
        # Normalize if using cosine distance
        if self.normalize:
            anchor_proj = F.normalize(anchor_proj, p=2, dim=1)
            positive_proj = F.normalize(positive_proj, p=2, dim=1)
            negative_proj = F.normalize(negative_proj, p=2, dim=1)
        
        # Compute distances
        pos_distance = torch.sum((anchor_proj - positive_proj) ** 2, dim=1)
        neg_distance = torch.sum((anchor_proj - negative_proj) ** 2, dim=1)
        
        # Triplet loss
        loss = F.relu(pos_distance - neg_distance + self.margin)
        loss = loss.mean()
        
        metrics = {
            'pos_distance': pos_distance.mean().item(),
            'neg_distance': neg_distance.mean().item(),
            'margin_violations': (pos_distance + self.margin > neg_distance).float().mean().item()
        }
        
        return loss, metrics
    
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Get normalized embedding for a single input."""
        proj = self.proj(x)
        if self.normalize:
            proj = F.normalize(proj, p=2, dim=1)
        return proj


class ContrastiveLossNetwork(nn.Module):
    """
    Network using contrastive loss for learning gait similarities.
    
    For pairs of embeddings, learns to:
    - Minimize distance for same-person pairs
    - Maximize distance for different-person pairs
    """
    
    def __init__(self, embedding_dim: int = 256, margin: float = 2.0):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.margin = margin
        
        self.proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def forward(self, embedding1: torch.Tensor,
                embedding2: torch.Tensor,
                labels: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute contrastive loss.
        
        Args:
            embedding1: (batch, embedding_dim)
            embedding2: (batch, embedding_dim)
            labels: (batch,) - 1 if same person, 0 if different
            
        Returns:
            loss: scalar contrastive loss
            metrics: dict with distances
        """
        # Project
        proj1 = F.normalize(self.proj(embedding1), p=2, dim=1)
        proj2 = F.normalize(self.proj(embedding2), p=2, dim=1)
        
        # Euclidean distance
        distance = torch.sqrt(torch.sum((proj1 - proj2) ** 2, dim=1) + 1e-8)
        
        # Contrastive loss
        # Same pair (label=1): minimize distance
        # Different pair (label=0): maximize distance (up to margin)
        loss_same = labels * distance ** 2
        loss_diff = (1 - labels) * F.relu(self.margin - distance) ** 2
        
        loss = (loss_same + loss_diff).mean()
        
        metrics = {
            'avg_distance': distance.mean().item(),
            'same_pair_dist': (distance * labels).sum().item() / (labels.sum().item() + 1e-8),
            'diff_pair_dist': (distance * (1 - labels)).sum().item() / ((1 - labels).sum().item() + 1e-8)
        }
        
        return loss, metrics


class IdentityVerifier(nn.Module):
    """
    Complete identity verification module.
    
    Takes two gait embeddings (video and claimed identity)
    and outputs:
    1. Binary prediction (authentic/deepfake)
    2. Similarity score
    3. Confidence
    """
    
    def __init__(self,
                 embedding_dim: int = 256,
                 hidden_dim: int = 128,
                 threshold: float = 0.5,
                 dropout: float = 0.3):
        super().__init__()
        
        self.threshold = threshold
        self.hidden_dim = hidden_dim
        
        # Shared projection for both embeddings
        self.proj = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Direct classification from rich comparison features
        # concat(a, b, |a-b|, a*b) = hidden_dim * 4
        comparison_dim = hidden_dim * 4
        self.classifier = nn.Sequential(
            nn.Linear(comparison_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # [deepfake, authentic]
        )
    
    def forward(self, video_embedding: torch.Tensor,
                claimed_embedding: torch.Tensor) -> dict:
        """
        Verify if video matches claimed identity.
        
        Args:
            video_embedding: (batch, embedding_dim)
            claimed_embedding: (batch, embedding_dim)
            
        Returns:
            dict with 'prediction', 'similarity', 'logits', 'confidence'
        """
        # Project both embeddings through shared projection
        video_proj = self.proj(video_embedding)
        claimed_proj = self.proj(claimed_embedding)
        
        # Compute rich comparison features
        diff = torch.abs(video_proj - claimed_proj)
        product = video_proj * claimed_proj
        combined = torch.cat([video_proj, claimed_proj, diff, product], dim=1)  # (B, H*4)
        
        # Classify directly from rich comparison
        logits = self.classifier(combined)
        probs = F.softmax(logits, dim=1)
        
        # Cosine similarity for interpretability
        similarity = F.cosine_similarity(video_proj, claimed_proj, dim=1)
        
        # Prediction: 1 = authentic, 0 = deepfake
        prediction = (probs[:, 1] > self.threshold).long()
        confidence = probs.max(dim=1).values
        
        return {
            'prediction': prediction,
            'similarity': (similarity + 1) / 2,  # Map from [-1,1] to [0,1]
            'logits': logits,
            'probs': probs,
            'confidence': confidence
        }
    
    def verify(self, video_embedding: torch.Tensor,
               claimed_embedding: torch.Tensor,
               threshold: Optional[float] = None) -> Tuple[bool, float]:
        """
        Simple verification API for inference.
        
        Returns:
            is_authentic: bool
            similarity_score: float
        """
        threshold = threshold or self.threshold
        
        with torch.no_grad():
            result = self.forward(video_embedding, claimed_embedding)
            similarity = result['similarity'].item()
            is_authentic = similarity > threshold
            
        return is_authentic, similarity


if __name__ == "__main__":
    # Test identity verification modules
    print("Testing Identity Verification Networks...")
    print("-" * 50)
    
    batch_size = 4
    embedding_dim = 256
    
    # Create sample embeddings
    video_emb = torch.randn(batch_size, embedding_dim)
    claimed_emb = torch.randn(batch_size, embedding_dim)
    
    # Test comparison network
    comparison = GaitComparisonNetwork(embedding_dim=embedding_dim)
    similarity = comparison(video_emb, claimed_emb)
    print(f"Comparison: {video_emb.shape} vs {claimed_emb.shape} -> {similarity.shape}")
    
    # Test triplet network
    triplet = TripletLossNetwork(embedding_dim=embedding_dim)
    anchor = torch.randn(batch_size, embedding_dim)
    positive = torch.randn(batch_size, embedding_dim)
    negative = torch.randn(batch_size, embedding_dim)
    loss, metrics = triplet(anchor, positive, negative)
    print(f"Triplet Loss: {loss.item():.4f}, metrics: {metrics}")
    
    # Test identity verifier
    verifier = IdentityVerifier(embedding_dim=embedding_dim)
    result = verifier(video_emb, claimed_emb)
    print(f"Verifier: prediction={result['prediction']}, similarity={result['similarity']}")
    
    # Count parameters
    print("-" * 50)
    print(f"Comparison params: {sum(p.numel() for p in comparison.parameters()):,}")
    print(f"Triplet params: {sum(p.numel() for p in triplet.parameters()):,}")
    print(f"Verifier params: {sum(p.numel() for p in verifier.parameters()):,}")
