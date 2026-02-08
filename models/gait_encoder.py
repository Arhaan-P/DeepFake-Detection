"""
Gait Encoder - Spatial Feature Extraction
==========================================
CNN-based encoder for extracting spatial features from pose sequences.

This module implements:
1. 1D CNN for temporal convolutions on pose features
2. Feature pyramid for multi-scale analysis
3. Residual connections for better gradient flow

Author: DeepFake Detection Project
"""

import torch
import torch.nn as nn
from typing import Tuple


class ResidualBlock1D(nn.Module):
    """1D Residual block for temporal convolutions."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1):
        super().__init__()
        
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                               padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection
        self.skip = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class GaitEncoder(nn.Module):
    """
    CNN-based encoder for gait feature sequences.
    
    Takes pose sequences and extracts hierarchical spatial features.
    Input: (batch, seq_len, feature_dim)
    Output: (batch, seq_len, hidden_dim)
    """
    
    def __init__(self, 
                 input_dim: int = 78,  # 36 coords + 6 angles + 36 velocities
                 hidden_dims: Tuple[int, ...] = (64, 128, 256),
                 output_dim: int = 256,
                 dropout: float = 0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Initial projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Build residual blocks
        layers = []
        in_channels = hidden_dims[0]
        
        for out_channels in hidden_dims:
            layers.append(ResidualBlock1D(in_channels, out_channels))
            layers.append(nn.Dropout(dropout))
            in_channels = out_channels
        
        self.res_blocks = nn.Sequential(*layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
            
        Returns:
            (batch, seq_len, output_dim)
        """
        # Project input
        x = self.input_proj(x)  # (B, T, hidden_dim)
        
        # Transpose for Conv1D: (B, C, T)
        x = x.transpose(1, 2)
        
        # Apply residual blocks
        x = self.res_blocks(x)
        
        # Transpose back: (B, T, C)
        x = x.transpose(1, 2)
        
        # Output projection
        x = self.output_proj(x)
        
        return x


class MultiScaleGaitEncoder(nn.Module):
    """
    Multi-scale CNN encoder using parallel convolutions with different kernel sizes.
    Captures both fine-grained and coarse temporal patterns.
    """
    
    def __init__(self,
                 input_dim: int = 78,
                 hidden_dim: int = 128,
                 output_dim: int = 256,
                 kernel_sizes: Tuple[int, ...] = (3, 5, 7),
                 dropout: float = 0.3):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Multi-scale convolution branches
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            branch = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, k, padding=k//2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, k, padding=k//2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )
            self.branches.append(branch)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden_dim * len(kernel_sizes), output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.output_dim = output_dim
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
            
        Returns:
            (batch, seq_len, output_dim)
        """
        # Project and transpose
        x = self.input_proj(x)  # (B, T, H)
        x = x.transpose(1, 2)   # (B, H, T)
        
        # Multi-scale processing
        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(x))
        
        # Concatenate branches
        x = torch.cat(branch_outputs, dim=1)  # (B, H*num_branches, T)
        
        # Fuse
        x = self.fusion(x)  # (B, output_dim, T)
        
        return x.transpose(1, 2)  # (B, T, output_dim)


if __name__ == "__main__":
    # Test the encoders
    print("Testing GaitEncoder...")
    
    # Create sample input: batch=4, seq_len=60, features=78
    x = torch.randn(4, 60, 78)
    
    # Test basic encoder
    encoder = GaitEncoder(input_dim=78, output_dim=256)
    out = encoder(x)
    print(f"GaitEncoder: {x.shape} -> {out.shape}")
    
    # Test multi-scale encoder
    ms_encoder = MultiScaleGaitEncoder(input_dim=78, output_dim=256)
    out = ms_encoder(x)
    print(f"MultiScaleGaitEncoder: {x.shape} -> {out.shape}")
    
    # Count parameters
    params = sum(p.numel() for p in encoder.parameters())
    print(f"GaitEncoder parameters: {params:,}")
    
    params = sum(p.numel() for p in ms_encoder.parameters())
    print(f"MultiScaleGaitEncoder parameters: {params:,}")
