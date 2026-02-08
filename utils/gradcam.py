"""
GradCAM for Gait-Based Deepfake Detection
==========================================
Specialized Grad-CAM implementation for 1D CNN + BiLSTM + Transformer
gait analysis model. Highlights which temporal frames and body joints
the model attends to when making its prediction.

Features:
- CNN layer-level Grad-CAM heatmaps
- Temporal (frame-level) importance scores
- Joint-level importance breakdown
- Visualization utilities for gait sequences

Author: DeepFake Detection Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple


# MediaPipe gait keypoints used in feature extraction (12 points)
GAIT_KEYPOINT_NAMES = [
    'L_Shoulder', 'R_Shoulder',
    'L_Hip', 'R_Hip',
    'L_Knee', 'R_Knee',
    'L_Ankle', 'R_Ankle',
    'L_Heel', 'R_Heel',
    'L_Foot', 'R_Foot'
]

# Joint angle names (6 angles)
JOINT_ANGLE_NAMES = [
    'L_Knee_Angle', 'R_Knee_Angle',
    'L_Hip_Angle', 'R_Hip_Angle',
    'L_Ankle_Angle', 'R_Ankle_Angle'
]

# Feature groups in the 78-dim vector
# 12 keypoints × 3 coords = 36 (normalized_coords)
# 6 joint angles = 6
# 12 keypoints × 3 velocities = 36 (velocities)
FEATURE_GROUPS = {
    'coords': (0, 36),      # 12 keypoints × 3 (x,y,z)
    'angles': (36, 42),      # 6 joint angles
    'velocities': (42, 78),  # 12 keypoints × 3 (vx,vy,vz)
}


class GaitGradCAM:
    """
    Grad-CAM for the gait CNN encoder layers.
    
    Computes spatial attention over the temporal sequence to show
    which frames are most important for the model's decision.
    """
    
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        """
        Args:
            model: GaitDeepfakeDetector model
            target_layer: CNN layer to hook. If None, uses last conv layer
                         of the gait encoder.
        """
        self.model = model
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        # Auto-detect target layer: last conv in gait encoder
        if target_layer is None:
            target_layer = self._find_last_conv_layer(model)
        
        self.target_layer = target_layer
        self._register_hooks()
    
    def _find_last_conv_layer(self, model: nn.Module) -> nn.Module:
        """Find the last Conv1d layer in the gait encoder."""
        last_conv = None
        encoder = getattr(model, 'gait_encoder', model)
        for module in encoder.modules():
            if isinstance(module, nn.Conv1d):
                last_conv = module
        if last_conv is None:
            raise ValueError("No Conv1d layer found in model")
        return last_conv
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self.hooks = [h1, h2]
    
    def remove_hooks(self):
        """Remove registered hooks to free memory."""
        for h in self.hooks:
            h.remove()
        self.hooks = []
    
    def __call__(
        self,
        video_features: torch.Tensor,
        claimed_features: Optional[torch.Tensor] = None,
        target_class: Optional[int] = None,
        mode: str = 'verification'
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap over temporal dimension.
        
        Args:
            video_features: (1, seq_len, input_dim) gait features
            claimed_features: (1, seq_len, input_dim) enrolled identity (for verification)
            target_class: 0=deepfake, 1=authentic. None=use predicted class
            mode: 'verification' or 'classification'
            
        Returns:
            heatmap: (seq_len,) normalized importance per frame
        """
        self.model.eval()
        video_features = video_features.detach().requires_grad_(True)
        
        # Forward pass
        if mode == 'verification' and claimed_features is not None:
            output = self.model(video_features, claimed_features, mode='verification')
            logits = output['verification']['logits']
        else:
            output = self.model(video_features, mode='classification')
            logits = output['logits']
        
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        
        # Backward pass for target class
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class] = 1.0
        logits.backward(gradient=one_hot, retain_graph=True)
        
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Hooks did not fire — check target layer")
        
        # Grad-CAM: global average pool gradients → weight activations
        # activations shape: (B, C, T) from Conv1d
        weights = self.gradients.mean(dim=2, keepdim=True)  # (B, C, 1)
        cam = (weights * self.activations).sum(dim=1)  # (B, T)
        
        # ReLU + normalize
        cam = F.relu(cam)
        cam = cam.squeeze(0)
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.detach().cpu().numpy()


class JointImportanceAnalyzer:
    """
    Analyzes which body joints are most important for the model's decision
    using input-gradient attribution on the 78-dim feature vector.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
    
    def compute_joint_importance(
        self,
        video_features: torch.Tensor,
        claimed_features: Optional[torch.Tensor] = None,
        target_class: Optional[int] = None,
        mode: str = 'verification'
    ) -> Dict[str, np.ndarray]:
        """
        Compute per-joint and per-feature-group importance via input gradients.
        
        Args:
            video_features: (1, seq_len, 78) 
            claimed_features: (1, seq_len, 78) optional
            target_class: 0=deepfake, 1=authentic
            mode: 'verification' or 'classification'
            
        Returns:
            Dictionary with:
                'joint_importance': (12,) importance per gait keypoint
                'angle_importance': (6,) importance per joint angle
                'group_importance': dict with 'coords', 'angles', 'velocities' scores
                'temporal_importance': (seq_len,) frame-level importance
                'feature_heatmap': (seq_len, 78) full spatiotemporal heatmap
        """
        self.model.eval()
        video_features = video_features.clone().detach().requires_grad_(True)
        
        if mode == 'verification' and claimed_features is not None:
            output = self.model(video_features, claimed_features, mode='verification')
            logits = output['verification']['logits']
        else:
            output = self.model(video_features, mode='classification')
            logits = output['logits']
        
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class] = 1.0
        logits.backward(gradient=one_hot)
        
        # Input gradient attribution: |x * grad_x|
        grad = video_features.grad.abs().squeeze(0)  # (T, 78)
        attribution = (video_features.abs().squeeze(0) * grad).detach().cpu().numpy()  # (T, 78)
        
        # Temporal importance: sum over features
        temporal_importance = attribution.sum(axis=1)  # (T,)
        if temporal_importance.max() > 0:
            temporal_importance = temporal_importance / temporal_importance.max()
        
        # Joint importance from coordinate features (0:36) → 12 joints × 3 coords
        coord_attr = attribution[:, FEATURE_GROUPS['coords'][0]:FEATURE_GROUPS['coords'][1]]
        coord_attr = coord_attr.reshape(-1, 12, 3)  # (T, 12, 3)
        joint_importance = coord_attr.sum(axis=(0, 2))  # (12,)
        
        # Also add velocity contributions (42:78) → 12 joints × 3 velocities
        vel_attr = attribution[:, FEATURE_GROUPS['velocities'][0]:FEATURE_GROUPS['velocities'][1]]
        vel_attr = vel_attr.reshape(-1, 12, 3)
        joint_importance += vel_attr.sum(axis=(0, 2))
        
        if joint_importance.max() > 0:
            joint_importance = joint_importance / joint_importance.max()
        
        # Angle importance (36:42) → 6 angles
        angle_attr = attribution[:, FEATURE_GROUPS['angles'][0]:FEATURE_GROUPS['angles'][1]]
        angle_importance = angle_attr.sum(axis=0)  # (6,)
        if angle_importance.max() > 0:
            angle_importance = angle_importance / angle_importance.max()
        
        # Feature group importance
        group_importance = {}
        for name, (start, end) in FEATURE_GROUPS.items():
            group_importance[name] = float(attribution[:, start:end].sum())
        total = sum(group_importance.values())
        if total > 0:
            group_importance = {k: v / total for k, v in group_importance.items()}
        
        return {
            'joint_importance': joint_importance,
            'angle_importance': angle_importance,
            'group_importance': group_importance,
            'temporal_importance': temporal_importance,
            'feature_heatmap': attribution
        }


def plot_temporal_heatmap(
    heatmap: np.ndarray,
    title: str = 'Temporal Importance (Grad-CAM)',
    fps: float = 30.0,
    save_path: Optional[str] = None
):
    """
    Plot frame-level importance as a time-series heatmap.
    
    Args:
        heatmap: (seq_len,) importance per frame 
        title: Plot title
        fps: Video FPS for x-axis labeling
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 3))
    
    time_axis = np.arange(len(heatmap)) / fps
    ax.fill_between(time_axis, 0, heatmap, alpha=0.4, color='red')
    ax.plot(time_axis, heatmap, color='darkred', linewidth=1.5)
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Importance', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved temporal heatmap to {save_path}")
    plt.show()


def plot_joint_importance(
    joint_importance: np.ndarray,
    angle_importance: Optional[np.ndarray] = None,
    title: str = 'Body Joint Importance',
    save_path: Optional[str] = None
):
    """
    Plot per-joint importance as a horizontal bar chart.
    
    Args:
        joint_importance: (12,) importance per gait keypoint
        angle_importance: (6,) importance per joint angle (optional)
        title: Plot title
        save_path: Optional path to save figure
    """
    n_plots = 2 if angle_importance is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    
    # Joint importance
    sorted_idx = np.argsort(joint_importance)
    colors = plt.cm.Reds(joint_importance[sorted_idx] * 0.8 + 0.2)
    axes[0].barh(
        [GAIT_KEYPOINT_NAMES[i] for i in sorted_idx],
        joint_importance[sorted_idx],
        color=colors
    )
    axes[0].set_xlabel('Importance', fontsize=12)
    axes[0].set_title('Gait Keypoint Importance', fontsize=14)
    axes[0].set_xlim(0, 1.05)
    
    # Angle importance
    if angle_importance is not None:
        sorted_idx_a = np.argsort(angle_importance)
        colors_a = plt.cm.Blues(angle_importance[sorted_idx_a] * 0.8 + 0.2)
        axes[1].barh(
            [JOINT_ANGLE_NAMES[i] for i in sorted_idx_a],
            angle_importance[sorted_idx_a],
            color=colors_a
        )
        axes[1].set_xlabel('Importance', fontsize=12)
        axes[1].set_title('Joint Angle Importance', fontsize=14)
        axes[1].set_xlim(0, 1.05)
    
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved joint importance plot to {save_path}")
    plt.show()


def plot_feature_group_importance(
    group_importance: Dict[str, float],
    title: str = 'Feature Group Contribution',
    save_path: Optional[str] = None
):
    """
    Plot pie chart of feature group contributions (coords vs angles vs velocities).
    
    Args:
        group_importance: Dict with 'coords', 'angles', 'velocities' proportions
        title: Plot title
        save_path: Optional path to save figure
    """
    labels = ['Coordinates\n(12×3=36)', 'Joint Angles\n(6)', 'Velocities\n(12×3=36)']
    sizes = [
        group_importance.get('coords', 0),
        group_importance.get('angles', 0),
        group_importance.get('velocities', 0)
    ]
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    explode = (0.05, 0.05, 0.05)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sizes, explode=explode, labels=labels, colors=colors,
           autopct='%1.1f%%', shadow=True, startangle=90,
           textprops={'fontsize': 11})
    ax.set_title(title, fontsize=14, pad=20)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved feature group importance to {save_path}")
    plt.show()


def generate_explainability_report(
    model: nn.Module,
    video_features: torch.Tensor,
    claimed_features: Optional[torch.Tensor] = None,
    prediction_label: str = '',
    save_dir: Optional[str] = None,
    mode: str = 'verification'
) -> Dict[str, np.ndarray]:
    """
    Generate a full explainability report with Grad-CAM and joint analysis.
    
    Args:
        model: Trained GaitDeepfakeDetector
        video_features: (1, seq_len, 78)
        claimed_features: (1, seq_len, 78) for verification mode
        prediction_label: Label for plot titles (e.g. 'AUTHENTIC', 'DEEPFAKE')
        save_dir: Directory to save plots (None = display only)
        mode: 'verification' or 'classification'
        
    Returns:
        Dictionary with all computed importance scores
    """
    import os
    
    # 1. Grad-CAM temporal heatmap
    gradcam = GaitGradCAM(model)
    temporal_heatmap = gradcam(video_features, claimed_features, mode=mode)
    gradcam.remove_hooks()
    
    # 2. Joint importance analysis
    analyzer = JointImportanceAnalyzer(model)
    importance = analyzer.compute_joint_importance(
        video_features, claimed_features, mode=mode
    )
    
    # Build save paths
    temp_path = os.path.join(save_dir, 'temporal_heatmap.png') if save_dir else None
    joint_path = os.path.join(save_dir, 'joint_importance.png') if save_dir else None
    group_path = os.path.join(save_dir, 'feature_groups.png') if save_dir else None
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # 3. Plot temporal heatmap
    plot_temporal_heatmap(
        temporal_heatmap,
        title=f'Temporal Importance — {prediction_label}',
        save_path=temp_path
    )
    
    # 4. Plot joint importance
    plot_joint_importance(
        importance['joint_importance'],
        importance['angle_importance'],
        title=f'Body Joint Importance — {prediction_label}',
        save_path=joint_path
    )
    
    # 5. Plot feature group pie chart
    plot_feature_group_importance(
        importance['group_importance'],
        title=f'Feature Group Contribution — {prediction_label}',
        save_path=group_path
    )
    
    return {
        'temporal_heatmap': temporal_heatmap,
        **importance
    }
