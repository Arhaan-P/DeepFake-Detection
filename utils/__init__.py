"""Utility modules for gait feature extraction and data loading.

Modules:
    pose_extraction: MediaPipe-based 78-dim gait feature extractor.
    data_loader: PyTorch Dataset with balanced pair sampling.
    logger: Timestamped file + console logging.
    gradcam: GradCAM explainability for gait encoder.
    visualization: Plotting utilities for training curves and metrics.
"""

from utils.pose_extraction import GaitFeatureExtractor, extract_features_from_dataset
from utils.data_loader import GaitDataset, create_data_loaders

__all__ = [
    'GaitFeatureExtractor',
    'extract_features_from_dataset',
    'GaitDataset',
    'create_data_loaders'
]
