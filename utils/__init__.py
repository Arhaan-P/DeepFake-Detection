"""Utility modules for gait feature extraction and data loading.

Modules:
    pose_extraction: MediaPipe-based 78-dim gait feature extractor.
    data_loader: PyTorch Dataset with balanced pair sampling.
    logger: Timestamped file + console logging.
    gradcam: GradCAM explainability for gait encoder.
    visualization: Plotting utilities for training curves and metrics.

Future-work modules (numpy-only unless noted):
    pose_backends: pluggable pose estimators in a canonical 33-slot layout (cv2).
    pose_quality: pose-signal quality metrics (jitter, bone-length stability...).
    gait_cycle: gait events, cadence, rhythm, cycle normalisation, DTW.
    gait_features: frozen baseline 78-D math + new biomechanical families.
    gait_descriptors: clip loading and descriptor building for experiments.
    keypoint_augment: keypoint-level augmentation and robustness perturbations.
    verification_metrics: ROC/EER, calibration, operating points, statistics.

Package-level names are resolved lazily so that importing a numpy-only
submodule does not pull in MediaPipe.
"""

import importlib

_EXPORTS = {
    "GaitDataset": "utils.data_loader",
    "create_data_loaders": "utils.data_loader",
    "GaitFeatureExtractor": "utils.pose_extraction",
    "extract_features_from_dataset": "utils.pose_extraction",
}

__all__ = [
    "GaitDataset",
    "GaitFeatureExtractor",
    "create_data_loaders",
    "extract_features_from_dataset",
]


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module 'utils' has no attribute {name!r}")
