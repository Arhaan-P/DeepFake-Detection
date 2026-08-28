"""Gait-based deepfake detection model components.

Modules:
    gait_encoder: 1D CNN spatial encoder for per-frame gait features.
    temporal_model: BiLSTM + Transformer dual-path temporal encoder.
    identity_verifier: Siamese comparison network with verification head.
    full_pipeline: End-to-end GaitDeepfakeDetector assembly.
"""

from models.full_pipeline import (
    GaitDeepfakeDetector,
    GaitDeepfakeDetectorWithTriplet,
    create_model,
)
from models.gait_encoder import GaitEncoder, MultiScaleGaitEncoder
from models.identity_verifier import (
    ContrastiveLossNetwork,
    GaitComparisonNetwork,
    IdentityVerifier,
    TripletLossNetwork,
)
from models.temporal_model import (
    BiLSTMEncoder,
    DualPathTemporalModel,
    TemporalAttentionPool,
    TransformerEncoder,
)

__all__ = [
    "BiLSTMEncoder",
    "ContrastiveLossNetwork",
    "DualPathTemporalModel",
    "GaitComparisonNetwork",
    "GaitDeepfakeDetector",
    "GaitDeepfakeDetectorWithTriplet",
    "GaitEncoder",
    "IdentityVerifier",
    "MultiScaleGaitEncoder",
    "TemporalAttentionPool",
    "TransformerEncoder",
    "TripletLossNetwork",
    "create_model",
]
