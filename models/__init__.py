"""Gait-based deepfake detection model components.

Modules:
    gait_encoder: 1D CNN spatial encoder for per-frame gait features.
    temporal_model: BiLSTM + Transformer dual-path temporal encoder.
    identity_verifier: Siamese comparison network with verification head.
    full_pipeline: End-to-end GaitDeepfakeDetector assembly.
"""

from models.gait_encoder import GaitEncoder, MultiScaleGaitEncoder
from models.temporal_model import (
    BiLSTMEncoder, 
    TransformerEncoder, 
    DualPathTemporalModel,
    TemporalAttentionPool
)
from models.identity_verifier import (
    IdentityVerifier,
    GaitComparisonNetwork,
    TripletLossNetwork,
    ContrastiveLossNetwork
)
from models.full_pipeline import (
    GaitDeepfakeDetector,
    GaitDeepfakeDetectorWithTriplet,
    create_model
)

__all__ = [
    'GaitEncoder',
    'MultiScaleGaitEncoder',
    'BiLSTMEncoder',
    'TransformerEncoder',
    'DualPathTemporalModel',
    'TemporalAttentionPool',
    'IdentityVerifier',
    'GaitComparisonNetwork',
    'TripletLossNetwork',
    'ContrastiveLossNetwork',
    'GaitDeepfakeDetector',
    'GaitDeepfakeDetectorWithTriplet',
    'create_model'
]
