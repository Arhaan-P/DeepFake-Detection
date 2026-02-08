# Models package
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
