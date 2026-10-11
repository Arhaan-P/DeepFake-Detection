"""Explicit V2 contracts. Public results never contain biometric scores or other IDs."""

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional, Protocol
import re
import numpy as np


class Status(str, Enum):
    OK = "OK"
    CLAIM_NOT_ENROLLED = "CLAIM_NOT_ENROLLED"
    INSUFFICIENT_GAIT = "INSUFFICIENT_GAIT"
    INVALID_INPUT = "INVALID_INPUT"
    PROCESSING_ERROR = "PROCESSING_ERROR"
    POLICY_NOT_READY = "POLICY_NOT_READY"


def valid_claim(value):
    return (
        isinstance(value, str)
        and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value) is not None
    )


@dataclass(frozen=True)
class VerificationResult:
    claimed_subject_id: str
    status: Status
    verdict: Optional[bool] = None
    schema_version: str = "gait-verification-v1"

    def __post_init__(self):
        if self.status == Status.OK:
            if type(self.verdict) is not bool:
                raise ValueError("OK requires a Boolean verdict")
        elif self.verdict is not None:
            raise ValueError("Abstention/error requires verdict=null")

    def public_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Recording:
    asset_id: str
    original_take_id: str
    subject_id: str
    session_id: str
    view: str
    recorded_at: str
    source_sha256: str
    split_role: str
    variant_id: str = "raw"
    parent_asset_id: Optional[str] = None


@dataclass(frozen=True)
class VideoMetadata:
    path: str
    sha256: str
    width: int
    height: int
    nominal_fps: str
    average_fps: str
    time_base: str
    timestamps: tuple[float, ...]
    codec: str
    rotation: int = 0


@dataclass(frozen=True)
class PoseSequence:
    xy: np.ndarray  # T,J,2; image pixels, x right/y down (unrotated stream)
    confidence: np.ndarray  # T,J; [0,1], zero for missing
    observed: np.ndarray  # T,J; missing coordinates must be finite placeholders
    timestamps: np.ndarray  # actual source PTS seconds, no fabricated FPS clock
    layout: str
    width: int
    height: int
    original_take_id: str
    source_sha256: str
    backend_id: str
    frame_indices: np.ndarray
    track_ids: np.ndarray
    bboxes: np.ndarray  # T,4; xyxy, zero for missing
    variant_id: str = "raw"

    def validate(self):
        counts = {"coco17": 17, "halpe26": 26, "mediapipe33": 33}
        if self.layout not in counts:
            raise ValueError("Unsupported pose layout")
        if self.xy.ndim != 3 or self.xy.shape[1:] != (counts[self.layout], 2):
            raise ValueError("Wrong joint count/coordinate shape")
        n = len(self.xy)
        if n == 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("Empty sequence or invalid image dimensions")
        if (
            self.confidence.shape != self.xy.shape[:2]
            or self.observed.shape != self.xy.shape[:2]
        ):
            raise ValueError("Confidence/observed shape mismatch")
        if self.observed.dtype != np.bool_:
            raise ValueError("Observed mask must be Boolean")
        for value in (self.xy, self.confidence, self.timestamps, self.bboxes):
            if not np.isfinite(value).all():
                raise ValueError("NaN/Inf in pose cache")
        if ((self.confidence < 0) | (self.confidence > 1)).any():
            raise ValueError("Confidence outside [0,1]")
        if self.timestamps.shape != (n,) or np.any(np.diff(self.timestamps) <= 0):
            raise ValueError("PTS must strictly increase")
        if self.frame_indices.shape != (n,) or np.any(np.diff(self.frame_indices) <= 0):
            raise ValueError("Frame indices must strictly increase")
        if self.track_ids.shape != (n,) or self.bboxes.shape != (n, 4):
            raise ValueError("Track/bbox shape mismatch")
        if not self.original_take_id or not self.source_sha256 or not self.backend_id:
            raise ValueError("Missing pose provenance")


@dataclass(frozen=True)
class QualityAssessment:
    usable: bool
    reasons: tuple[str, ...] = ()
    valid_fraction: float = 0.0
    policy_version: str = "pilot-provisional-v1"


@dataclass(frozen=True)
class ModelMetadata:
    model_id: str
    repository: str
    revision: str
    checkpoint_sha256: str
    checkpoint_bytes: int
    training_dataset: str
    input_contract: str
    license: str
    readiness: str = "BLOCKED"


@dataclass(frozen=True)
class GaitEmbedding:
    values: np.ndarray
    model_id: str  # must include checkpoint + preprocessing recipe fingerprint
    original_take_id: str
    source_sha256: str

    def normalized(self):
        z = np.asarray(self.values, dtype=np.float64)
        if z.ndim != 1 or z.size == 0 or not np.isfinite(z).all():
            raise ValueError("Invalid embedding")
        norm = np.linalg.norm(z)
        if norm < 1e-12:
            raise ValueError("Zero embedding")
        return (z / norm).astype(np.float32)


class GaitEncoder(Protocol):
    metadata: ModelMetadata
    embedding_dim: Optional[int]

    def encode(self, sequence: PoseSequence) -> GaitEmbedding: ...
