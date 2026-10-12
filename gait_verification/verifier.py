"""One claimed identity per call; no identification or deepfake inference."""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from .contracts import Status, VerificationResult, valid_claim
from .enrollments import prototype, validate_templates
from .skeleton_contracts import InsufficientGait, assess_quality


@dataclass(frozen=True)
class VerificationPolicy:
    model_id: str
    threshold: Optional[float] = None
    version: str = "uncalibrated"
    calibration_manifest_sha256: Optional[str] = None
    test_only: bool = False
    aggregation: str = "quality_weighted_centroid"

    def ready(self, allow_test_policy=False):
        return (
            self.threshold is not None
            and np.isfinite(self.threshold)
            and -1 <= self.threshold <= 1
            and self.aggregation == "quality_weighted_centroid"
            and (
                (self.test_only and allow_test_policy)
                or (not self.test_only and bool(self.calibration_manifest_sha256))
            )
        )


class ClaimVerifier:
    def __init__(
        self, store, encoder=None, extractor=None, policy=None, allow_test_policy=False,
        quality_assessor=assess_quality,
    ):
        self.store, self.encoder, self.extractor = store, encoder, extractor
        self.policy = policy
        self.allow_test_policy = allow_test_policy
        self.quality_assessor = quality_assessor

    def _score(self, claim, probe, templates, quality):
        if not quality.usable:
            return VerificationResult(claim, Status.INSUFFICIENT_GAIT)
        validate_templates(claim, templates)
        if any(
            t.embedding.original_take_id == probe.original_take_id
            or t.embedding.source_sha256 == probe.source_sha256
            for t in templates
        ):
            return VerificationResult(claim, Status.INVALID_INPUT)
        if self.policy is None or not self.policy.ready(self.allow_test_policy):
            return VerificationResult(claim, Status.POLICY_NOT_READY)
        if probe.model_id != self.policy.model_id or any(
            t.embedding.model_id != probe.model_id for t in templates
        ):
            return VerificationResult(claim, Status.POLICY_NOT_READY)
        q, p = probe.normalized(), prototype(templates)
        if q.shape != p.shape:
            raise ValueError("Embedding dimension mismatch")
        score = float(np.clip(np.dot(q, p), -1, 1))
        return VerificationResult(
            claim, Status.OK, bool(score >= self.policy.threshold)
        )

    def verify_embedding(self, probe, claimed_subject_id, quality):
        claim = claimed_subject_id
        if not valid_claim(claim):
            return VerificationResult("", Status.INVALID_INPUT)
        try:
            templates = self.store.get_claim(claim)
            if not templates:
                return VerificationResult(claim, Status.CLAIM_NOT_ENROLLED)
            return self._score(claim, probe, templates, quality)
        except Exception:
            return VerificationResult(claim, Status.PROCESSING_ERROR)

    def verify(self, video_path, claimed_subject_id):
        claim = claimed_subject_id
        if not valid_claim(claim):
            return VerificationResult("", Status.INVALID_INPUT)
        try:
            templates = self.store.get_claim(claim)  # exactly one lookup
            if not templates:
                return VerificationResult(claim, Status.CLAIM_NOT_ENROLLED)
            if self.extractor is None or self.encoder is None:
                return VerificationResult(claim, Status.POLICY_NOT_READY)
            sequence = self.extractor.extract(video_path)
            quality = self.quality_assessor(sequence)
            if not quality.usable:
                return VerificationResult(claim, Status.INSUFFICIENT_GAIT)
            if self.encoder.metadata.readiness != "READY":
                return VerificationResult(claim, Status.POLICY_NOT_READY)
            probe = self.encoder.encode(sequence)
            return self._score(claim, probe, templates, quality)
        except InsufficientGait:
            return VerificationResult(claim, Status.INSUFFICIENT_GAIT)
        except (ValueError, FileNotFoundError):
            return VerificationResult(claim, Status.INVALID_INPUT)
        except Exception:
            return VerificationResult(claim, Status.PROCESSING_ERROR)
