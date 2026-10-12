"""One claimed identity per call; no identification or deepfake inference.

Status precedence (first match wins, every non-OK status has verdict=null):
  1 INVALID_INPUT       claim syntax, missing/unreadable/corrupt video, or the probe
                        shares an original recording with a claimed reference
  2 PROCESSING_ERROR    enrollment store unreadable/corrupt
  3 CLAIM_NOT_ENROLLED  no active reference for the claim (or none with the probe's recipe)
  4 INSUFFICIENT_GAIT   pose extraction/quality/encoder abstains
  5 PROCESSING_ERROR    any other runtime or model failure
  6 POLICY_NOT_READY    scoring succeeded but no authorized policy covers this recipe
  7 OK                  only with an authorized policy (TEST_ONLY only when explicitly allowed)
A missing policy is evaluated last, so it can never hide invalid input or unusable gait.
`*_detailed` additionally returns a RESTRICTED diagnostics dict (raw score, reasons);
it must never be forwarded to an untrusted caller.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
from .contracts import Status, VerificationResult, valid_claim
from .enrollments import validate_templates
from .media import InvalidMedia
from .scoring import BASELINE, score_claim
from .skeleton_contracts import InsufficientGait, assess_quality


@dataclass(frozen=True)
class VerificationPolicy:
    model_id: str
    threshold: Optional[float] = None
    version: str = "uncalibrated"
    calibration_manifest_sha256: Optional[str] = None
    test_only: bool = False
    aggregation: str = BASELINE

    def ready(self, allow_test_policy=False):
        return bool(
            self.threshold is not None
            and np.isfinite(self.threshold)
            and -1 <= self.threshold <= 1
            and self.aggregation == BASELINE
            and (
                (self.test_only and allow_test_policy)
                or (not self.test_only and bool(self.calibration_manifest_sha256))
            )
        )


def _origin_ids(embedding, recording=None):
    ids = {embedding.original_take_id, embedding.source_sha256}
    if recording is not None:
        ids |= {
            recording.original_take_id,
            recording.source_sha256,
            recording.asset_id,
            recording.parent_asset_id,
        }
    return ids - {None, ""}


class ClaimVerifier:
    def __init__(
        self,
        store,
        encoder=None,
        extractor=None,
        policy=None,
        allow_test_policy=False,
        quality_assessor=assess_quality,
        scoring_strategy=BASELINE,
    ):
        self.store, self.encoder, self.extractor = store, encoder, extractor
        self.policy = policy
        self.allow_test_policy = allow_test_policy
        self.quality_assessor = quality_assessor
        self.scoring_strategy = scoring_strategy

    @staticmethod
    def _out(claim, status, diag, reason, verdict=None):
        diag = dict(diag, status=status.value, reason=reason)
        return VerificationResult(claim, status, verdict), diag

    def _score(self, claim, probe, templates, quality, lineage_ids, diag):
        if not quality.usable:
            diag["quality_reasons"] = list(quality.reasons)
            return self._out(claim, Status.INSUFFICIENT_GAIT, diag, "QUALITY_GATE")
        probe_ids = _origin_ids(probe) | set(lineage_ids)
        if any(probe_ids & _origin_ids(t.embedding, t.recording) for t in templates):
            return self._out(claim, Status.INVALID_INPUT, diag, "SHARED_ORIGINAL")
        diag["model_id"] = probe.model_id
        compatible = tuple(
            t for t in templates if t.embedding.model_id == probe.model_id
        )
        if not compatible:
            return self._out(claim, Status.CLAIM_NOT_ENROLLED, diag, "RECIPE_MISMATCH")
        validate_templates(claim, compatible)
        scored = score_claim(probe, compatible, self.scoring_strategy)
        diag.update(
            score=scored.score,
            strategy=scored.strategy,
            reference_count=scored.reference_count,
        )
        policy = self.policy
        if (
            policy is None
            or not policy.ready(self.allow_test_policy)
            or policy.model_id != probe.model_id
            or policy.aggregation != scored.strategy
            or (
                not policy.test_only
                and self.encoder is not None
                and self.encoder.metadata.readiness != "READY"
            )
        ):
            return self._out(
                claim, Status.POLICY_NOT_READY, diag, "NO_AUTHORIZED_POLICY"
            )
        diag.update(policy_version=policy.version, test_only_policy=policy.test_only)
        return self._out(
            claim, Status.OK, diag, "SCORED", bool(scored.score >= policy.threshold)
        )

    def _lookup(self, claim):
        if not valid_claim(claim):
            return None, self._out("", Status.INVALID_INPUT, {}, "INVALID_CLAIM")
        try:
            templates = self.store.get_claim(claim)  # exactly one lookup
        except Exception:
            return None, self._out(claim, Status.PROCESSING_ERROR, {}, "STORE_ERROR")
        if not templates:
            return None, self._out(
                claim, Status.CLAIM_NOT_ENROLLED, {}, "NO_REFERENCES"
            )
        return templates, None

    def verify_embedding_detailed(
        self, probe, claimed_subject_id, quality, lineage_ids=()
    ):
        claim = claimed_subject_id
        templates, early = self._lookup(claim)
        if early:
            return early
        try:
            return self._score(claim, probe, templates, quality, lineage_ids, {})
        except Exception:
            return self._out(claim, Status.PROCESSING_ERROR, {}, "SCORING_ERROR")

    def verify_embedding(self, probe, claimed_subject_id, quality, lineage_ids=()):
        return self.verify_embedding_detailed(
            probe, claimed_subject_id, quality, lineage_ids
        )[0]

    def verify_detailed(self, video_path, claimed_subject_id, lineage_ids=()):
        claim = claimed_subject_id
        if valid_claim(claim) and not Path(str(video_path)).is_file():
            return self._out(claim, Status.INVALID_INPUT, {}, "VIDEO_NOT_FOUND")
        templates, early = self._lookup(claim)
        if early:
            return early
        if self.extractor is None or self.encoder is None:
            return self._out(
                claim, Status.POLICY_NOT_READY, {}, "PIPELINE_NOT_INSTALLED"
            )
        diag = {}
        try:
            sequence = self.extractor.extract(video_path)
            quality = self.quality_assessor(sequence)
            if not quality.usable:
                return self._score(claim, None, templates, quality, (), diag)
            probe = self.encoder.encode(sequence)
            diag["quality_policy"] = quality.policy_version
            return self._score(claim, probe, templates, quality, lineage_ids, diag)
        except (InvalidMedia, FileNotFoundError):
            return self._out(claim, Status.INVALID_INPUT, diag, "INVALID_MEDIA")
        except InsufficientGait as exc:
            diag["quality_reasons"] = str(exc).split(",")
            return self._out(claim, Status.INSUFFICIENT_GAIT, diag, "ENCODER_ABSTAINED")
        except Exception:
            return self._out(claim, Status.PROCESSING_ERROR, diag, "RUNTIME_ERROR")

    def verify(self, video_path, claimed_subject_id, lineage_ids=()):
        return self.verify_detailed(video_path, claimed_subject_id, lineage_ids)[0]
