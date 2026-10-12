"""Video -> native pose cache -> window quality -> frozen embedding -> enrollment/query.

Reuses the Phase 0/1A/1B components; nothing here scores a gallery. Query and enrollment
share the exact same extractor, quality policy and windowed encoder, which is what makes
their model IDs comparable.
"""

from dataclasses import asdict
from datetime import datetime
import hashlib
from pathlib import Path
import re
from .contracts import Recording, valid_claim
from .enrollments import SCHEMA, EnrollmentTemplate
from .media import probe_video
from .pose import NativePoseExtractor, load_pose, save_pose
from .pose_quality import WindowQualityPolicy, analyze_pose, assess_window_quality
from .skeleton_contracts import InsufficientGait
from .verifier import ClaimVerifier

AGGREGATION = "equal-weight-mean-of-accepted-contiguous-segments-v1"
_TAKE = re.compile(r"[A-Za-z0-9_.-]{1,128}")


class EnrollmentRefused(ValueError):
    """The recording is not an eligible enrollment; carries a stable reason code."""


class CachedPoseExtractor:
    """Real pose extraction with an immutable cache keyed by bytes + pose recipe + take.

    Diagnostics are always collected so a transient bystander frame cannot silently
    abort or silently pass; the Phase 1A window analysis decides what is usable.
    """

    def __init__(
        self, detector, cache_dir=None, original_take_id=None, variant_id="raw"
    ):
        self.detector, self.original_take_id, self.variant_id = (
            detector,
            original_take_id,
            variant_id,
        )
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def extract(self, video_path):
        meta = probe_video(video_path)
        take = self.original_take_id or meta.sha256
        key = hashlib.sha256(
            f"{meta.sha256}:{self.detector.backend_id}:{self.variant_id}:{take}".encode()
        ).hexdigest()[:24]
        cached = self.cache_dir / f"{key}.npz" if self.cache_dir else None
        if cached is not None and cached.exists():
            sequence = load_pose(cached)
            if (
                sequence.source_sha256 != meta.sha256
                or sequence.original_take_id != take
                or list(sequence.timestamps) != list(meta.timestamps)
                or not sequence.backend_id.startswith(self.detector.backend_id)
            ):
                raise ValueError("Pose cache does not match its source recording")
            return sequence
        sequence = NativePoseExtractor(
            self.detector, take, self.variant_id, collect_diagnostics=True
        ).extract(video_path)
        if cached is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            save_pose(sequence, cached)  # exclusive create; caches are immutable
        return sequence


def query_verifier(store, encoder, extractor, policy=None, **kwargs):
    """Claim verifier using the same window-quality recipe as enrollment."""
    policy = policy or WindowQualityPolicy()
    return ClaimVerifier(
        store,
        encoder,
        extractor,
        quality_assessor=lambda s: assess_window_quality(s, policy),
        **kwargs,
    )


def lineage_from_manifest(manifest):
    """Original IDs declared by a stabilization manifest, for derivative queries."""
    recording = manifest.get("recording", {})
    ids = {
        recording.get("original_take_id"),
        recording.get("source_sha256"),
        recording.get("asset_id"),
        manifest.get("input", {}).get("sha256"),
    }
    ids.discard(None)
    if not ids:
        raise ValueError("Derivative manifest declares no original lineage")
    return frozenset(ids)


def _quality_summary(report):
    s = report["summary"]
    keys = (
        "frames",
        "pts_span_sec",
        "full_body_visibility_fraction",
        "valid_ankle_fraction",
        "tracking_break_count",
        "accepted_window_count",
        "rejected_window_count",
        "accepted_encoder_segment_count",
        "rejected_window_reasons",
        "video_rejection_reasons",
    )
    return {k: s[k] for k in keys} | {"policy_version": report["policy"]["version"]}


def check_eligibility(sequence, policy):
    """Raise EnrollmentRefused unless Phase 1A quality finds usable gait on one track."""
    report = analyze_pose(sequence, policy)
    if not report["summary"]["usable"]:
        s = report["summary"]
        reasons = s["video_rejection_reasons"] + sorted(s["rejected_window_reasons"])
        raise EnrollmentRefused(",".join(reasons) or "no_accepted_gait_segments")
    return report


def validate_enrollment_metadata(
    subject_id, take_id, session_id, recorded_at, consent_ref
):
    """Required metadata is never guessed or defaulted."""
    if not valid_claim(subject_id):
        raise EnrollmentRefused("invalid_subject_id")
    if not _TAKE.fullmatch(take_id or ""):
        raise EnrollmentRefused("invalid_take_id")
    if not session_id or not consent_ref:
        raise EnrollmentRefused("missing_session_or_consent_reference")
    try:
        datetime.fromisoformat(recorded_at)
    except (TypeError, ValueError) as exc:
        raise EnrollmentRefused("invalid_recorded_at") from exc


def build_template(
    sequence,
    report,
    embedding,
    encoder,
    policy,
    *,
    subject_id,
    take_id,
    session_id,
    view,
    recorded_at,
    consent_ref,
    split_role="development",
    asset_id=None,
):
    if (
        embedding.original_take_id != take_id
        or embedding.source_sha256 != sequence.source_sha256
    ):
        raise EnrollmentRefused("embedding_lineage_mismatch")
    recording = Recording(
        asset_id=asset_id or take_id,
        original_take_id=take_id,
        subject_id=subject_id,
        session_id=session_id,
        view=view,
        recorded_at=recorded_at,
        source_sha256=sequence.source_sha256,
        split_role=split_role,
    )
    provenance = {
        "enrollment_schema": SCHEMA,
        "pose_backend_id": sequence.backend_id,
        "pose_variant_id": sequence.variant_id,
        "gait_model": asdict(encoder.metadata),
        "window_policy": asdict(policy),
        "window_aggregation": AGGREGATION,
        "quality": _quality_summary(report),
        "consent_ref": consent_ref,
        "independence_unit": "original_take_id; windows/derivatives are not references",
    }
    enrollment_id = (
        "enr-"
        + hashlib.sha256(
            f"{subject_id}|{take_id}|{embedding.model_id}".encode()
        ).hexdigest()[:20]
    )
    return EnrollmentTemplate(
        subject_id, embedding, recording, 1.0, enrollment_id, provenance
    )


def enroll_original(
    video_path,
    extractor,
    encoders,
    store,
    policy,
    *,
    subject_id,
    take_id,
    session_id,
    view,
    recorded_at,
    consent_ref,
    split_role="development",
    asset_id=None,
    skip_enrolled=False,
):
    """Extract once, then store one INDEPENDENT reference per encoder recipe.

    All recipes are written in ONE atomic store update, so an interruption cannot
    leave a partial recipe group. Refusals raise before anything is written; store
    errors (duplicate take, cross-subject reuse, ...) propagate and are never repaired.
    `skip_enrolled` makes resumption idempotent: recipes already stored for this take
    are left untouched and not re-added.
    """
    validate_enrollment_metadata(
        subject_id, take_id, session_id, recorded_at, consent_ref
    )
    sequence = extractor.extract(video_path)
    report = check_eligibility(sequence, policy)
    done = set()
    if skip_enrolled:
        done = {
            t.embedding.model_id
            for t in store.get_claim(subject_id)
            if t.recording.original_take_id == take_id
        }
    templates = []
    for encoder in encoders:
        try:
            embedding = encoder.encode(sequence)
        except InsufficientGait as exc:
            raise EnrollmentRefused(str(exc)) from exc
        if embedding.model_id in done:
            continue
        templates.append(
            build_template(
                sequence,
                report,
                embedding,
                encoder,
                policy,
                subject_id=subject_id,
                take_id=take_id,
                session_id=session_id,
                view=view,
                recorded_at=recorded_at,
                consent_ref=consent_ref,
                split_role=split_role,
                asset_id=asset_id,
            )
        )
    if templates:
        store.add_references(subject_id, templates)
    return templates


def enroll_registered(
    registry,
    split,
    asset_ids,
    data_root,
    extractor_factory,
    encoders,
    store,
    policy,
    roles=("development",),
    authorization=None,
):
    """Manifest-driven batch enrollment of EXPLICITLY selected registered originals.

    Metadata (take, subject, session, view, time, consent reference, split role) comes
    from the validated registry and split, never from flags or filenames. Every asset
    gets a result row; one failure never stops or hides the others. Re-running is
    idempotent (`ALREADY_ENROLLED`). `extractor_factory(original_take_id, variant_id)`
    receives lineage only - no subject identity.
    """
    from .capture_registry import ProvenanceError
    from .media import sha256_file
    from .splits import require_roles

    require_roles(split, roles, authorization)
    results = []
    for asset_id in sorted(set(asset_ids)):
        row = {"asset_id": asset_id}
        results.append(row)
        try:
            asset = registry.asset(asset_id)
            role = split["assignments"].get(asset.subject_id)
            if not asset.is_original:
                raise EnrollmentRefused("derivative_not_enrollable")
            if role not in roles:
                raise EnrollmentRefused("subject_role_not_selected")
            recording = registry.recording(asset_id, role)
            path = registry.path_of(asset, data_root)
            if not path.is_file() or sha256_file(path) != asset.sha256:
                raise EnrollmentRefused("source_missing_or_checksum_mismatch")
            made = enroll_original(
                path,
                extractor_factory(asset.original_take_id, asset.variant_id),
                encoders,
                store,
                policy,
                subject_id=asset.subject_id,
                take_id=asset.original_take_id,
                session_id=asset.session_id,
                view=asset.view,
                recorded_at=asset.recorded_at,
                consent_ref=asset.consent_ref,
                split_role=recording.split_role,
                asset_id=asset_id,
                skip_enrolled=True,
            )
            row.update(
                status="ENROLLED" if made else "ALREADY_ENROLLED",
                enrollment_ids=[t.enrollment_id for t in made],
                model_ids=[t.embedding.model_id for t in made],
            )
        except EnrollmentRefused as exc:
            row.update(status="REFUSED", reason=str(exc))
        except (ProvenanceError, ValueError, OSError) as exc:
            row.update(status="REFUSED", reason=f"{type(exc).__name__}: {exc}")
        except Exception as exc:  # reported, never silently dropped
            row.update(status="ERROR", reason=type(exc).__name__)
    return results


def verify_registered(registry, video_path, claimed_subject_id, make_verifier):
    """Managed-study query: lineage comes from the capture registry, never from flags.

    The file is identified by its ACTUAL bytes. Unregistered or unreadable files fail
    closed (INVALID_INPUT) and are never treated as new originals. `make_verifier`
    receives only (original_take_id, variant_id); the registry's subject ID is never
    passed to the claim-only verifier.
    """
    from .contracts import Status, VerificationResult

    claim = claimed_subject_id if valid_claim(claimed_subject_id) else ""
    try:
        asset = registry.resolve_file(video_path)
    except OSError:
        asset, reason = None, "VIDEO_NOT_FOUND"
    else:
        reason = "UNREGISTERED_PROVENANCE"
    if asset is None:
        diag = {"status": Status.INVALID_INPUT.value, "reason": reason}
        return VerificationResult(claim, Status.INVALID_INPUT), diag
    verifier = make_verifier(asset.original_take_id, asset.variant_id)
    result, diag = verifier.verify_detailed(
        video_path, claimed_subject_id, registry.lineage_ids(asset.asset_id)
    )
    return result, dict(diag, query_asset_id=asset.asset_id)
