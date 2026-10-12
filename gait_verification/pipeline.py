"""Video -> native pose cache -> window quality -> frozen embedding -> enrollment/query.

Reuses the Phase 0/1A/1B components; nothing here scores a gallery. Query and enrollment
share the exact same extractor, quality policy and windowed encoder, which is what makes
their model IDs comparable.
"""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from .contracts import Recording, valid_claim
from .enrollments import SCHEMA, EnrollmentTemplate
from .media import probe_video, sha256_file
from .pose import NativePoseExtractor, load_pose, save_pose
from .pose_quality import WindowQualityPolicy, analyze_pose, assess_window_quality
from .skeleton_contracts import InsufficientGait
from .verifier import ClaimVerifier

AGGREGATION = "equal-weight-mean-of-accepted-contiguous-segments-v1"
_TAKE = re.compile(r"[A-Za-z0-9_.-]{1,128}")


class EnrollmentRefused(ValueError):
    """The recording is not an eligible enrollment; carries a stable reason code."""


POSE_ARTIFACT_SCHEMA = "pose-artifact-v1"


class CacheIntegrityError(ValueError):
    """A pose cache entry is unsealed, altered, stale or unreadable. Never trusted."""

    def __init__(self, reason):
        super().__init__(f"Pose cache is not trustworthy: {reason}")
        self.reason = reason


def extractor_config():
    """Pose-extraction settings that change the artifact; sealed into every sidecar."""
    e = NativePoseExtractor(None, collect_diagnostics=True)
    return {
        "extractor": "native-pose-v1",
        "max_center_jump": e.max_center_jump,
        "collect_diagnostics": e.collect_diagnostics,
    }


def pose_cache_key(source_sha256, backend_id, variant_id, take):
    """Artifact name: bytes + pose recipe + variant + take (unchanged since Phase 1C)."""
    text = f"{source_sha256}:{backend_id}:{variant_id}:{take}"
    return hashlib.sha256(text.encode()).hexdigest()[:24]


def check_pose_artifact(npz, expect):
    """Verify a pose cache entry against `expect` (source/backend/variant/take/config).

    Returns the sidecar record. Raises CacheIntegrityError with a stable reason code.
    Read-only; the caller decides whether to quarantine.
    """
    side = Path(npz).with_suffix(".json")
    try:
        record = json.loads(side.read_text())
    except (OSError, ValueError):
        raise CacheIntegrityError("SIDECAR_MISSING_OR_UNREADABLE") from None
    if (
        not isinstance(record, dict)
        or record.get("schema_version") != POSE_ARTIFACT_SCHEMA
    ):
        raise CacheIntegrityError("SIDECAR_SCHEMA")
    for name, value in expect.items():
        if record.get(name) != value:
            raise CacheIntegrityError(f"STALE_{name.upper()}")
    checksum = record.get("npz_sha256")
    if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise CacheIntegrityError("SIDECAR_CHECKSUM")
    if not isinstance(record.get("frames"), int) or record["frames"] < 1:
        raise CacheIntegrityError("SIDECAR_FRAMES")
    try:
        if sha256_file(npz) != record["npz_sha256"]:
            raise CacheIntegrityError("ARTIFACT_CHECKSUM_MISMATCH")
    except OSError:
        raise CacheIntegrityError("ARTIFACT_UNREADABLE") from None
    return record


def _publish_exclusive(write, final):
    """Write to a private temp file, fsync, then hard-link: all-or-nothing, no overwrite."""
    tmp = final.with_name(f".{final.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            write(f)
            f.flush()
            os.fsync(f.fileno())
        os.link(tmp, final)
    finally:
        tmp.unlink(missing_ok=True)


class CachedPoseExtractor:
    """Real pose extraction with an immutable, integrity-sealed cache.

    Entries are keyed by bytes + pose recipe + variant + take. Each is an NPZ plus a
    sidecar recording the NPZ's SHA-256, the source SHA-256 and the extractor settings;
    both are published atomically (NPZ first). A hit is reused only if the sidecar
    matches and the NPZ bytes still hash to it. `require_integrity` refuses entries
    with no sidecar (legacy Phase 1C-1D caches); otherwise those keep the old checks.
    Diagnostics are always collected so a transient bystander frame cannot silently
    abort or silently pass; the Phase 1A window analysis decides what is usable.
    """

    def __init__(
        self,
        detector,
        cache_dir=None,
        original_take_id=None,
        variant_id="raw",
        require_integrity=False,
    ):
        self.detector, self.original_take_id, self.variant_id = (
            detector,
            original_take_id,
            variant_id,
        )
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.require_integrity = require_integrity

    def extract(self, video_path):
        meta = probe_video(video_path)
        take = self.original_take_id or meta.sha256
        key = pose_cache_key(
            meta.sha256, self.detector.backend_id, self.variant_id, take
        )
        cached = self.cache_dir / f"{key}.npz" if self.cache_dir else None
        if cached is not None and cached.exists():
            if self.require_integrity or cached.with_suffix(".json").exists():
                check_pose_artifact(
                    cached,
                    dict(
                        source_sha256=meta.sha256,
                        backend_id=self.detector.backend_id,
                        variant_id=self.variant_id,
                        original_take_id=take,
                        extractor=extractor_config(),
                    ),
                )
            try:
                sequence = load_pose(cached)
            except (OSError, ValueError, KeyError, EOFError) as exc:
                raise CacheIntegrityError("ARTIFACT_UNREADABLE") from exc
            if (
                sequence.source_sha256 != meta.sha256
                or sequence.original_take_id != take
                or list(sequence.timestamps) != list(meta.timestamps)
                or not sequence.backend_id.startswith(self.detector.backend_id)
            ):
                raise CacheIntegrityError("CACHE_DOES_NOT_MATCH_SOURCE")
            return sequence
        sequence = NativePoseExtractor(
            self.detector, take, self.variant_id, collect_diagnostics=True
        ).extract(video_path)
        if cached is not None:
            self._store(sequence, cached, key, meta, take)
        return sequence

    def _store(self, sequence, cached, key, meta, take):
        self.cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        tmp = cached.with_name(f".{cached.name}.{os.getpid()}.part")
        try:
            save_pose(sequence, tmp)  # exclusive create
            with tmp.open("rb") as f:
                os.fsync(f.fileno())
            digest = sha256_file(tmp)
            record = dict(
                schema_version=POSE_ARTIFACT_SCHEMA,
                key=key,
                source_sha256=meta.sha256,
                backend_id=self.detector.backend_id,
                variant_id=self.variant_id,
                original_take_id=take,
                extractor=extractor_config(),
                npz_sha256=digest,
                npz_bytes=tmp.stat().st_size,
                frames=len(sequence.timestamps),
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            os.chmod(tmp, 0o600)
            os.link(tmp, cached)  # FileExistsError rather than replacing
            data = json.dumps(record, indent=2, sort_keys=True).encode()
            _publish_exclusive(lambda f: f.write(data), cached.with_suffix(".json"))
        finally:
            tmp.unlink(missing_ok=True)


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
    if sequence.variant_id != "raw":
        raise EnrollmentRefused("enrollment_requires_original_pose")
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
    expected_source_sha256=None,
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
    initial_sha = expected_source_sha256 or sha256_file(video_path)
    if sha256_file(video_path) != initial_sha:
        raise EnrollmentRefused("source_missing_or_checksum_mismatch")
    sequence = extractor.extract(video_path)
    if sequence.source_sha256 != initial_sha:
        raise EnrollmentRefused("pose_source_mismatch")
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
        if sha256_file(video_path) != initial_sha:
            raise EnrollmentRefused("source_changed_during_enrollment")
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
                expected_source_sha256=asset.sha256,
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


def verify_registered(
    registry,
    video_path,
    claimed_subject_id,
    make_verifier,
    *,
    data_root=None,
    split=None,
):
    """Managed-study query: lineage comes from the capture registry, never from flags.

    The file is identified by its ACTUAL bytes. Unregistered or unreadable files fail
    closed (INVALID_INPUT) and are never treated as new originals. `make_verifier`
    receives only (original_take_id, variant_id); the registry's subject ID is never
    passed to the claim-only verifier.
    """
    from .contracts import Status, VerificationResult
    from .dataset_processing import final_test_denied, source_issue
    from .splits import validate_split

    claim = claimed_subject_id if valid_claim(claimed_subject_id) else ""

    def refused(reason, status=Status.INVALID_INPUT):
        return VerificationResult(claim, status), {
            "status": status.value,
            "reason": reason,
        }

    if not claim:
        return refused("INVALID_CLAIM")
    if data_root is None or split is None:
        return refused("MANAGED_CONTEXT_REQUIRED")
    try:
        validate_split(split, registry.subjects())
        if split["assignments"].get(claim) == "locked_final":
            return refused("LOCKED_FINAL_ACCESS_DENIED")
        # Check registered paths before hashing or initializing any model. A byte
        # alias is checked again after resolution; filenames never establish identity.
        supplied = Path(video_path).resolve()
        for candidate in registry.assets():
            if registry.path_of(candidate, data_root) == supplied and final_test_denied(
                candidate, split
            ):
                return refused("LOCKED_FINAL_ACCESS_DENIED")
    except (OSError, ValueError, KeyError, TypeError):
        return refused("MANAGED_CONTEXT_INVALID")
    try:
        asset = registry.resolve_file(video_path)
    except OSError:
        asset, reason = None, "VIDEO_NOT_FOUND"
    else:
        reason = "UNREGISTERED_PROVENANCE"
    if asset is None:
        return refused(reason)
    if final_test_denied(asset, split):
        return refused("LOCKED_FINAL_ACCESS_DENIED")
    if reason := source_issue(registry, asset, data_root):
        return refused(reason)
    try:
        verifier = make_verifier(asset.original_take_id, asset.variant_id)
        result, diag = verifier.verify_detailed(
            video_path, claimed_subject_id, registry.lineage_ids(asset.asset_id)
        )
    except Exception:
        return refused("PIPELINE_SETUP_ERROR", Status.PROCESSING_ERROR)
    try:
        changed = (
            source_issue(registry, asset, data_root)
            or sha256_file(video_path) != asset.sha256
        )
    except OSError:
        changed = True
    if changed:
        return refused("SOURCE_CHANGED_DURING_VERIFICATION")
    return result, dict(diag, query_asset_id=asset.asset_id)
