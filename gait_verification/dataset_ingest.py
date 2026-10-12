"""Capture-data validation report, one row per registered recording.

Four layers are kept apart and never merged into one verdict:
  media_validation              program-verified: bytes, ffprobe timing, declared-vs-measured
  operator_metadata             operator-supplied fields: complete or listed as missing
  pose_engineering_gate         filled by processing (provisional Phase 1A gate)
  scientific_capture_suitability always NOT_ASSESSED: it needs real pilot evidence

Nothing is inferred from a filename, and a consent reference is never treated as consent.
The report holds no paths, consent references or embeddings.
"""

from collections import Counter
import re
from .capture_registry import _placeholder, load_capture_schema

# PROVISIONAL tolerances for flagging declared-vs-measured differences, not validated limits.
DURATION_REL_TOL = 0.02
FPS_REL_TOL = 0.02
_REGISTRY_ONLY = {"asset_id", "variant_id", "parent_asset_id", "split_role"}
_REGISTRY_ONLY |= {"transformation_manifest_ref", "transformation_manifest_sha256"}
_NOT_METADATA = {"sha256", "source_file_uri", "actual_frame_timestamps_ref"}


def diagnose_rows(rows):
    """Best-effort structured issues for a registry document that failed to load.

    Read-only and never raises. The registry itself still fails closed; this only tells
    the operator WHICH assets collide.
    """
    issues = []
    rows = [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []

    def group(key, code, only_raw=False):
        seen = {}
        for r in rows:
            value = r.get(key)
            if (
                isinstance(value, str)
                and value
                and (not only_raw or r.get("variant_id") == "raw")
            ):
                seen.setdefault(value, []).append(r.get("asset_id"))
        for ids in seen.values():
            if len(ids) > 1:
                issues.append(dict(code=code, asset_ids=sorted(map(str, ids))))

    group("asset_id", "DUPLICATE_ASSET_ID")
    group("sha256", "DUPLICATE_BYTES")
    group("original_take_id", "DUPLICATE_ORIGINAL_TAKE", only_raw=True)
    by_sha = {}
    for r in rows:
        if r.get("variant_id") == "raw" and r.get("sha256"):
            by_sha.setdefault(r["sha256"], set()).add(r.get("subject_id"))
    for sha, subjects in by_sha.items():
        if len(subjects) > 1:
            ids = sorted(str(r.get("asset_id")) for r in rows if r.get("sha256") == sha)
            issues.append(
                dict(code="CONFLICTING_SUBJECTS_FOR_SAME_BYTES", asset_ids=ids)
            )
    by_take = {}
    for r in rows:
        if r.get("variant_id") == "raw" and r.get("original_take_id"):
            by_take.setdefault(r["original_take_id"], set()).add(r.get("subject_id"))
    for take, subjects in by_take.items():
        if len(subjects) > 1:
            issues.append(
                dict(code="CONFLICTING_SUBJECTS_FOR_SAME_TAKE", original_take_id=take)
            )
    for r in rows:
        for key, code in (
            ("session_id", "PLACEHOLDER_SESSION"),
            ("consent_ref", "PLACEHOLDER_OR_MISSING_CONSENT_REF"),
        ):
            value = r.get(key)
            if r.get("variant_id") == "raw" and (
                not isinstance(value, str) or not value or _placeholder(value)
            ):
                issues.append(dict(code=code, asset_ids=[str(r.get("asset_id"))]))
    return sorted(
        issues,
        key=lambda i: (i["code"], str(i.get("asset_ids", i.get("original_take_id")))),
    )


def _fraction(text):
    try:
        a, _, b = str(text).partition("/")
        return float(a) / float(b) if b else float(a)
    except (ValueError, ZeroDivisionError):
        return None


def _declared_vs_measured(capture, measured):
    """Differences between operator-declared values and ffprobe facts (flag, not verdict)."""
    out = {}
    declared = capture.get("duration_sec")
    actual = measured.get("duration_sec") or measured.get("pts_span_sec")
    if declared is not None and actual:
        rel = abs(declared - actual) / actual
        out["duration_sec"] = dict(
            declared=declared, measured=actual, differs=rel > DURATION_REL_TOL
        )
    declared = capture.get("nominal_fps")
    actual = _fraction(measured.get("average_fps"))
    if isinstance(declared, (int, float)) and actual:
        out["fps"] = dict(
            declared=declared,
            measured_average=actual,
            differs=abs(declared - actual) / actual > FPS_REL_TOL,
        )
    declared = capture.get("resolution")
    if declared is not None:
        m = re.fullmatch(r"\s*(\d+)\s*[xX×]\s*(\d+)\s*", str(declared))
        shown = (measured["display_width"], measured["display_height"])
        out["resolution"] = (
            dict(
                declared=str(declared),
                measured=f"{shown[0]}x{shown[1]}",
                differs=sorted(map(int, m.groups())) != sorted(shown),
            )
            if m
            else dict(
                declared=str(declared),
                measured=f"{shown[0]}x{shown[1]}",
                differs=None,
                note="declared value not parseable as WxH",
            )
        )
    return out


def ingest_report(registry, data_root, measure=True, schema=None):
    """Per-recording validation: bytes, ffprobe facts, metadata completeness, lineage."""
    schema = schema or load_capture_schema()
    verification = registry.verify_files(data_root, measure)
    required = [f for f in schema["required_fields"] if f not in _NOT_METADATA]
    children = {}
    for a in registry.assets():
        if not a.is_original:
            children.setdefault(registry.original(a.asset_id).asset_id, []).append(
                a.variant_id
            )
    rows = []
    for a in registry.assets():
        entry = verification["files"][a.asset_id]
        issues = [
            e.split(": ", 1)[1]
            for e in verification["errors"]
            if e.startswith(f"{a.asset_id}: ")
        ]
        measured = entry.get("measured")
        media = dict(
            file_present=entry["exists"],
            sha256_verified=entry["sha256_verified"],
            measured=measured,
            readable=None if measured is None and not measure else measured is not None,
            declared_vs_measured=_declared_vs_measured(a.capture, measured)
            if measured and a.is_original
            else {},
            issues=issues,
        )
        if a.is_original:
            missing = sorted(
                f
                for f in required
                if a.capture.get(f) is None and f not in _REGISTRY_ONLY
            )
            metadata = dict(
                missing_fields=missing,
                complete=not missing,
                consent_reference_present=bool(a.consent_ref),
                consent_verified_by_program=False,
                note="Missing optional capture fields are listed, not guessed; the registry already refused missing identity/session/time/consent/hash.",
            )
        else:
            metadata = dict(
                inherits_from=a.parent_asset_id,
                transformation_manifest_declared=bool(a.transformation_manifest_ref),
            )
        rows.append(
            dict(
                asset_id=a.asset_id,
                kind="original" if a.is_original else "derivative",
                variant_id=a.variant_id,
                original_take_id=a.original_take_id,
                subject_id=a.subject_id,
                session_id=a.session_id,
                view=a.view,
                parent_asset_id=a.parent_asset_id,
                derivative_variants=sorted(children.get(a.asset_id, [])),
                media_validation=media,
                operator_metadata=metadata,
                pose_engineering_gate=dict(status="NOT_RUN", provisional=True),
                scientific_capture_suitability=dict(
                    status="NOT_ASSESSED",
                    reason="needs real pilot evidence (register SCI-01/SCI-02)",
                ),
            )
        )
    originals = [r for r in rows if r["kind"] == "original"]
    mismatched = Counter(
        k
        for r in originals
        for k, v in r["media_validation"]["declared_vs_measured"].items()
        if v.get("differs")
    )
    return dict(
        schema_version="dataset-ingest-report-v1",
        registry_fingerprint=registry.fingerprint,
        ok=verification["ok"],
        errors=verification["errors"],
        summary=dict(
            registry.summary(),
            originals_missing_metadata=sum(
                not r["operator_metadata"]["complete"] for r in originals
            ),
            declared_vs_measured_differences=dict(sorted(mismatched.items())),
            measured=bool(measure),
        ),
        recordings=rows,
        provisional_tolerances=dict(duration_rel=DURATION_REL_TOL, fps_rel=FPS_REL_TOL),
        restricted=True,
    )
