"""Editable pilot intake sheet -> existing capture-registry-v1, never a new registry.

Suggested paths/IDs are convenience placeholders until explicitly confirmed.
Measured facts are returned separately; no source or intake file is modified.
"""

from collections import Counter
from copy import deepcopy

from .capture_registry import CaptureRegistry, _contained, load_capture_schema
from .media import decode_frames, metadata_dict, probe_video

VIEWS = ("left45", "right45", "toward", "away")
SHEET_SCHEMA = "two-subject-pilot-intake-v1"
EVIDENCE = "TWO_SUBJECT_PILOT_DIAGNOSTIC"
NOTICE = "TWO-PERSON EXPLORATORY PILOT — NOT A VALIDATED BIOMETRIC PERFORMANCE RESULT"


def intake_template(subjects):
    from .contracts import valid_claim

    if (
        len(subjects) != 2
        or len(set(subjects)) != 2
        or not all(map(valid_claim, subjects))
    ):
        raise ValueError("Exactly two distinct pseudonymous subjects required")
    rows = []
    for subject in subjects:
        for view in VIEWS:
            for number in (1, 2):
                take = f"{subject}_{view}_set{number}"
                capture = dict.fromkeys(load_capture_schema()["required_fields"])
                capture.update(
                    asset_id=take,
                    original_take_id=take,
                    subject_id=subject,
                    view_label=view,
                    source_file_uri=f"originals/{subject}/{view}/set{number}.mp4",
                    variant_id="raw",
                    parent_asset_id=None,
                    split_role="development",
                )
                rows.append(
                    dict(
                        set_number=number,
                        capture=capture,
                        operator_confirmed=False,
                        consent_confirmed=False,
                        independent_original_confirmed=False,
                    )
                )
    return dict(
        schema_version=SHEET_SCHEMA,
        evidence_class=EVIDENCE,
        subjects=list(subjects),
        rows=rows,
    )


def validate_intake(document, data_root):
    """Measure explicit paths, fully decode, validate assertions; all-or-nothing import.

    Returns (restricted report, validated registry or None, measured editable sheet).
    A missing hash may be measured; a supplied contradictory hash is always refused.
    Filesystem times are never used as capture evidence. No locked role is opened.
    """
    if (
        document.get("schema_version") != SHEET_SCHEMA
        or document.get("evidence_class") != EVIDENCE
    ):
        raise ValueError("Unsupported pilot intake")
    subjects = document["subjects"]
    expected = {(s, v, n) for s in subjects for v in VIEWS for n in (1, 2)}
    intake_template(subjects)  # validate subject syntax/count without trusting folders
    measured_sheet = deepcopy(document)
    rows, captures, missing, errors, hashes = [], [], [], [], {}
    matrix = []
    for item in measured_sheet["rows"]:
        capture = item["capture"]
        key = (
            capture.get("subject_id"),
            capture.get("view_label"),
            item.get("set_number"),
        )
        matrix.append(key)
        entry = dict(
            asset_id=capture.get("asset_id"),
            subject_id=key[0],
            view=key[1],
            set_number=key[2],
            source_file_uri=capture.get("source_file_uri"),
            issues=[],
            decoder_health="NOT_RUN",
            registration_status="BLOCKED",
        )
        rows.append(entry)
        issues = entry["issues"]
        if key not in expected:
            issues.append("UNEXPECTED_MATRIX_ROW")
        if capture.get("split_role") != "development":
            issues.append("PILOT_REQUIRES_DEVELOPMENT; protected media not opened")
            continue
        for name in (
            "operator_confirmed",
            "consent_confirmed",
            "independent_original_confirmed",
        ):
            if item.get(name) is not True:
                issues.append(f"REQUIRES_EXPLICIT_{name}")
        for name in (
            "session_id",
            "date",
            "consent_ref",
            "viewing_angle_definition",
            "walking_direction",
            "camera_motion_mode",
        ):
            if not capture.get(name):
                issues.append(f"MISSING_METADATA:{name}")
        if capture.get("quality_review_status") != "accepted":
            issues.append("CAPTURE_REVIEW_REQUIRED_OR_RECAPTURE")
        try:
            path = _contained(data_root, capture["source_file_uri"])
            if not path.is_file():
                missing.append(capture["source_file_uri"])
                raise ValueError("MISSING_ORIGINAL")
            meta = probe_video(path)
            facts = metadata_dict(meta)
            facts.pop("path")
            entry["measured"] = facts
            hashes.setdefault(meta.sha256, []).append(capture["asset_id"])
            if capture.get("sha256") not in (None, "", meta.sha256):
                issues.append("SOURCE_SHA256_MISMATCH")
            # Read every actual frame, not just container headers.
            count = sum(1 for _ in decode_frames(meta))
            entry.update(
                decoder_health="OK",
                frame_count=count,
                storage_bytes=path.stat().st_size,
            )
            capture.update(
                sha256=meta.sha256,
                duration_sec=meta.duration_sec,
                resolution=f"{meta.width}x{meta.height}",
                nominal_fps=meta.nominal_fps,
                actual_frame_timestamps_ref="ffprobe:source-presentation-timestamps",
            )
            # All existing capture schema checks still apply, including timestamp TZ.
            CaptureRegistry(
                dict(schema_version="capture-registry-v1", assets=[capture])
            )
            captures.append(capture)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issues.append(str(exc))
        if not issues:
            entry["registration_status"] = "VALIDATED_PENDING_REGISTRATION"
    counts = Counter(matrix)
    if set(matrix) != expected or any(n != 1 for n in counts.values()):
        errors.append(
            "Expected exactly 2 subjects × 4 views × 2 independently confirmed takes"
        )
    duplicates = [ids for ids in hashes.values() if len(ids) > 1]
    if duplicates:
        errors.append("DUPLICATE_SOURCE_BYTES")
    registry = None
    if not errors and not any(r["issues"] for r in rows):
        try:
            registry = CaptureRegistry(
                dict(schema_version="capture-registry-v1", assets=captures)
            )
        except ValueError as exc:
            errors.append(str(exc))
    report = dict(
        schema_version="two-subject-intake-report-v1",
        evidence_class=EVIDENCE,
        notice=NOTICE,
        ok=registry is not None,
        expected_originals=16,
        found_originals=sum("measured" in r for r in rows),
        missing_originals=missing,
        missing_matrix_rows=[list(k) for k in sorted(expected - set(matrix))],
        duplicate_sources=duplicates,
        errors=errors,
        recordings=rows,
        restricted=True,
        consent_verified_by_program=False,
        scientific_suitability="NOT_ASSESSED",
    )
    return report, registry, measured_sheet
