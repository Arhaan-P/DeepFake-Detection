"""Explicit small paired development pilot; no automatic dataset traversal."""

from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
from .evaluation import validate_recordings
from .media import probe_video, decode_frames, sha256_file
from .pose import NativePoseExtractor, save_pose
from .pose_quality import analyze_pose, WindowQualityPolicy, BONES
from .skeleton_contracts import coco_arrays
from .stabilization import stabilize, StabilizationConfig

VARIANTS = ("raw", "vidstab-s10", "vidstab-s15")
MEASURES = (
    "person_detection_fraction",
    "ambiguous_detection_fraction",
    "tracking_failure_fraction",
    "full_body_visibility_fraction",
    "valid_ankle_fraction",
    "confidence_mean",
    "confidence_p10",
    "missing_joint_fraction",
    "low_confidence_fraction",
    "tracking_continuity_fraction",
    "tracking_break_count",
    "jitter_events",
    "mean_bone_length_cv",
    "accepted_window_count",
    "rejected_window_count",
    "border_proximity_fraction",
    "mean_border_dark_fraction",
    "potential_blur_fraction",
    "potential_camera_artifact_events",
)


def compare_reports(reports):
    """Require complete like-for-like triples; equally weight source recordings."""
    groups = {}
    for report in reports:
        take, variant = report["original_take_id"], report["variant_id"]
        if variant not in VARIANTS or variant in groups.setdefault(take, {}):
            raise ValueError("Unknown or duplicate pilot variant")
        groups[take][variant] = report
    if not groups:
        raise ValueError("Empty pilot")
    paired = {}
    for take, triple in groups.items():
        if set(triple) != set(VARIANTS):
            raise ValueError("Incomplete RAW/S10/S15 triple")
        raw = triple["raw"]
        for variant, report in triple.items():
            if (
                report["policy"] != raw["policy"]
                or report["backend_id"].split(":display-rotation=")[0]
                != raw["backend_id"].split(":display-rotation=")[0]
            ):
                raise ValueError("Paired analysis configuration mismatch")
            if report["original_source_sha256"] != raw["original_source_sha256"]:
                raise ValueError("Paired source checksum mismatch")
            if (
                not report["timing_consistent"]
                or report["summary"]["frames"] != raw["summary"]["frames"]
            ):
                raise ValueError("Paired frame/timestamp mismatch")
            if "video_metadata" in raw and "video_metadata" in report:
                a = np.asarray(raw["video_metadata"]["timestamps"])
                b = np.asarray(report["video_metadata"]["timestamps"])
                if len(a) != len(b) or not np.allclose(
                    a - a[0], b - b[0], atol=0.0001, rtol=0
                ):
                    raise ValueError("Paired frame/timestamp mismatch")
        paired[take] = {
            variant: {
                key: float(triple[variant]["summary"][key] - raw["summary"][key])
                if triple[variant]["summary"][key] is not None
                and raw["summary"][key] is not None
                else None
                for key in MEASURES
            }
            for variant in VARIANTS[1:]
        }
    aggregate = {}
    support = {}
    for variant in VARIANTS:
        aggregate[variant] = {}
        support[variant] = {}
        for key in MEASURES:
            values = [
                g[variant]["summary"][key]
                for g in groups.values()
                if g[variant]["summary"][key] is not None
            ]
            aggregate[variant][key] = float(np.mean(values)) if values else None
            support[variant][key] = len(values)
        aggregate[variant]["rejected_window_reasons"] = {
            reason: sum(
                g[variant]["summary"]["rejected_window_reasons"].get(reason, 0)
                for g in groups.values()
            )
            for reason in sorted(
                {
                    r
                    for g in groups.values()
                    for r in g[variant]["summary"]["rejected_window_reasons"]
                }
            )
        }
    return dict(
        schema_version="paired-pilot-v1",
        independent_original_count=len(groups),
        aggregation="equal original-recording means; frames/windows are correlated within recordings",
        per_original={
            t: {v: g[v]["summary"] for v in VARIANTS} for t, g in groups.items()
        },
        paired_deltas_from_raw=paired,
        aggregate=aggregate,
        aggregate_available_original_counts=support,
        recommendation="No automatic winner: inspect coverage/anatomy/borders by view; biometric validation pending",
    )


def write_overlay(video, sequence, output):
    """Three local diagnostic frames, no public upload."""
    import cv2

    xy, confidence, observed = coco_arrays(sequence)
    selected = set(np.linspace(0, len(xy) - 1, min(3, len(xy)), dtype=int).tolist())
    meta = probe_video(video)
    for i, pts, frame in decode_frames(meta):
        if i not in selected:
            continue
        frame = frame.copy()
        valid = observed[i] & (confidence[i] >= 0.3)
        for a, b in BONES:
            if valid[a] and valid[b]:
                cv2.line(
                    frame,
                    tuple(xy[i, a].astype(int)),
                    tuple(xy[i, b].astype(int)),
                    (0, 255, 0),
                    3,
                )
        for j in np.flatnonzero(valid):
            cv2.circle(frame, tuple(xy[i, j].astype(int)), 4, (0, 0, 255), -1)
        cv2.putText(
            frame,
            f"{sequence.variant_id} frame {i} PTS {pts:.4f}",
            (12, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )
        path = Path(output) / f"overlay-{i:05d}.jpg"
        ok, data = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError("Overlay encoding failed")
        with open(path, "xb") as f:
            f.write(data.tobytes())


def run_pilot(entries, detector, output_dir, policy=WindowQualityPolicy()):
    """Entries are explicit (video_path, Recording) pairs; all must be development."""
    records = [r for _, r in entries]
    validate_recordings(records)
    if any(r.variant_id != "raw" or r.split_role != "development" for r in records):
        raise ValueError("Pilot must use development originals")
    # Validate every source before any expensive processing or directory mutation.
    for video, r in entries:
        if sha256_file(video) != r.source_sha256:
            raise ValueError("Source checksum mismatch")
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    reports = []
    for index, (source, record) in enumerate(entries):
        for smoothing in (None, 10, 15):
            variant = "raw" if smoothing is None else f"vidstab-s{smoothing}"
            work = root / f"source-{index:02d}" / variant
            manifest = stabilize(
                source,
                work,
                record,
                StabilizationConfig(
                    enabled=smoothing is not None, smoothing=smoothing or 10
                ),
            )
            video = manifest["output"]["path"]
            sequence = NativePoseExtractor(
                detector, record.original_take_id, variant, collect_diagnostics=True
            ).extract(video)
            save_pose(sequence, work / "pose.npz")
            report = analyze_pose(sequence, policy)
            report.update(
                original_source_sha256=record.source_sha256,
                recording=asdict(record),
                timing_consistent=manifest["status"] == "OK",
                timing_transform=manifest["timing_transform"],
                video_metadata=manifest["output"],
            )
            with open(work / "quality.json", "x") as f:
                json.dump(report, f, indent=2, allow_nan=False)
            write_overlay(video, sequence, work)
            reports.append(report)
            print(
                f"{record.original_take_id} {variant}: {report['summary']}", flush=True
            )
    comparison = compare_reports(reports)
    with open(root / "comparison.json", "x") as f:
        json.dump(comparison, f, indent=2, allow_nan=False)
    return comparison
