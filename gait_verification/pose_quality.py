"""Temporal pilot diagnostics. No interpolation, identity labels or biometric policy."""

from collections import Counter
from dataclasses import asdict, dataclass, fields, replace
import numpy as np
from .contracts import QualityAssessment
from .skeleton_contracts import coco_arrays, InsufficientGait

BONES = (
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


@dataclass(frozen=True)
class WindowQualityPolicy:
    # Engineering starting values; freeze one config across paired variants.
    version: str = "pilot-window-provisional-v1"
    window_seconds: float = 2.0
    min_frames: int = 31
    min_confidence: float = 0.3
    border_pixels: float = 2.0
    min_valid_fraction: float = 0.8
    max_interval_ratio: float = 3.0
    max_joint_step_body_heights: float = 0.15
    max_center_step_body_heights: float = 0.25
    max_scale_step_fraction: float = 0.2
    max_jitter_fraction: float = 0.1
    min_ankle_excursion_body_heights: float = 0.005
    blur_laplacian_variance: float = 20.0

    def validate(self):
        numeric = [v for k, v in asdict(self).items() if k != "version"]
        if not np.isfinite(numeric).all() or min(numeric) < 0:
            raise ValueError("Invalid quality policy")
        if (
            self.window_seconds <= 0
            or self.min_frames < 31
            or self.max_interval_ratio <= 1
        ):
            raise ValueError("Invalid window/time policy")
        if (
            not 0 < self.min_valid_fraction <= 1
            or not 0 <= self.max_jitter_fraction <= 1
        ):
            raise ValueError("Invalid quality fractions")


def slice_pose(sequence, start, stop):
    """Keep source PTS, indices, masks and original lineage exactly."""
    updates = {
        f.name: getattr(sequence, f.name)[start:stop].copy()
        for f in fields(sequence)
        if isinstance(getattr(sequence, f.name), np.ndarray)
    }
    return replace(sequence, **updates)


def _runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def analyze_pose(sequence, policy=WindowQualityPolicy()):
    policy.validate()
    xy, conf, observed = coco_arrays(sequence)
    n = len(xy)
    visible = observed & (conf >= policy.min_confidence)
    inside = (
        (xy[..., 0] > policy.border_pixels)
        & (xy[..., 0] < sequence.width - policy.border_pixels)
        & (xy[..., 1] > policy.border_pixels)
        & (xy[..., 1] < sequence.height - policy.border_pixels)
    )
    detection = sequence.track_ids >= 0
    counts = getattr(sequence, "detection_counts", None)
    person_detected = counts > 0 if counts is not None else detection
    complete = (visible & inside).all(axis=1) & detection
    ankles = (visible[:, 15:17] & inside[:, 15:17]).all(axis=1) & detection
    dt = np.r_[0.0, np.diff(sequence.timestamps)]
    median_dt = float(np.median(dt[1:])) if n > 1 else 0.0
    time_gap = (
        dt > policy.max_interval_ratio * median_dt if median_dt else np.zeros(n, bool)
    )
    track_break = np.r_[False, np.diff(sequence.track_ids) != 0]
    # Joint geometry is only measured on observed pairs. Placeholder zeros never
    # enter bone statistics. Image-plane variation cannot establish camera cause.
    scale = np.ones(n)
    for i in range(n):
        if visible[i].any():
            scale[i] = max(float(np.ptp(xy[i, visible[i], 1])), 1.0)
    center = (xy[:, 11] + xy[:, 12]) / 2
    relative = (xy - center[:, None]) / scale[:, None, None]
    pair = visible[1:] & visible[:-1] & detection[1:, None] & detection[:-1, None]
    pair &= (~(track_break[1:] | time_gap[1:]))[:, None]
    step = np.linalg.norm(np.diff(relative, axis=0), axis=-1)
    jitter = np.r_[
        False, ((step > policy.max_joint_step_body_heights) & pair).any(axis=1)
    ]
    center_step = np.r_[
        0.0, np.linalg.norm(np.diff(center, axis=0), axis=1) / scale[:-1]
    ]
    scale_step = np.r_[0.0, np.abs(np.diff(scale)) / scale[:-1]]
    camera = (center_step > policy.max_center_step_body_heights) | (
        scale_step > policy.max_scale_step_fraction
    )
    camera &= np.r_[False, pair[:, [11, 12]].all(axis=1)]
    bone_cv = {}
    for a, b in BONES:
        lengths = np.linalg.norm(xy[:, a] - xy[:, b], axis=1)
        valid = visible[:, a] & visible[:, b] & detection
        values = lengths[valid]
        bone_cv[f"{a}-{b}"] = (
            float(values.std() / values.mean())
            if len(values) > 1 and values.mean() > 0
            else None
        )
    blur = getattr(sequence, "blur_scores", None)
    borders = getattr(sequence, "border_dark_fraction", None)
    frames = []
    for i in range(n):
        categories = []
        if not complete[i]:
            categories.append("insufficient_pose_information")
        if (observed[i] & ~inside[i]).any():
            categories.append("missing_or_cropped_body_parts")
        if not ankles[i]:
            categories.append("missing_or_low_confidence_ankles")
        if track_break[i] or not detection[i]:
            categories.append("tracking_failure")
        if camera[i]:
            categories.append("potential_camera_motion_artifact")
        if jitter[i]:
            categories.append("excessive_skeleton_step")
        if time_gap[i]:
            categories.append("timestamp_gap")
        if blur is not None and blur[i] < policy.blur_laplacian_variance:
            categories.append("potential_motion_blur")
        if not categories:
            categories.append("ordinary_observed_motion_not_a_walking_diagnosis")
        frames.append(
            dict(
                frame_index=int(sequence.frame_indices[i]),
                timestamp_sec=float(sequence.timestamps[i]),
                track_id=int(sequence.track_ids[i]),
                categories=categories,
                observed_fraction=float(observed[i].mean()),
                valid_joint_fraction=float((visible[i] & inside[i]).mean()),
                valid_ankles=bool(ankles[i]),
                body_height_pixels=float(scale[i]),
                center_step_body_heights=float(center_step[i]),
                scale_step_fraction=float(scale_step[i]),
                blur_score=float(blur[i]) if blur is not None else None,
                border_dark_fraction=float(borders[i]) if borders is not None else None,
            )
        )
    windows, accepted_segments = [], []
    # Nonoverlapping time bins, not nominal-FPS bins; include short tail explicitly.
    bins = np.floor(
        (sequence.timestamps - sequence.timestamps[0]) / policy.window_seconds
    ).astype(int)
    for bin_id in np.unique(bins):
        indices = np.flatnonzero(bins == bin_id)
        start, stop = int(indices[0]), int(indices[-1] + 1)
        reasons = []
        if stop - start < policy.min_frames:
            reasons.append("too_short")
        if complete[start:stop].mean() < policy.min_valid_fraction:
            reasons.append("insufficient_pose_information")
        if ankles[start:stop].mean() < policy.min_valid_fraction:
            reasons.append("missing_or_cropped_ankles")
        if time_gap[start:stop].any():
            reasons.append("timestamp_gap")
        if jitter[start:stop].mean() > policy.max_jitter_fraction:
            reasons.append("excessive_skeleton_steps")
        # No encoder sees an unobserved joint, a track break or a large time gap.
        safe = complete[start:stop].copy()
        safe &= ~(track_break[start:stop] | time_gap[start:stop] | jitter[start:stop])
        runs = [
            (start + int(a), start + int(b))
            for a, b in _runs(safe)
            if b - a >= policy.min_frames
        ]
        if not runs:
            reasons.append("no_contiguous_encoder_segment")
        valid_indices = np.flatnonzero(ankles[start:stop]) + start
        excursion = (
            float(np.std(relative[valid_indices, 15:17], axis=0).max())
            if len(valid_indices)
            else 0.0
        )
        if excursion < policy.min_ankle_excursion_body_heights:
            reasons.append("insufficient_motion_evidence")
        if not reasons:
            accepted_segments.extend(runs)
        windows.append(
            dict(
                window_id=int(bin_id),
                start=start,
                stop=stop,
                first_source_frame=int(sequence.frame_indices[start]),
                last_source_frame=int(sequence.frame_indices[stop - 1]),
                start_pts=float(sequence.timestamps[start]),
                end_pts=float(sequence.timestamps[stop - 1]),
                accepted=not reasons,
                rejection_reasons=reasons,
                valid_fraction=float(complete[start:stop].mean()),
                ankle_fraction=float(ankles[start:stop].mean()),
                tracking_break_count=int(track_break[start + 1 : stop].sum()),
                jitter_events=int(jitter[start:stop].sum()),
                camera_artifact_events=int(camera[start:stop].sum()),
                ankle_excursion_body_heights=excursion,
                encoder_segments=[list(r) for r in runs] if not reasons else [],
            )
        )
    video_reasons = []
    if len(set(sequence.track_ids[detection].tolist())) > 1:
        # Never average potentially different walkers after uncertain association.
        video_reasons.append("ambiguous_track_identity")
        accepted_segments = []
    valid_conf = conf[observed]
    summary = dict(
        frames=n,
        pts_span_sec=float(sequence.timestamps[-1] - sequence.timestamps[0]),
        person_detection_fraction=float(person_detected.mean()),
        ambiguous_detection_fraction=float((counts > 1).mean()) if counts is not None else None,
        tracking_failure_fraction=float((~detection).mean()),
        full_body_visibility_fraction=float(complete.mean()),
        valid_ankle_fraction=float(ankles.mean()),
        missing_joint_fraction=float((~observed).mean()),
        low_confidence_fraction=float((observed & ~visible).mean()),
        confidence_mean=float(valid_conf.mean()) if valid_conf.size else None,
        confidence_p10=float(np.quantile(valid_conf, 0.1)) if valid_conf.size else None,
        confidence_min=float(valid_conf.min()) if valid_conf.size else None,
        tracking_break_count=int(track_break.sum()),
        tracking_continuity_fraction=float((~track_break & detection).mean()),
        jitter_events=int(jitter.sum()),
        potential_camera_artifact_events=int(camera.sum()),
        border_proximity_fraction=float((observed & ~inside).any(axis=1).mean()),
        bone_length_cv=bone_cv,
        mean_bone_length_cv=float(
            np.mean([v for v in bone_cv.values() if v is not None])
        )
        if any(v is not None for v in bone_cv.values())
        else None,
        median_frame_interval_sec=median_dt,
        max_frame_interval_sec=float(dt.max()),
        timestamp_gap_count=int(time_gap.sum()),
        accepted_window_count=sum(w["accepted"] for w in windows),
        rejected_window_count=sum(not w["accepted"] for w in windows),
        accepted_encoder_segment_count=len(accepted_segments),
        rejected_window_reasons=dict(
            Counter(r for w in windows for r in w["rejection_reasons"])
        ),
        mean_border_dark_fraction=float(borders.mean())
        if borders is not None
        else None,
        potential_blur_fraction=float((blur < policy.blur_laplacian_variance).mean())
        if blur is not None
        else None,
        usable=bool(accepted_segments),
        video_rejection_reasons=video_reasons,
    )
    return dict(
        schema_version="pilot-pose-quality-v1",
        policy=asdict(policy),
        original_take_id=sequence.original_take_id,
        source_sha256=sequence.source_sha256,
        backend_id=sequence.backend_id,
        variant_id=sequence.variant_id,
        summary=summary,
        frames=frames,
        windows=windows,
        accepted_windows=[w["window_id"] for w in windows if w["accepted"]],
        rejected_windows=[w["window_id"] for w in windows if not w["accepted"]],
        accepted_segments=[list(r) for r in accepted_segments],
        limitations=[
            "COCO17 has ankles, no heel/toe observations",
            "Camera, blur and dark-border flags are proxies requiring visual QA",
            "Pose motion is not proof of gait or biometric accuracy",
        ],
    )


def assess_window_quality(sequence, policy=WindowQualityPolicy()):
    report = analyze_pose(sequence, policy)
    s = report["summary"]
    return QualityAssessment(
        s["usable"],
        ()
        if s["usable"]
        else tuple(s["video_rejection_reasons"] + list(s["rejected_window_reasons"])),
        s["full_body_visibility_fraction"],
        policy.version,
    )


class WindowedEncoder:
    """Opt-in V2 walk aggregation over fully observed, unchanged temporal runs."""

    def __init__(self, encoder, policy=WindowQualityPolicy()):
        policy.validate()
        if policy.min_frames < getattr(encoder, "min_frames", 31):
            raise ValueError("Window policy is shorter than encoder minimum")
        self.encoder, self.policy = encoder, policy
        self.metadata = encoder.metadata
        self.embedding_dim = encoder.embedding_dim

    def encode(self, sequence):
        import hashlib
        import json
        from .contracts import GaitEmbedding

        report = analyze_pose(sequence, self.policy)
        segments = report["accepted_segments"]
        if not segments:
            raise InsufficientGait("No accepted gait windows")
        embeddings = [
            self.encoder.encode(slice_pose(sequence, a, b)) for a, b in segments
        ]
        if len({e.model_id for e in embeddings}) != 1:
            raise ValueError("Mixed window recipes")
        recipe = hashlib.sha256(
            json.dumps(asdict(self.policy), sort_keys=True).encode()
        ).hexdigest()[:16]
        # Equal weights per accepted segment are a declared engineering recipe.
        result = GaitEmbedding(
            np.mean([e.normalized() for e in embeddings], axis=0),
            embeddings[0].model_id + ":windows=" + recipe,
            sequence.original_take_id,
            sequence.source_sha256,
        )
        self.embedding_dim = len(result.values)
        return replace(result, values=result.normalized())
