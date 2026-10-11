"""Anatomical mappings, not conversions from the historical 78-D descriptor.

COCO definition: open-mmlab/mmpose configs/_base_/datasets/coco.py.
GaitPT custom order: official datasets/augmentation.py CustomRepresentation.
The repeated nose is SOURCE behavior, not an invented neck joint.
"""

import numpy as np
from .contracts import PoseSequence, QualityAssessment

COCO17 = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
MEDIAPIPE_TO_COCO = (0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
GAITPT_ORDER = (0, 1, 3, 0, 2, 4, 5, 7, 9, 6, 8, 10, 11, 13, 15, 12, 14, 16)


class InsufficientGait(ValueError):
    pass


def coco_arrays(sequence: PoseSequence):
    sequence.validate()
    indices = (
        MEDIAPIPE_TO_COCO if sequence.layout == "mediapipe33" else tuple(range(17))
    )
    return (
        sequence.xy[:, indices].copy(),
        sequence.confidence[:, indices],
        sequence.observed[:, indices],
    )


def assess_quality(sequence, min_frames=31, min_confidence=0.3, border_pixels=2):
    """Conservative provisional gate, NOT a validated walking/biometric classifier.

    Reject gaps rather than interpolating unobserved measurements. Motion/blur and
    detailed foot QA still require the capture pilot. Native COCO lacks toes.
    """
    xy, conf, observed = coco_arrays(sequence)
    good = np.all(observed & (conf >= min_confidence), axis=1)
    in_frame = (
        (xy[..., 0] > border_pixels)
        & (xy[..., 0] < sequence.width - border_pixels)
        & (xy[..., 1] > border_pixels)
        & (xy[..., 1] < sequence.height - border_pixels)
    )
    good &= in_frame.all(axis=1)
    reasons = []
    if len(xy) < min_frames:
        reasons.append("too_short")
    if not good.all():
        reasons.append("missing_low_confidence_or_cropped_joints")
    if len(set(sequence.track_ids.tolist())) != 1 or sequence.track_ids[0] < 0:
        reasons.append("ambiguous_or_missing_track")
    # Detect timestamp gaps relative to measured cadence, not nominal FPS.
    dt = np.diff(sequence.timestamps)
    if len(dt) and dt.max() > 3 * np.median(dt):
        reasons.append("timestamp_gap")
    return QualityAssessment(not reasons, tuple(reasons), float(good.mean()))


def gaitpt_grew_inputs(sequence):
    """Exact official two-center-crop GREW evaluation geometry (30 frames).

    Pixels / IMAGE HEIGHT for both axes, no hip centering. Confidence discarded
    after validation. Preserve source PTS in cache; native sampling uses indices.
    Official GREW loader excludes lengths <=30; mirror/flip defaults are off.
    """
    quality = assess_quality(sequence)
    if not quality.usable:
        raise InsufficientGait(",".join(quality.reasons))
    xy, _, _ = coco_arrays(sequence)
    n = len(xy)
    halves = (xy[: max(n // 2, 30)], xy[min(n // 2, n - 1 - 30) : n - 1])
    crops = []
    for half in halves:
        start = int(len(half) / 2 - 30 / 2)
        crops.append(half[start : start + 30, GAITPT_ORDER] / sequence.height)
    return np.stack(crops).astype(np.float32)
