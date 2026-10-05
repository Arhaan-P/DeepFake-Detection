"""
Gait Feature Families (future work, plan Sections 5-6 / experiments E3-E4)
=========================================================================
Pure-numpy feature math on canonical 33-slot pose sequences (MediaPipe index
space, see utils/pose_backends.py). No MediaPipe import, so it runs anywhere.

Two kinds of family:

  per-frame   (N, d) signals, e.g. coordinates, angles, derivatives, sway
  clip-level  (d,)   gait parameters, e.g. cadence, stride regularity; the
                     descriptor builder broadcasts them over time so the
                     unchanged temporal verifier can compare them

The three BASELINE families (`coords`, `angles`, `velocity`) reproduce
utils/pose_extraction.py exactly -- same 2D angle definition, same padded
first difference on the 60-frame resampled sequence -- and are verified
against it in tests/test_future_work.py. They are deliberately not
"improved": the baseline is frozen (plan Section 2), and every new idea is a
separate family that an ablation adds on top.

New per-frame families use physical time (seconds) and body-scale units
(torso lengths), because the baseline's per-resampled-step velocity silently
mixes gait speed with clip duration.

Author: DeepFake Detection Project
"""

from collections.abc import Sequence
from typing import Callable, Dict, List

import numpy as np

from utils.gait_cycle import GaitCycleAnalysis, analyse_gait, lowpass

# Same 12 landmarks as GaitFeatureExtractor.GAIT_LANDMARKS (kept local so this
# module does not import cv2/mediapipe).
GAIT_LANDMARKS = [11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

L_SH, R_SH, L_HIP, R_HIP = 11, 12, 23, 24
L_KNEE, R_KNEE, L_ANK, R_ANK = 25, 26, 27, 28
L_HEEL, R_HEEL, L_TOE, R_TOE = 29, 30, 31, 32

# Left/right pairs inside the 12-joint gait layout (for mirroring/symmetry).
GAIT_LR_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
# ...and in canonical 33-slot space.
CANONICAL_LR_PAIRS = [
    (1, 4),
    (2, 5),
    (3, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
    (17, 18),
    (19, 20),
    (21, 22),
    (23, 24),
    (25, 26),
    (27, 28),
    (29, 30),
    (31, 32),
]


# ============================================================
# Baseline math (frozen; mirrors utils/pose_extraction.py)
# ============================================================


def resample_sequence(sequence: np.ndarray, length: int) -> np.ndarray:
    """Linear index-space resampling, identical to
    GaitFeatureExtractor._normalize_sequence_length."""
    t = len(sequence)
    if t == length:
        return sequence
    old = np.linspace(0, t - 1, t)
    new = np.linspace(0, t - 1, length)
    flat = sequence.reshape(t, -1)
    out = np.empty((length, flat.shape[1]))
    for i in range(flat.shape[1]):
        out[:, i] = np.interp(new, old, flat[:, i])
    return out.reshape(length, *sequence.shape[1:])


def _angle_2d(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """Angle at p2 (degrees), image-plane, vectorised over leading axes."""
    v1 = p1[..., :2] - p2[..., :2]
    v2 = p3[..., :2] - p2[..., :2]
    cos = np.sum(v1 * v2, axis=-1) / (
        np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-8
    )
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def _angle_3d(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    v1 = p1 - p2
    v2 = p3 - p2
    cos = np.sum(v1 * v2, axis=-1) / (
        np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-8
    )
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def baseline_joint_angles(pose: np.ndarray) -> np.ndarray:
    """(T, 33, 3) -> (T, 6): L/R knee, L/R hip, L/R ankle, in the baseline's
    order (left_knee, right_knee, left_hip, right_hip, left_ankle, right_ankle)."""
    p = pose
    return np.stack(
        [
            _angle_2d(p[:, L_HIP], p[:, L_KNEE], p[:, L_ANK]),
            _angle_2d(p[:, R_HIP], p[:, R_KNEE], p[:, R_ANK]),
            _angle_2d(p[:, L_SH], p[:, L_HIP], p[:, L_KNEE]),
            _angle_2d(p[:, R_SH], p[:, R_HIP], p[:, R_KNEE]),
            _angle_2d(p[:, L_KNEE], p[:, L_ANK], p[:, L_TOE]),
            _angle_2d(p[:, R_KNEE], p[:, R_ANK], p[:, R_TOE]),
        ],
        axis=1,
    )


def hip_centred_gait_coords(pose: np.ndarray) -> np.ndarray:
    """(T, 33, 3) -> (T, 12, 3) gait landmarks minus the mid-hip."""
    hip = (pose[:, L_HIP, :3] + pose[:, R_HIP, :3]) / 2
    return pose[:, GAIT_LANDMARKS, :3] - hip[:, None, :]


def padded_diff(x: np.ndarray) -> np.ndarray:
    """First difference padded by repeating the last row (baseline velocity)."""
    d = np.diff(x, axis=0)
    return np.concatenate([d, d[-1:]], axis=0)


def baseline_descriptor(pose60: np.ndarray) -> np.ndarray:
    """The frozen 78-D descriptor on an already-resampled (60, 33, 3) pose."""
    coords = hip_centred_gait_coords(pose60)
    t = len(coords)
    return np.concatenate(
        [
            coords.reshape(t, -1),
            baseline_joint_angles(pose60),
            padded_diff(coords).reshape(t, -1),
        ],
        axis=1,
    )


# ============================================================
# Helpers for the new families
# ============================================================


def torso_length(pose: np.ndarray) -> float:
    """Median shoulder-mid to hip-mid distance (image units) -- body scale."""
    sh = (pose[:, L_SH, :2] + pose[:, R_SH, :2]) / 2
    hp = (pose[:, L_HIP, :2] + pose[:, R_HIP, :2]) / 2
    return float(np.median(np.linalg.norm(sh - hp, axis=1)) + 1e-6)


def leg_length(pose: np.ndarray) -> float:
    """Median hip-knee-ankle chain length, averaged over both legs."""
    segs = []
    for hip, knee, ank in ((L_HIP, L_KNEE, L_ANK), (R_HIP, R_KNEE, R_ANK)):
        segs.append(
            np.linalg.norm(pose[:, hip, :2] - pose[:, knee, :2], axis=1)
            + np.linalg.norm(pose[:, knee, :2] - pose[:, ank, :2], axis=1)
        )
    return float(np.median(np.concatenate(segs)) + 1e-6)


def time_derivative(x: np.ndarray, t: np.ndarray, order: int = 1) -> np.ndarray:
    """Repeated central difference w.r.t. real timestamps (seconds)."""
    out = x
    for _ in range(order):
        out = np.gradient(out, t, axis=0)
    return out


def _smooth(x: np.ndarray, fps: float, cutoff: float = 6.0) -> np.ndarray:
    return lowpass(x, fps, cutoff)


# ============================================================
# Per-frame families: fn(pose (N,33,3), t (N,), fps) -> (N, d)
# ============================================================


def fam_coords_2d(pose, t, fps):
    return hip_centred_gait_coords(pose)[:, :, :2].reshape(len(pose), -1)


def fam_velocity_2d(pose, t, fps):
    c = hip_centred_gait_coords(pose)[:, :, :2]
    return padded_diff(c).reshape(len(pose), -1)


def fam_coords_scaled(pose, t, fps):
    """Hip-centred coordinates in torso lengths (distance-invariant)."""
    return hip_centred_gait_coords(pose).reshape(len(pose), -1) / torso_length(pose)


def _scaled_smoothed(pose, fps):
    return _smooth(hip_centred_gait_coords(pose) / torso_length(pose), fps)


def fam_acceleration(pose, t, fps):
    """Second derivative, torso-lengths / s^2, of 6 Hz low-passed coordinates."""
    a = time_derivative(_scaled_smoothed(pose, fps), t, 2)
    return a.reshape(len(pose), -1)


def fam_jerk(pose, t, fps):
    """Third derivative, torso-lengths / s^3."""
    j = time_derivative(_scaled_smoothed(pose, fps), t, 3)
    return j.reshape(len(pose), -1)


def fam_angular_velocity(pose, t, fps):
    """d/dt of the six baseline angles, degrees / s."""
    return time_derivative(_smooth(baseline_joint_angles(pose), fps), t, 1)


def _tilt(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed angle (deg) of the image-plane vector a->b from horizontal."""
    d = b[:, :2] - a[:, :2]
    return np.degrees(np.arctan2(d[:, 1], np.abs(d[:, 0]) + 1e-8))


def _from_vertical(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    d = bottom[:, :2] - top[:, :2]
    return np.degrees(np.arctan2(d[:, 0], d[:, 1] + 1e-8))


def fam_angles_ext(pose, t, fps):
    """Eight extra biomechanical angles (degrees):
    trunk lean, pelvic obliquity, shoulder tilt, L/R foot pitch,
    inter-thigh angle, L/R shank inclination."""
    p = pose
    sh = (p[:, L_SH] + p[:, R_SH]) / 2
    hp = (p[:, L_HIP] + p[:, R_HIP]) / 2
    thigh_l = p[:, L_KNEE, :2] - p[:, L_HIP, :2]
    thigh_r = p[:, R_KNEE, :2] - p[:, R_HIP, :2]
    cos = np.sum(thigh_l * thigh_r, axis=1) / (
        np.linalg.norm(thigh_l, axis=1) * np.linalg.norm(thigh_r, axis=1) + 1e-8
    )
    return np.stack(
        [
            _from_vertical(sh, hp),
            _tilt(p[:, L_HIP], p[:, R_HIP]),
            _tilt(p[:, L_SH], p[:, R_SH]),
            _tilt(p[:, L_HEEL], p[:, L_TOE]),
            _tilt(p[:, R_HEEL], p[:, R_TOE]),
            np.degrees(np.arccos(np.clip(cos, -1, 1))),
            _from_vertical(p[:, L_KNEE], p[:, L_ANK]),
            _from_vertical(p[:, R_KNEE], p[:, R_ANK]),
        ],
        axis=1,
    )


def fam_angles_3d(pose, t, fps):
    """The six baseline angles computed in 3D (uses the estimator's z)."""
    p = pose
    return np.stack(
        [
            _angle_3d(p[:, L_HIP], p[:, L_KNEE], p[:, L_ANK]),
            _angle_3d(p[:, R_HIP], p[:, R_KNEE], p[:, R_ANK]),
            _angle_3d(p[:, L_SH], p[:, L_HIP], p[:, L_KNEE]),
            _angle_3d(p[:, R_SH], p[:, R_HIP], p[:, R_KNEE]),
            _angle_3d(p[:, L_KNEE], p[:, L_ANK], p[:, L_TOE]),
            _angle_3d(p[:, R_KNEE], p[:, R_ANK], p[:, R_TOE]),
        ],
        axis=1,
    )


def fam_symmetry(pose, t, fps):
    """Per-frame left-minus-right differences: knee, hip, ankle angle;
    heel height; foot position along the walking axis (torso lengths)."""
    ang = baseline_joint_angles(pose)
    ga = analyse_gait(pose, t, fps)
    scale = torso_length(pose)
    heel_h = (pose[:, R_HEEL, 1] - pose[:, L_HEEL, 1]) / scale  # image y is down
    fwd = (ga.heel_fwd_l - ga.heel_fwd_r) / scale
    return np.stack(
        [
            ang[:, 0] - ang[:, 1],
            ang[:, 2] - ang[:, 3],
            ang[:, 4] - ang[:, 5],
            heel_h,
            fwd,
        ],
        axis=1,
    )


def fam_foot(pose, t, fps):
    """Foot dynamics: L/R heel and toe height above the lowest foot point in
    the frame (torso lengths), and L/R toe vertical velocity (torso lengths/s)."""
    scale = torso_length(pose)
    y = pose[:, [L_HEEL, R_HEEL, L_TOE, R_TOE], 1]
    ground = np.max(y, axis=1, keepdims=True)  # image y grows downward
    heights = (ground - y) / scale
    toe_vel = time_derivative(_smooth(pose[:, [L_TOE, R_TOE], 1] / scale, fps), t, 1)
    return np.concatenate([heights, -toe_vel], axis=1)


def _detrend(x: np.ndarray, t: np.ndarray, deg: int = 2) -> np.ndarray:
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        coef = np.polyfit(t, x[:, j], deg)
        out[:, j] = x[:, j] - np.polyval(coef, t)
    return out


def fam_com(pose, t, fps):
    """Centre-of-mass proxy: detrended mid-hip x/y sway (walking progression
    and perspective drift removed by a quadratic fit), plus trunk vector x/y,
    all in torso lengths."""
    scale = torso_length(pose)
    hp = (pose[:, L_HIP, :2] + pose[:, R_HIP, :2]) / 2
    sh = (pose[:, L_SH, :2] + pose[:, R_SH, :2]) / 2
    sway = _detrend(hp, t) / scale
    trunk = (sh - hp) / scale
    return np.concatenate([sway, trunk], axis=1)


def fam_phase(pose, t, fps):
    """sin/cos of left and right gait phase (0 at each heel strike)."""
    ga = analyse_gait(pose, t, fps)
    return np.stack(
        [
            np.sin(ga.phase_l),
            np.cos(ga.phase_l),
            np.sin(ga.phase_r),
            np.cos(ga.phase_r),
        ],
        axis=1,
    )


# ============================================================
# Clip-level families: fn(pose, t, fps) -> (d,)
# ============================================================


def fam_rhythm(pose, t, fps):
    return analyse_gait(pose, t, fps).rhythm_vector()


def fam_stride(pose, t, fps):
    return analyse_gait(pose, t, fps).stride_vector(leg_length(pose))


def fam_frequency(pose, t, fps):
    """Harmonic amplitudes (k = 1..4, relative to total) of four gait signals
    at the detected stride frequency: L/R knee angle, mid-hip vertical, L-R
    heel separation."""
    ga = analyse_gait(pose, t, fps)
    ang = baseline_joint_angles(pose)
    hip_y = (pose[:, L_HIP, 1] + pose[:, R_HIP, 1]) / 2
    signals = [ang[:, 0], ang[:, 1], hip_y, ga.heel_fwd_l - ga.heel_fwd_r]
    out = []
    for s in signals:
        amps = ga.harmonic_amplitudes(s, n_harmonics=4)
        out.extend(amps / (amps.sum() + 1e-8))
    return np.array(out)


PER_FRAME_FAMILIES: Dict[str, Callable] = {
    "coords_2d": fam_coords_2d,
    "velocity_2d": fam_velocity_2d,
    "coords_scaled": fam_coords_scaled,
    "acceleration": fam_acceleration,
    "jerk": fam_jerk,
    "angular_velocity": fam_angular_velocity,
    "angles_ext": fam_angles_ext,
    "angles_3d": fam_angles_3d,
    "symmetry": fam_symmetry,
    "foot": fam_foot,
    "com": fam_com,
    "phase": fam_phase,
}
CLIP_FAMILIES: Dict[str, Callable] = {
    "rhythm": fam_rhythm,
    "stride": fam_stride,
    "frequency": fam_frequency,
}
BASELINE_FAMILIES = ("coords", "angles", "velocity")
ALL_FAMILIES = BASELINE_FAMILIES + tuple(PER_FRAME_FAMILIES) + tuple(CLIP_FAMILIES)

FAMILY_DIMS = {
    "coords": 36,
    "angles": 6,
    "velocity": 36,
    "coords_2d": 24,
    "velocity_2d": 24,
    "coords_scaled": 36,
    "acceleration": 36,
    "jerk": 36,
    "angular_velocity": 6,
    "angles_ext": 8,
    "angles_3d": 6,
    "symmetry": 5,
    "foot": 6,
    "com": 4,
    "phase": 4,
    "rhythm": len(GaitCycleAnalysis.RHYTHM_NAMES),
    "stride": len(GaitCycleAnalysis.STRIDE_NAMES),
    "frequency": 16,
}


def family_slices(families: Sequence[str]) -> Dict[str, slice]:
    """Channel range of each family inside a descriptor built from `families`."""
    out, start = {}, 0
    for f in families:
        out[f] = slice(start, start + FAMILY_DIMS[f])
        start += FAMILY_DIMS[f]
    return out


def descriptor_dim(families: Sequence[str]) -> int:
    return sum(FAMILY_DIMS[f] for f in families)


# ============================================================
# Named feature sets (the ablation hierarchy of plan Section 9)
# ============================================================

FEATURE_SETS: Dict[str, List[str]] = {
    # E0 baseline and its matched 2D control (for z-less pose backends)
    "baseline78": ["coords", "angles", "velocity"],
    "baseline_2d": ["coords_2d", "angles", "velocity_2d"],
    # feature-group decomposition of the baseline
    "coords_only": ["coords"],
    "coords_angles": ["coords", "angles"],
    # E4 biomechanics, one family at a time on top of the baseline
    "b+acceleration": ["coords", "angles", "velocity", "acceleration"],
    "b+jerk": ["coords", "angles", "velocity", "jerk"],
    "b+angular_velocity": ["coords", "angles", "velocity", "angular_velocity"],
    "b+angles_ext": ["coords", "angles", "velocity", "angles_ext"],
    "b+angles_3d": ["coords", "angles", "velocity", "angles_3d"],
    "b+symmetry": ["coords", "angles", "velocity", "symmetry"],
    "b+foot": ["coords", "angles", "velocity", "foot"],
    "b+com": ["coords", "angles", "velocity", "com"],
    "b+stride": ["coords", "angles", "velocity", "stride"],
    "scaled_baseline": ["coords_scaled", "angles", "velocity"],
    # E3 rhythm, one family at a time on top of the baseline
    "b+rhythm": ["coords", "angles", "velocity", "rhythm"],
    "b+frequency": ["coords", "angles", "velocity", "frequency"],
    "b+phase": ["coords", "angles", "velocity", "phase"],
    # combined sets
    "b+biomech": [
        "coords",
        "angles",
        "velocity",
        "acceleration",
        "angles_ext",
        "symmetry",
        "foot",
        "com",
        "stride",
    ],
    "b+rhythm_all": ["coords", "angles", "velocity", "rhythm", "frequency", "phase"],
    "full_engineered": [
        "coords",
        "angles",
        "velocity",
        "acceleration",
        "angles_ext",
        "symmetry",
        "foot",
        "com",
        "stride",
        "rhythm",
        "frequency",
        "phase",
    ],
    # interpretable-only: no raw coordinates at all
    "interpretable_only": [
        "angles",
        "angles_ext",
        "symmetry",
        "foot",
        "com",
        "stride",
        "rhythm",
        "frequency",
    ],
}
