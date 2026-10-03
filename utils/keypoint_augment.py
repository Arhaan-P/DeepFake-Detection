"""
Keypoint-Level Augmentation and Robustness Perturbations (plan Sections 8-9)
===========================================================================
The baseline augments VIDEOS (16x) and re-runs MediaPipe on every copy. For
controlled experiments across several pose backends that is 16x the pose
inference per backend, so the future-work harness augments the extracted
keypoint streams instead. Every operation acts on a `Clip`
(utils/gait_descriptors.py): raw canonical pose (N, 33, 3), timestamps t (s).

Training presets
----------------
rhythm_safe  (default)  rotation +/-10 deg, zoom 0.9-1.1, small translation,
                        keypoint jitter, temporal crop 70-100%, 10% frame drop.
                        Leaves cadence and left/right structure intact, so
                        rhythm/symmetry features are not trained away.
paper_like              rhythm_safe + horizontal mirror (L/R swapped) and
                        playback speed 0.8x / 1.2x, approximating the paper's
                        video-level hflip and speed augmentations.

Test-time perturbations (robustness, E8) are deterministic and named, e.g.
"noise:0.01", "occlude_feet:0.3", "fps:10", "truncate:0.5", "scale:0.6",
"drop:0.3", "speed:1.2", "mirror".

Author: DeepFake Detection Project
"""

from dataclasses import replace
from typing import List

import numpy as np

from utils.gait_features import CANONICAL_LR_PAIRS

FOOT_JOINTS = [25, 26, 27, 28, 29, 30, 31, 32]  # knees, ankles, heels, toes

PRESETS = {
    "none": {},
    "rhythm_safe": {
        "rotate_deg": 10.0,
        "zoom": (0.9, 1.1),
        "shift": 0.03,
        "noise": 0.003,
        "crop": (0.7, 1.0),
        "drop": 0.1,
    },
    "paper_like": {
        "rotate_deg": 10.0,
        "zoom": (0.9, 1.1),
        "shift": 0.03,
        "noise": 0.003,
        "crop": (0.7, 1.0),
        "drop": 0.1,
        "mirror_p": 0.5,
        "speeds": (0.8, 1.0, 1.2),
    },
}


def _affine(pose: np.ndarray, aspect: float, angle_deg=0.0, zoom=1.0, shift=(0, 0)):
    """Rotate/zoom/shift image-normalised x,y about the frame centre, in
    isotropic pixel space (x scaled by aspect = W/H)."""
    out = pose.copy()
    x = (pose[..., 0] - 0.5) * aspect
    y = pose[..., 1] - 0.5
    a = np.radians(angle_deg)
    xr = zoom * (np.cos(a) * x - np.sin(a) * y)
    yr = zoom * (np.sin(a) * x + np.cos(a) * y)
    out[..., 0] = xr / aspect + 0.5 + shift[0]
    out[..., 1] = yr + 0.5 + shift[1]
    out[..., 2] = pose[..., 2] * zoom
    return out


def mirror_pose(pose: np.ndarray) -> np.ndarray:
    """Horizontal mirror with anatomical left/right swapped, so the result is
    a plausible mirrored walker rather than a person with crossed labels."""
    out = pose.copy()
    out[..., 0] = 1.0 - pose[..., 0]
    for a, b in CANONICAL_LR_PAIRS:
        out[:, [a, b]] = out[:, [b, a]]
    return out


def _subset(clip, keep: np.ndarray):
    return replace(
        clip,
        pose=clip.pose[keep],
        t=clip.t[keep],
        conf=None if clip.conf is None else clip.conf[keep],
    )


def augment_clip(clip, rng: np.random.Generator, preset: str = "rhythm_safe"):
    """One random augmented copy of `clip` under a named preset."""
    cfg = PRESETS[preset]
    if not cfg:
        return clip
    pose, t = clip.pose, clip.t

    if "speeds" in cfg:
        s = float(rng.choice(cfg["speeds"]))
        t = t / s
    if rng.random() < cfg.get("mirror_p", 0.0):
        pose = mirror_pose(pose)
    pose = _affine(
        pose,
        clip.aspect,
        angle_deg=rng.uniform(-cfg["rotate_deg"], cfg["rotate_deg"]),
        zoom=rng.uniform(*cfg["zoom"]),
        shift=rng.uniform(-cfg["shift"], cfg["shift"], size=2),
    )
    pose = pose + rng.normal(0, cfg["noise"], size=pose.shape)
    clip = replace(clip, pose=pose, t=t)

    n = len(t)
    frac = rng.uniform(*cfg["crop"])
    length = max(int(round(frac * n)), min(n, 20))
    start = int(rng.integers(0, n - length + 1))
    keep = np.arange(start, start + length)
    if cfg.get("drop"):
        mask = rng.random(len(keep)) >= cfg["drop"]
        mask[[0, -1]] = True
        keep = keep[mask]
    return _subset(clip, keep)


def perturb_clip(clip, spec: str, seed: int = 0):
    """Deterministic named test-time perturbation (see module docstring)."""
    rng = np.random.default_rng(seed)
    kind, _, val = spec.partition(":")
    v = float(val) if val else 0.0
    pose, t = clip.pose, clip.t

    if kind == "noise":  # keypoint jitter, std in image-height units
        return replace(clip, pose=pose + rng.normal(0, v, size=pose.shape))
    if kind == "occlude_feet":  # fraction of frames with lower limbs hidden
        pose = pose.copy()
        n = len(t)
        width = max(1, int(v * n))
        start = int(rng.integers(0, max(1, n - width)))
        seg = slice(start, start + width)
        # interpolate lower-limb joints across the occluded window, the same
        # gap-filling the clip loader applies to real missing detections
        for j in FOOT_JOINTS:
            for c in range(3):
                known = np.ones(n, bool)
                known[seg] = False
                pose[~known, j, c] = np.interp(t[~known], t[known], pose[known, j, c])
        return replace(clip, pose=pose)
    if kind == "fps":  # lower capture frame rate
        step = max(1, int(round(clip.fps / v)))
        return replace(_subset(clip, np.arange(0, len(t), step)), fps=clip.fps / step)
    if kind == "truncate":  # keep only the first fraction of the clip
        n = max(10, int(round(v * len(t))))
        return _subset(clip, np.arange(n))
    if kind == "drop":  # random missing frames
        keep = rng.random(len(t)) >= v
        keep[[0, -1]] = True
        return _subset(clip, np.where(keep)[0])
    if kind == "scale":  # camera farther away (v < 1) or closer (v > 1)
        return replace(clip, pose=_affine(pose, clip.aspect, zoom=v))
    if kind == "rotate":
        return replace(clip, pose=_affine(pose, clip.aspect, angle_deg=v))
    if kind == "speed":
        return replace(clip, t=t / v)
    if kind == "mirror":
        return replace(clip, pose=mirror_pose(pose))
    raise ValueError(f"Unknown perturbation '{spec}'")


ROBUSTNESS_SUITE: List[str] = [
    "noise:0.005",
    "noise:0.01",
    "noise:0.02",
    "occlude_feet:0.2",
    "occlude_feet:0.4",
    "fps:15",
    "fps:10",
    "truncate:0.66",
    "truncate:0.4",
    "drop:0.3",
    "scale:0.6",
    "scale:1.3",
    "rotate:8",
    "speed:0.85",
    "speed:1.15",
    "mirror",
]
