"""
Clip Loading and Descriptor Building (future-work experiment harness)
=====================================================================
Turns a full-frame-rate pose track (utils/pose_backends.PoseTrack) into the
(T, D) descriptor sequence the verifiers consume, for any combination of
feature families (utils/gait_features.py) and either sequence mode:

  clip   (baseline)  Detected frames are resampled to 60 steps by index, the
                     exact procedure of utils/pose_extraction.py; baseline
                     families are computed on that 60-step pose, new
                     per-frame families are computed at the native frame rate
                     and resampled onto the same grid, clip-level families are
                     broadcast over time.

  cycle  (RQ4)       Every per-frame family is computed at the native rate,
                     cut into heel-strike-to-heel-strike cycles, each cycle is
                     resampled to 60 points of %-gait-cycle, and cycles are
                     averaged. Phase-aligned, so walking speed and clip start
                     no longer shift the comparison. Falls back to `clip` mode
                     when fewer than two strides are detected.

Author: DeepFake Detection Project
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from utils.gait_cycle import analyse_gait, cycle_normalise
from utils.gait_features import (
    BASELINE_FAMILIES,
    CLIP_FAMILIES,
    PER_FRAME_FAMILIES,
    baseline_joint_angles,
    hip_centred_gait_coords,
    padded_diff,
    resample_sequence,
)

NAME_RE = re.compile(r"^(?P<identity>.+)_(?P<view>[A-Za-z]+)(?P<take>\d+)$")


@dataclass
class Clip:
    """One walking clip after detection filtering and gap filling."""

    name: str
    identity: str
    view: str
    take: str
    pose: np.ndarray  # (N, 33, 3) image-normalised (or world) coordinates
    t: np.ndarray  # (N,) seconds
    fps: float
    aspect: float  # W / H; 1.0 for world coordinates
    conf: Optional[np.ndarray] = None  # (N, 33)
    meta: Dict = field(default_factory=dict)

    @property
    def pose_iso(self) -> np.ndarray:
        """Isotropic coordinates (x rescaled to image-height units)."""
        if self.aspect == 1.0:
            return self.pose
        p = self.pose.copy()
        p[..., 0] *= self.aspect
        return p

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) > 1 else 0.0


def parse_clip_name(stem: str):
    """'Vedant2_S3' -> ('Vedant2', 'S', '3'). Unknown formats keep the stem."""
    m = NAME_RE.match(stem)
    if not m:
        return stem, "unknown", "0"
    return m.group("identity"), m.group("view").upper(), m.group("take")


def _fill_gaps(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaNs along time for every (joint, coord) series."""
    out = x.copy()
    flat = out.reshape(len(t), -1)
    for j in range(flat.shape[1]):
        col = flat[:, j]
        ok = np.isfinite(col)
        if ok.all():
            continue
        if ok.sum() >= 2:
            col[~ok] = np.interp(t[~ok], t[ok], col[ok])
        else:
            col[~ok] = 0.0
    return flat.reshape(x.shape)


def clip_from_track(
    track, name: str, source: str = "image", min_frames: int = 10
) -> Optional[Clip]:
    """Build a Clip from a PoseTrack. Frames without a person detection are
    dropped (as the baseline does); missing joints inside detected frames are
    interpolated. `source='world'` uses the backend's 3D estimate."""
    det = track.detected()
    if det.sum() < min_frames:
        return None
    idx = np.where(det)[0]
    t = idx / track.fps
    if source == "world":
        if track.world is None:
            raise ValueError(f"{name}: backend {track.backend} has no 3D output")
        pose, aspect = track.world[idx], 1.0
    else:
        pose, aspect = track.keypoints[idx], track.width / max(track.height, 1)
    pose = _fill_gaps(pose, t)
    identity, view, take = parse_clip_name(name)
    return Clip(
        name=name,
        identity=identity,
        view=view,
        take=take,
        pose=pose,
        t=t,
        fps=track.fps,
        aspect=aspect,
        conf=track.confidence[idx],
        meta={"backend": track.backend, "frames_total": track.num_frames},
    )


def _baseline_family(name: str, pose60: np.ndarray) -> np.ndarray:
    coords = hip_centred_gait_coords(pose60)
    t = len(pose60)
    if name == "coords":
        return coords.reshape(t, -1)
    if name == "angles":
        return baseline_joint_angles(pose60)
    if name == "velocity":
        return padded_diff(coords).reshape(t, -1)
    raise KeyError(name)


def _native_family(name: str, clip: Clip) -> np.ndarray:
    """A per-frame family evaluated at the native frame rate, (N, d)."""
    pose = clip.pose
    if name in BASELINE_FAMILIES:
        # Baseline families at native rate (cycle mode); angles keep the
        # baseline's anisotropic image coordinates to stay comparable.
        return _baseline_family(name, pose)
    return PER_FRAME_FAMILIES[name](clip.pose_iso, clip.t, clip.fps)


def build_descriptor(
    clip: Clip, families: Sequence[str], mode: str = "clip", length: int = 60
) -> np.ndarray:
    """(length, D) descriptor for `families`, in the family order given."""
    per_frame = [f for f in families if f not in CLIP_FAMILIES]
    static = {
        f: CLIP_FAMILIES[f](clip.pose_iso, clip.t, clip.fps)
        for f in families
        if f in CLIP_FAMILIES
    }

    blocks: Dict[str, np.ndarray] = {}
    if mode == "cycle":
        ga = analyse_gait(clip.pose_iso, clip.t, clip.fps)
        events = ga.hs_l if len(ga.hs_l) >= len(ga.hs_r) else ga.hs_r
        natives = [_native_family(f, clip) for f in per_frame]
        native = (
            np.concatenate(natives, axis=1) if natives else np.zeros((len(clip.t), 0))
        )
        cycles = cycle_normalise(native, clip.t, events, length)
        if cycles is not None:
            mean_cycle = cycles.mean(axis=0)
            start = 0
            for f, arr in zip(per_frame, natives):
                blocks[f] = mean_cycle[:, start : start + arr.shape[1]]
                start += arr.shape[1]
        else:
            mode = "clip"

    if mode == "clip":
        pose60 = resample_sequence(clip.pose, length)
        for f in per_frame:
            if f in BASELINE_FAMILIES:
                blocks[f] = _baseline_family(f, pose60)
            else:
                blocks[f] = resample_sequence(_native_family(f, clip), length)

    out: List[np.ndarray] = []
    for f in families:
        if f in static:
            out.append(np.repeat(static[f][None, :], length, axis=0))
        else:
            out.append(blocks[f])
    desc = np.concatenate(out, axis=1)
    return np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
