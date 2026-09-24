"""
Pose-Signal Quality Metrics (future work, plan Section 3 / P1)
=============================================================
"Compare pose quality first": before asking whether a pose estimator improves
verification, measure whether it produces a cleaner gait signal on the SAME
frames. All metrics are computed on one PoseTrack (utils/pose_backends.py)
over the 12 gait joints, in isotropic image units normalised by torso length
so that numbers are comparable across videos and camera distances.

  detection_rate        frames with a person / all frames
  joint_missing_rate    missing gait joints inside detected frames
  mean_confidence       mean visibility / score of the gait joints
  jitter                RMS of the >6 Hz residual of joint trajectories
                        (walking content is below ~6 Hz; above it is noise)
  accel_rms             RMS joint acceleration (temporal smoothness)
  bone_length_cv        coefficient of variation of thigh, shank and foot
                        lengths over time (a rigid segment should not change;
                        2D projection adds some variation, identical across
                        backends on the same video)
  lr_swap_rate          frames where the left/right ankle labels jump to the
                        other leg's previous position
  planted_heel_speed    heel speed during its slowest 25% of frames, i.e. how
                        still a planted foot looks (foot/heel stability)
  period_confidence     autocorrelation height of the detected stride period
                        (can the gait cycle be recovered at all?)
  latency_ms            per-frame inference time

Author: DeepFake Detection Project
"""

from typing import Dict

import numpy as np

from utils.gait_cycle import analyse_gait, lowpass
from utils.gait_features import GAIT_LANDMARKS

SEGMENTS = {
    "thigh": [(23, 25), (24, 26)],
    "shank": [(25, 27), (26, 28)],
    "foot": [(29, 31), (30, 32)],
}


def _torso(xy: np.ndarray) -> float:
    sh = (xy[:, 11] + xy[:, 12]) / 2
    hp = (xy[:, 23] + xy[:, 24]) / 2
    return float(np.nanmedian(np.linalg.norm(sh - hp, axis=1)) + 1e-6)


def track_quality(track) -> Dict[str, float]:
    """Quality metrics of one PoseTrack (see module docstring)."""
    kp = track.keypoints
    n = len(kp)
    det = track.detected()
    out = {
        "frames": int(n),
        "detection_rate": float(det.mean()) if n else 0.0,
        "latency_ms": (
            float(np.mean(track.latency_ms))
            if track.latency_ms is not None
            else float("nan")
        ),
    }
    if det.sum() < 20:
        return out
    idx = np.where(det)[0]
    t = idx / track.fps
    xy = kp[idx][:, :, :2].copy()
    xy[..., 0] *= track.width / max(track.height, 1)  # isotropic
    gait = xy[:, GAIT_LANDMARKS]
    out["joint_missing_rate"] = float(np.mean(~np.isfinite(gait[..., 0])))
    out["mean_confidence"] = float(np.nanmean(track.confidence[idx][:, GAIT_LANDMARKS]))

    # gap-fill for the temporal metrics
    for j in range(xy.shape[1]):
        for c in range(2):
            col = xy[:, j, c]
            ok = np.isfinite(col)
            if 2 <= ok.sum() < len(col):
                col[~ok] = np.interp(t[~ok], t[ok], col[ok])
            elif ok.sum() < 2:
                col[:] = 0.0
    scale = _torso(xy)
    g = xy[:, GAIT_LANDMARKS] / scale
    smooth = lowpass(g, track.fps, 6.0)
    out["jitter"] = float(np.sqrt(np.mean((g - smooth) ** 2)))
    acc = np.diff(g, n=2, axis=0) * track.fps**2
    out["accel_rms"] = float(np.sqrt(np.mean(acc**2)))

    cvs = {}
    for seg, pairs in SEGMENTS.items():
        lens = np.concatenate(
            [np.linalg.norm(xy[:, a] - xy[:, b], axis=1) for a, b in pairs]
        )
        cvs[seg] = float(np.std(lens) / (np.mean(lens) + 1e-9))
    out["bone_length_cv"] = float(np.mean(list(cvs.values())))
    out.update({f"bone_cv_{k}": v for k, v in cvs.items()})

    la, ra = xy[:, 27], xy[:, 28]
    keep = np.linalg.norm(la[1:] - la[:-1], axis=1) + np.linalg.norm(
        ra[1:] - ra[:-1], axis=1
    )
    swap = np.linalg.norm(la[1:] - ra[:-1], axis=1) + np.linalg.norm(
        ra[1:] - la[:-1], axis=1
    )
    sep = np.linalg.norm(la - ra, axis=1)[1:]
    out["lr_swap_rate"] = float(np.mean((swap < 0.5 * keep) & (sep > 0.05 * scale)))

    speeds = []
    for heel in (29, 30):
        v = np.linalg.norm(
            np.diff(lowpass(xy[:, heel], track.fps, 6.0), axis=0), axis=1
        )
        v = v * track.fps / scale
        speeds.append(np.mean(np.sort(v)[: max(1, len(v) // 4)]))
    out["planted_heel_speed"] = float(np.mean(speeds))

    try:
        pose = kp[idx].copy()
        for j in range(pose.shape[1]):
            for c in range(3):
                col = pose[:, j, c]
                ok = np.isfinite(col)
                if 2 <= ok.sum() < len(col):
                    col[~ok] = np.interp(t[~ok], t[ok], col[ok])
                elif ok.sum() < 2:
                    col[:] = 0.0
        pose[..., 0] *= track.width / max(track.height, 1)
        ga = analyse_gait(pose, t, track.fps)
        out["period_confidence"] = float(ga.period_confidence)
        out["stride_period_s"] = float(ga.stride_period)
        out["phase_locking"] = float(ga.rhythm_dict()["phase_locking"])
    except Exception:  # degenerate track
        out["period_confidence"] = float("nan")
    return out
