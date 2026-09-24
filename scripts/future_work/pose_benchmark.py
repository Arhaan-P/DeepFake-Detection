"""
Pose-backend benchmark: signal quality on identical videos (plan P1, E1/E2).
===========================================================================
For every pose cache in data/pose_cache/, computes the quality metrics of
utils/pose_quality.py per video, then aggregates by backend and by view
(frontal F / side S). Also measures agreement with a reference backend
(median per-joint distance, torso lengths) so "different" can be separated
from "better".

Verification performance for the same backends comes from the E1/E2
experiments (run_experiment.py); compare_results.py joins the two, which is
the plan's two-step comparison: signal quality first, verification second.

Usage:
    python scripts/future_work/pose_benchmark.py
    python scripts/future_work/pose_benchmark.py --reference mediapipe_heavy

Author: DeepFake Detection Project
"""

import argparse
import json
import os
import sys
from glob import glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from utils.gait_descriptors import parse_clip_name
from utils.gait_features import GAIT_LANDMARKS
from utils.pose_backends import GAIT_JOINT_NAMES, PoseTrack
from utils.pose_quality import track_quality

KEY_METRICS = [
    "detection_rate",
    "mean_confidence",
    "jitter",
    "accel_rms",
    "bone_length_cv",
    "lr_swap_rate",
    "planted_heel_speed",
    "period_confidence",
    "phase_locking",
    "latency_ms",
]


def agreement(track, ref) -> np.ndarray:
    """Per-gait-joint median 2D distance to the reference, torso lengths."""
    n = min(track.num_frames, ref.num_frames)
    a = track.keypoints[:n, GAIT_LANDMARKS, :2].copy()
    b = ref.keypoints[:n, GAIT_LANDMARKS, :2].copy()
    asp = track.width / max(track.height, 1)
    a[..., 0] *= asp
    b[..., 0] *= asp
    sh = (ref.keypoints[:n, 11, :2] + ref.keypoints[:n, 12, :2]) / 2
    hp = (ref.keypoints[:n, 23, :2] + ref.keypoints[:n, 24, :2]) / 2
    torso = np.nanmedian(np.linalg.norm(sh - hp, axis=1)) + 1e-6
    d = np.linalg.norm(a - b, axis=2) / torso
    return np.nanmedian(d, axis=0)


def main():
    ap = argparse.ArgumentParser(description="Pose-backend quality benchmark")
    ap.add_argument("--cache_root", default="data/pose_cache")
    ap.add_argument("--reference", default="mediapipe_heavy")
    ap.add_argument("--out", default="outputs/future_work/pose_benchmark")
    args = ap.parse_args()

    backends = sorted(
        d
        for d in os.listdir(args.cache_root)
        if os.path.isdir(Path(args.cache_root) / d)
    )
    ref_dir = Path(args.cache_root) / args.reference
    results = {"reference": args.reference, "backends": {}}

    for be in backends:
        per_video = {}
        agree = []
        for f in sorted(glob(str(Path(args.cache_root) / be / "*.npz"))):
            stem = Path(f).stem
            tr = PoseTrack.load(f)
            q = track_quality(tr)
            q["view"] = parse_clip_name(stem)[1]
            per_video[stem] = q
            rf = ref_dir / f"{stem}.npz"
            if rf.exists() and be != args.reference:
                agree.append(agreement(tr, PoseTrack.load(str(rf))))

        def agg(rows):
            out = {}
            for m in KEY_METRICS:
                vals = np.array([r.get(m, np.nan) for r in rows], dtype=float)
                out[m] = float(np.nanmean(vals)) if np.isfinite(vals).any() else None
            return out

        rows = list(per_video.values())
        entry = {
            "n_videos": len(rows),
            "degraded": "__" in be,
            "overall": agg(rows),
            "by_view": {
                v: agg([r for r in rows if r["view"] == v])
                for v in sorted(set(r["view"] for r in rows))
            },
            "per_video": per_video,
        }
        if agree:
            a = np.nanmean(np.array(agree), axis=0)
            entry["agreement_vs_reference"] = {
                "mean_torso_lengths": float(np.nanmean(a)),
                "per_joint": dict(zip(GAIT_JOINT_NAMES, map(float, a))),
            }
        results["backends"][be] = entry
        o = entry["overall"]
        print(
            f"{be:<28s} n={len(rows):2d} det={o['detection_rate']:.3f} "
            f"conf={o['mean_confidence'] or float('nan'):.2f} jitter={o['jitter'] or float('nan'):.4f} "
            f"boneCV={o['bone_length_cv'] or float('nan'):.3f} swap={o['lr_swap_rate'] or float('nan'):.4f} "
            f"period_conf={o['period_confidence'] or float('nan'):.2f} "
            f"lat={o['latency_ms']:.0f}ms"
        )

    os.makedirs(args.out, exist_ok=True)
    with open(Path(args.out) / "pose_benchmark.json", "w") as fh:
        json.dump(results, fh, indent=1)

    # markdown table (clean backends first, degraded after)
    lines = [
        "| Backend | Detect | Conf | Jitter | Accel RMS | Bone CV | L/R swaps "
        "| Planted heel | Period conf | Agreement* | ms/frame |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    order = sorted(results["backends"], key=lambda b: ("__" in b, b))
    for be in order:
        e = results["backends"][be]
        o = e["overall"]
        ag = e.get("agreement_vs_reference", {}).get("mean_torso_lengths")

        def f(v, fmt):
            return "–" if v is None else format(v, fmt)

        lines.append(
            f"| {be} | {f(o['detection_rate'], '.3f')} | {f(o['mean_confidence'], '.2f')} "
            f"| {f(o['jitter'], '.4f')} | {f(o['accel_rms'], '.1f')} "
            f"| {f(o['bone_length_cv'], '.3f')} | {f(o['lr_swap_rate'], '.4f')} "
            f"| {f(o['planted_heel_speed'], '.3f')} | {f(o['period_confidence'], '.2f')} "
            f"| {'ref' if be == args.reference else f(ag, '.3f')} | {f(o['latency_ms'], '.0f')} |"
        )
    lines.append(
        f"\n*median per-joint distance to `{args.reference}`, in torso lengths."
    )
    (Path(args.out) / "pose_benchmark.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}/pose_benchmark.json and .md")


if __name__ == "__main__":
    main()
