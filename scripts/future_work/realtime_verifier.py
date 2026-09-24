"""
Real-time / streaming gait verifier (plan Section 11 / P8).
===========================================================
Processes live walking video (webcam) or files frame by frame and emits a
rolling verification score for a claimed identity:

  * pose runs per frame with a MediaPipe VIDEO-mode tracker (temporal
    tracking; the pose benchmark shows it halves keypoint jitter vs IMAGE mode)
  * every --hop seconds, the last --window seconds of skeletons are verified
    with the deployable bundle (utils/gait_verifier.py); the score is smoothed
    with an exponential moving average on the log-odds scale
  * privacy-preserving by construction: frames are discarded after pose
    inference; --skeleton_log optionally stores ONLY keypoints, timestamps and
    scores (no pixels)
  * unknown identities are refused up front
  * multi-camera: pass several --source values; per-window scores are fused
    on the log-odds scale, weighted by each view's pose quality (detection
    rate x mean joint confidence in the window)

Usage:
    python scripts/future_work/realtime_verifier.py \\
        --checkpoint outputs/future_work/checkpoints/E0_baseline.pt \\
        --claimed Teja --source 0                      # webcam
    python scripts/future_work/realtime_verifier.py --checkpoint ... \\
        --claimed Teja --source data/videos/Teja_S1.mp4 --source data/videos/Teja_F1.mp4

Author: DeepFake Detection Project
"""

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from utils.gait_descriptors import clip_from_track
from utils.gait_verifier import GaitVerifier
from utils.pose_backends import NUM_CANONICAL, PoseTrack, create_backend


def _logit(p):
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return np.log(p / (1 - p))


class StreamView:
    """One camera: capture + tracker + rolling skeleton buffer."""

    def __init__(self, source, backend_name: str, window_s: float):
        self.cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
        if not self.cap.isOpened():
            raise OSError(f"Cannot open source {source}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.backend = create_backend(backend_name.replace("_video", "") + "_video")
        self.buf = deque(maxlen=int(window_s * self.fps) + 1)
        self.name = str(source)
        self.frame_idx = 0
        self.latency = deque(maxlen=100)

    def step(self) -> bool:
        ok, frame = self.cap.read()
        if not ok:
            return False
        t = self.frame_idx / self.fps
        t0 = time.perf_counter()
        out = self.backend._infer(frame, int(t * 1000))
        self.latency.append((time.perf_counter() - t0) * 1000)
        del frame  # privacy: pixels are never retained
        if out is None:
            kp = np.full((NUM_CANONICAL, 3), np.nan)
            cf = np.full(NUM_CANONICAL, np.nan)
        else:
            kp, cf, _ = out
        self.buf.append((self.frame_idx, kp, cf))
        self.frame_idx += 1
        return True

    def window_clip(self):
        if len(self.buf) < 10:
            return None, 0.0
        idx = np.array([b[0] for b in self.buf])
        kp = np.array([b[1] for b in self.buf])
        cf = np.array([b[2] for b in self.buf])
        track = PoseTrack(
            keypoints=kp,
            confidence=cf,
            fps=self.fps,
            width=self.w,
            height=self.h,
            backend=self.backend.name,
        )
        clip = clip_from_track(track, f"stream_{self.name}")
        if clip is None:
            return None, 0.0
        # re-anchor timestamps to real frame indices
        clip.t = idx[track.detected()] / self.fps
        quality = float(
            track.detected().mean() * np.nanmean(cf[:, [23, 24, 25, 26, 27, 28]])
        )
        return clip, quality


def main():
    ap = argparse.ArgumentParser(description="Rolling gait verification")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--claimed", required=True)
    ap.add_argument("--source", action="append", required=True)
    ap.add_argument("--window", type=float, default=4.0, help="seconds")
    ap.add_argument("--hop", type=float, default=0.5, help="seconds")
    ap.add_argument(
        "--ema", type=float, default=0.5, help="smoothing weight of new score"
    )
    ap.add_argument(
        "--skeleton_log", default="", help="write skeleton-only log (.jsonl)"
    )
    args = ap.parse_args()

    verifier = GaitVerifier.load(args.checkpoint)
    if args.claimed not in verifier.signatures:
        print(
            f"UNKNOWN_IDENTITY: '{args.claimed}' is not enrolled. "
            f"Enrolled: {verifier.identities}. Refusing to verify."
        )
        return
    views = [StreamView(s, verifier.cfg.backend, args.window) for s in args.source]
    hop_frames = max(1, int(args.hop * views[0].fps))
    log = open(args.skeleton_log, "w") if args.skeleton_log else None
    smoothed = None
    thr_logit = _logit(verifier.threshold)
    print(
        f"Verifying claim '{args.claimed}' on {len(views)} view(s); "
        f"threshold {verifier.threshold:.3f}; window {args.window}s hop {args.hop}s"
    )

    step = 0
    while all(v.step() for v in views):
        step += 1
        if step % hop_frames or len(views[0].buf) < views[0].buf.maxlen:
            continue
        logits, weights, per_view = [], [], {}
        for v in views:
            clip, q = v.window_clip()
            if clip is None:
                continue
            res = verifier.verify(clip, args.claimed)
            if "score_raw" not in res:
                continue
            logits.append(_logit(res["score_raw"]))
            weights.append(max(q, 1e-3))
            per_view[v.name] = {"score": res["score_raw"], "quality": q}
        if not logits:
            print(f"t={step / views[0].fps:6.2f}s  INSUFFICIENT_GAIT")
            continue
        fused = float(np.average(logits, weights=weights))
        smoothed = (
            fused if smoothed is None else (1 - args.ema) * smoothed + args.ema * fused
        )
        verdict = "AUTHENTIC" if smoothed >= thr_logit else "IDENTITY_MISMATCH"
        p = 1 / (1 + np.exp(-smoothed))
        lat = np.mean([np.mean(v.latency) for v in views])
        print(
            f"t={step / views[0].fps:6.2f}s  rolling score {p:.3f}  {verdict:<17s} "
            f"views={len(per_view)}  pose {lat:.0f} ms/frame"
        )
        if log:
            rec = {
                "t": step / views[0].fps,
                "score": p,
                "verdict": verdict,
                "views": per_view,
                "skeleton": {
                    v.name: np.nan_to_num(v.buf[-1][1]).round(4).tolist() for v in views
                },
            }
            log.write(json.dumps(rec) + "\n")
    if log:
        log.close()


if __name__ == "__main__":
    main()
