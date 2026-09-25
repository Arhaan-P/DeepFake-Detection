"""
Extract a full-frame-rate pose cache with any pose backend (plan P1, E1/E2).
============================================================================
The baseline pipeline resamples each clip to 60 frames BEFORE computing
features, so the stored .pkl has no real-time axis. Rhythm, cadence and
cycle features (plan Sections 5-6) need the original frame rate, and a fair
pose-backend comparison needs every backend run on identical frames. This
script therefore stores, per video, every frame's canonical 33-slot keypoints,
confidences, an optional 3D estimate, and per-frame latency:

    data/pose_cache/<backend>/<video_stem>.npz      (see utils.pose_backends)

Optional video degradation (plan Section 8: compression / lighting / blur /
resolution) is applied to frames before pose inference, writing to
data/pose_cache/<backend>__<degradation>/.

Usage:
    python scripts/future_work/extract_pose_cache.py --backend mediapipe_lite
    python scripts/future_work/extract_pose_cache.py --backend rtmpose_balanced --workers 4
    python scripts/future_work/extract_pose_cache.py --backend mediapipe_lite --degrade jpeg:20

Author: DeepFake Detection Project
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from utils.pose_backends import PoseBackend, create_backend

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def degrade_frame(frame: np.ndarray, spec: str) -> np.ndarray:
    """Apply one named degradation, e.g. 'jpeg:20', 'scale:0.33', 'gamma:2.5'.

    jpeg:Q      re-encode at JPEG quality Q (compression artefacts)
    scale:F     downscale by F and back up (low resolution / far camera)
    gamma:G     darken with gamma G > 1 plus sensor noise (low light)
    blur:K      Gaussian blur with kernel K (defocus / motion blur)
    """
    kind, _, val = spec.partition(":")
    if kind == "jpeg":
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(val)])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if kind == "scale":
        h, w = frame.shape[:2]
        f = float(val)
        small = cv2.resize(frame, (max(1, int(w * f)), max(1, int(h * f))))
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if kind == "gamma":
        g = float(val)
        dark = 255.0 * (frame / 255.0) ** g
        noise = np.random.default_rng(0).normal(0, 4.0, frame.shape)
        return np.clip(dark + noise, 0, 255).astype(np.uint8)
    if kind == "blur":
        k = int(val) | 1
        return cv2.GaussianBlur(frame, (k, k), 0)
    raise ValueError(f"Unknown degradation '{spec}'")


class DegradedBackend(PoseBackend):
    """Wraps a backend so every frame is degraded before inference."""

    def __init__(self, inner: PoseBackend, spec: str):
        self.inner, self.spec = inner, spec
        self.name = f"{inner.name}__{spec.replace(':', '')}"

    def reset(self):
        self.inner.reset()

    def _infer(self, frame, timestamp_ms):
        return self.inner._infer(degrade_frame(frame, self.spec), timestamp_ms)


_BACKEND = None


def _limit_onnxruntime_threads(n: int) -> None:
    """ONNX Runtime ignores OMP_NUM_THREADS and starts one thread per core in
    EVERY session; with several worker processes that oversubscribes the CPU
    many times over. rtmlib builds its sessions without SessionOptions, so
    inject a thread-capped one."""
    try:
        import onnxruntime as ort
    except ImportError:
        return
    original = ort.InferenceSession

    def capped(*args, **kwargs):
        if kwargs.get("sess_options") is None:
            so = ort.SessionOptions()
            so.intra_op_num_threads = n
            so.inter_op_num_threads = 1
            kwargs["sess_options"] = so
        return original(*args, **kwargs)

    ort.InferenceSession = capped


def _init_worker(backend_name, degrade, backend_kwargs, threads=1):
    global _BACKEND
    import torch

    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    _limit_onnxruntime_threads(threads)
    _BACKEND = create_backend(backend_name, **backend_kwargs)
    if degrade:
        _BACKEND = DegradedBackend(_BACKEND, degrade)


def _process(video_path: str, out_path: str, max_frames: int):
    t0 = time.time()
    track = _BACKEND.process_video(video_path, max_frames=max_frames)
    track.save(out_path)
    det = track.detected()
    return {
        "video": Path(video_path).stem,
        "frames": track.num_frames,
        "detected_frames": int(det.sum()),
        "fps": track.fps,
        "resolution": [track.width, track.height],
        "mean_latency_ms": float(np.mean(track.latency_ms)),
        "seconds": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser(description="Full-rate pose cache extraction")
    ap.add_argument("--backend", required=True)
    ap.add_argument("--videos_dir", default="data/videos")
    ap.add_argument("--cache_root", default="data/pose_cache")
    ap.add_argument("--degrade", default="", help="e.g. jpeg:20, scale:0.33")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument(
        "--threads", type=int, default=1, help="inference threads per worker"
    )
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="first N videos only")
    ap.add_argument("--prototxt", default="", help="OpenPose only")
    ap.add_argument("--caffemodel", default="", help="OpenPose only")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    backend_kwargs = {}
    if args.backend == "openpose_body25":
        backend_kwargs = {"prototxt": args.prototxt, "caffemodel": args.caffemodel}

    tag = args.backend + (f"__{args.degrade.replace(':', '')}" if args.degrade else "")
    out_dir = Path(args.cache_root) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(
        p for p in Path(args.videos_dir).iterdir() if p.suffix.lower() in VIDEO_EXTS
    )
    if args.limit:
        videos = videos[: args.limit]
    todo = [
        v for v in videos if args.overwrite or not (out_dir / f"{v.stem}.npz").exists()
    ]

    print("=" * 70)
    print(f"  POSE CACHE  backend={tag}")
    print(f"  videos: {len(videos)}  to process: {len(todo)}  workers: {args.workers}")
    print("=" * 70, flush=True)

    # Download model weights once in the parent before workers race for them.
    create_backend(args.backend, **backend_kwargs)

    manifest_path = out_dir / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:  # interrupted write; the .npz files remain
            print("  WARNING: unreadable manifest.json, starting a new one")

    t_start = time.time()
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(args.backend, args.degrade, backend_kwargs, args.threads),
    ) as pool:
        futures = {
            pool.submit(
                _process, str(v), str(out_dir / f"{v.stem}.npz"), args.max_frames
            ): v
            for v in todo
        }
        for i, fut in enumerate(as_completed(futures), 1):
            v = futures[fut]
            try:
                info = fut.result()
                manifest[v.stem] = info
                print(
                    f"  [{i:3d}/{len(todo)}] {v.stem:<16s} "
                    f"{info['detected_frames']:4d}/{info['frames']:4d} frames  "
                    f"{info['mean_latency_ms']:7.1f} ms/frame",
                    flush=True,
                )
            except Exception as e:  # keep going; report at the end
                manifest[v.stem] = {"video": v.stem, "error": str(e)}
                print(f"  [{i:3d}/{len(todo)}] {v.stem:<16s} FAILED: {e}", flush=True)
            tmp = manifest_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(manifest, indent=2))
            os.replace(tmp, manifest_path)  # atomic: never a half-written file

    print(f"\n  done in {time.time() - t_start:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
