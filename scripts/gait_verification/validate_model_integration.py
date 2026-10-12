"""Real pretrained CUDA audit on a provenance-verified native RTMPose cache.

Separate from unit tests and from future biometric evaluation. No thresholds,
training, calibration fitting, identity labels, rankings or model selection.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from gait_verification.backends.gaitpt import GaitPTEncoder
from gait_verification.backends.gpgait import (
    GPGaitEncoder,
    GPGaitPlusPlusEncoder,
    GPGaitPlusPlusCASIAEncoder,
)
from gait_verification.backends.fastpose_infer_worker import state_digest
from gait_verification.media import probe_video, sha256_file
from gait_verification.pose import load_pose
from gait_verification.pose_quality import WindowedEncoder, analyze_pose, slice_pose
from gait_verification.skeleton_contracts import InsufficientGait


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True)
    p.add_argument("--pose-cache", required=True)
    p.add_argument("--new-output-dir", required=True)
    p.add_argument(
        "--models",
        nargs="+",
        choices=["gaitpt", "gpgait", "gpgaitpp", "gpgaitpp_casia"],
        default=["gaitpt", "gpgait", "gpgaitpp_casia"],
    )
    p.add_argument("--source-root", default=".cache/gait_verification/sources")
    p.add_argument("--checkpoint-root", default=".cache/gait_verification/checkpoints")
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no fallback")
    torch.set_num_threads(2)
    s = load_pose(args.pose_cache)
    video = probe_video(args.video)
    if s.source_sha256 != video.sha256 or not s.backend_id.startswith(
        "rtmpose-coco17:"
    ):
        raise ValueError("Real RTMPose/source lineage required")
    np.testing.assert_array_equal(s.timestamps, video.timestamps)
    np.testing.assert_array_equal(s.frame_indices, np.arange(len(video.timestamps)))
    original_arrays = {
        k: v.copy() for k, v in vars(s).items() if isinstance(v, np.ndarray)
    }
    root = Path(args.new_output_dir)
    root.mkdir(parents=True, exist_ok=False)
    registry = json.loads(Path("configs/gait_verification/models.json").read_text())
    candidates = {
        "gaitpt": (
            GaitPTEncoder,
            "GaitPT",
            "grew-checkpoint.pth",
            registry["gaitpt_grew"]["checkpoint_sha256"],
        ),
        "gpgait": (
            GPGaitEncoder,
            "FastPoseGait",
            "gpgait-grew-57-model-only.pt",
            registry["gpgait"]["checkpoint_sha256_grew_model_only"],
        ),
    }
    if "gpgaitpp" in args.models:
        if not registry["gpgaitpp_grew"].get("model_only_sha256"):
            raise RuntimeError(
                "GREW GPGait++ checkpoint unavailable; see model registry retrieval blocker"
            )
        candidates["gpgaitpp"] = (
            GPGaitPlusPlusEncoder,
            "FastPoseGait",
            "gpgaitpp-grew-model-only.pt",
            registry["gpgaitpp_grew"]["model_only_sha256"],
        )
    if "gpgaitpp_casia" in args.models:
        candidates["gpgaitpp_casia"] = (
            GPGaitPlusPlusCASIAEncoder,
            "FastPoseGait",
            "gpgaitpp-casia-model-only.pt",
            registry["gpgaitpp_casia"]["model_only_sha256"],
        )
    results = {}
    for name in args.models:
        cls, source, filename, checksum = candidates[name]
        src, checkpoint = (
            Path(args.source_root) / source,
            Path(args.checkpoint_root) / filename,
        )
        encoder = cls(src, checkpoint, checksum)
        started = time.perf_counter()
        if name == "gaitpt":
            before = state_digest(encoder.model)
            torch.cuda.reset_peak_memory_stats()
            a, b = encoder.encode(s), encoder.encode(s)
            assert before == state_digest(encoder.model)
            assert not encoder.model.training and all(
                not p.requires_grad for p in encoder.model.parameters()
            )
            execution = dict(
                state_sha256=before,
                frozen=True,
                parameters=sum(p.numel() for p in encoder.model.parameters()),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            )
        else:
            a = encoder.encode(s, audit_path=root / f"{name}-worker.json")
            b = encoder.encode(s)
            execution = json.loads((root / f"{name}-worker.json").read_text())
        np.testing.assert_allclose(a.values, b.values, atol=1e-7, rtol=0)
        assert np.isclose(np.linalg.norm(a.values), 1)
        minimum = encoder.encode(slice_pose(s, 0, 31))
        assert len(minimum.values) == len(a.values)
        window = WindowedEncoder(encoder)
        wa, wb = window.encode(s), window.encode(s)
        np.testing.assert_allclose(wa.values, wb.values, atol=1e-7, rtol=0)
        assert np.isclose(np.linalg.norm(wa.values), 1) and wa.model_id != a.model_id
        rejected = []
        for fault in ("short", "missing", "cropped", "tracking"):
            bad = slice_pose(s, 0, len(s.xy))
            if fault == "short":
                bad = slice_pose(s, 0, 30)
            elif fault == "missing":
                bad.observed[:, 15] = False
            elif fault == "cropped":
                bad.xy[:, 15, 1] = s.height
            else:
                bad.track_ids[len(s.xy) // 2 :] += 1
            for path, target in (("native", encoder), ("window", window)):
                try:
                    target.encode(bad)
                except InsufficientGait:
                    rejected.append(f"{fault}:{path}")
                else:
                    raise AssertionError(f"Accepted {fault}:{path}")
        try:
            cls(src, checkpoint, "0" * 64)
        except ValueError:
            rejected.append("checksum")
        else:
            raise AssertionError("Bad checksum accepted")
        if name != "gaitpt":
            try:
                cls(src, checkpoint, checksum, device="cpu")
            except ValueError:
                rejected.append("cpu")
            else:
                raise AssertionError("Silent CPU accepted")
        for k, v in original_arrays.items():
            np.testing.assert_array_equal(v, getattr(s, k))
        assert sha256_file(checkpoint) == checksum
        results[name] = dict(
            metadata=asdict(encoder.metadata),
            model_id=a.model_id,
            windowed_model_id=wa.model_id,
            dimension=len(a.values),
            norm=float(np.linalg.norm(a.values)),
            repeat_delta=float(np.max(np.abs(a.values - b.values))),
            minimum_31_frames_passed=True,
            window_repeat_delta=float(np.max(np.abs(wa.values - wb.values))),
            rejection_checks=rejected,
            execution=execution,
            audit_seconds=time.perf_counter() - started,
        )
        print(name, "PASSED", flush=True)
        del encoder, window
        torch.cuda.empty_cache()
    assert sha256_file(args.video) == s.source_sha256
    report = dict(
        source_sha256=s.source_sha256,
        pose_cache_sha256=sha256_file(args.pose_cache),
        frames=len(s.xy),
        backend_id=s.backend_id,
        quality=analyze_pose(s)["summary"],
        models=results,
        torch=torch.__version__,
        cuda_build=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),
        evidence="real pretrained CUDA on historical RTMPose; no biometric evaluation",
    )
    with open(root / "models.json", "x") as f:
        json.dump(report, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
