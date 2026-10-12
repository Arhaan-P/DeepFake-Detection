"""Real-video CUDA evidence with ONNX node profiles and frozen pretrained forwards."""

import argparse
from dataclasses import asdict, replace
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
import onnxruntime as ort
from gait_verification.media import sha256_file
from gait_verification.pose import RTMPoseDetector, NativePoseExtractor, save_pose
from gait_verification.backends.gaitpt import GaitPTEncoder
from gait_verification.backends.gpgait import GPGaitEncoder
from gait_verification.pose_quality import WindowedEncoder, analyze_pose
from gait_verification.skeleton_contracts import InsufficientGait


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True)
    p.add_argument("--detector", required=True)
    p.add_argument("--pose-model", required=True)
    p.add_argument("--source-root", default=".cache/gait_verification/sources")
    p.add_argument("--checkpoint-root", default=".cache/gait_verification/checkpoints")
    p.add_argument("--new-output-dir", required=True)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    torch.set_num_threads(2)
    root = Path(args.new_output_dir)
    root.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(Path("configs/gait_verification/phase1a.json").read_text())
    models = json.loads(Path("configs/gait_verification/models.json").read_text())
    d = RTMPoseDetector(
        args.detector,
        args.pose_model,
        expected_detector_sha256=cfg["rtmpose"]["detector"]["sha256"],
        expected_pose_sha256=cfg["rtmpose"]["pose"]["sha256"],
    )
    artifacts = []
    for name, model, path in zip(
        ["detector", "pose"],
        [d.runtime.det_model, d.runtime.pose_model],
        [args.detector, args.pose_model],
    ):
        opts = ort.SessionOptions()
        opts.enable_profiling = True
        opts.profile_file_prefix = str(root / name)
        model.session = ort.InferenceSession(
            path,
            sess_options=opts,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        if "CUDAExecutionProvider" not in model.session.get_providers():
            raise RuntimeError(f"{name} CUDA initialization failed")
        model.session.disable_fallback()
        artifacts.append(
            dict(
                name=name,
                bytes=Path(path).stat().st_size,
                sha256=sha256_file(path),
                inputs=[i.shape for i in model.session.get_inputs()],
                providers=model.session.get_providers(),
            )
        )
    s = NativePoseExtractor(d, "cuda-runtime-smoke", collect_diagnostics=True).extract(
        args.video
    )
    save_pose(s, root / "pose.npz")
    nodes = {}
    for name, model in [
        ("detector", d.runtime.det_model),
        ("pose", d.runtime.pose_model),
    ]:
        profile = Path(model.session.end_profiling())
        events = json.loads(profile.read_text())
        counts = {}
        for event in events:
            provider = event.get("args", {}).get("provider")
            if provider:
                counts[provider] = counts.get(provider, 0) + 1
        if counts.get("CUDAExecutionProvider", 0) == 0:
            raise RuntimeError(f"No real CUDA nodes for {name}")
        nodes[name] = counts
    checkpoint_root = Path(args.checkpoint_root)
    original = checkpoint_root / "gpgait-grew-57.pt"
    if sha256_file(original) != models["gpgait"]["checkpoint_sha256_grew_original"]:
        raise ValueError("Original GPGait hash mismatch")
    results = {}
    for name, cls, source, filename, checksum in [
        (
            "gaitpt",
            GaitPTEncoder,
            "GaitPT",
            "grew-checkpoint.pth",
            models["gaitpt_grew"]["checkpoint_sha256"],
        ),
        (
            "gpgait",
            GPGaitEncoder,
            "FastPoseGait",
            "gpgait-grew-57-model-only.pt",
            models["gpgait"]["checkpoint_sha256_grew_model_only"],
        ),
    ]:
        encoder = cls(
            Path(args.source_root) / source,
            checkpoint_root / filename,
            checksum,
            device="cuda",
        )
        before = (
            {k: v.clone() for k, v in encoder.model.state_dict().items()}
            if name == "gaitpt"
            else None
        )
        a, b = encoder.encode(s), encoder.encode(s)
        delta = float(np.max(np.abs(a.values - b.values)))
        assert delta < 1e-7 and np.isclose(np.linalg.norm(a.values), 1)
        if before is not None:
            assert all(
                torch.equal(v, before[k]) for k, v in encoder.model.state_dict().items()
            )
            assert all(not v.requires_grad for v in encoder.model.parameters())
        window = WindowedEncoder(encoder)
        wa, wb = window.encode(s), window.encode(s)
        assert np.allclose(wa.values, wb.values, atol=1e-7) and np.isclose(
            np.linalg.norm(wa.values), 1
        )
        invalid = replace(s, observed=s.observed.copy())
        invalid.observed[:, 15] = False
        try:
            encoder.encode(invalid)
        except InsufficientGait:
            pass
        else:
            raise AssertionError("Missing joints accepted")
        results[name] = dict(
            metadata=asdict(encoder.metadata),
            embedding_dim=len(a.values),
            norm=float(np.linalg.norm(a.values)),
            repeat_delta=delta,
            windowed_dimension=len(wa.values),
            windowed_norm=float(np.linalg.norm(wa.values)),
            windowed_repeat_delta=float(np.max(np.abs(wa.values - wb.values))),
        )
        print(name, results[name], flush=True)
        del encoder, window, before
    report = dict(
        video=args.video,
        source_sha256=s.source_sha256,
        frames=len(s.xy),
        width=s.width,
        height=s.height,
        artifacts=artifacts,
        cuda_nodes=nodes,
        models=results,
        quality=analyze_pose(s)["summary"],
        gpu=torch.cuda.get_device_name(0),
        driver=subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip(),
        python=sys.version,
        versions={
            n: importlib.metadata.version(n)
            for n in [
                "torch",
                "torchvision",
                "onnxruntime-gpu",
                "rtmlib",
                "numpy",
                "opencv-python",
            ]
        },
    )
    (root / "runtime.json").write_text(json.dumps(report, indent=2))
    print("REAL CUDA RUNTIME: PASSED", flush=True)


if __name__ == "__main__":
    main()
