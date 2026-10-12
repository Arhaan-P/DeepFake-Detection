"""Isolated historical math/model reuse. No legacy verifier, training or gallery API."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
from contextlib import redirect_stdout

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def features(pose):
    from utils.gait_features import baseline_descriptor, resample_sequence

    return baseline_descriptor(resample_sequence(pose, 60)).astype(np.float32)


def state_digest(model):
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def score(checkpoint, checksum, device, query, references):
    import torch
    from models.verifier_variants import TCNVerifier
    from gait_verification.e0_checkpoint import (
        CONFIG_SHA256,
        SCHEMA,
        SHIPPED_SHA256,
        TRAINING_ORIGIN,
    )
    from gait_verification.media import sha256_file

    if sha256_file(checkpoint) != checksum:
        raise ValueError("E0 checkpoint checksum mismatch")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA_UNAVAILABLE_NO_CPU_FALLBACK")
    # Preserve float32 numerical parity across CPU/CUDA; TF32 convolution can
    # otherwise differ despite deterministic repeated CUDA outputs.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    obj = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if set(obj) != {
        "schema_version",
        "model",
        "feature_stats",
        "config",
        "training_origin",
        "source_sha256",
        "source_config_sha256",
    }:
        raise ValueError("Unsupported E0 inference envelope")
    if (
        obj["schema_version"] != SCHEMA
        or obj["source_sha256"] != SHIPPED_SHA256
        or obj["training_origin"] != TRAINING_ORIGIN
        or obj["source_config_sha256"] != CONFIG_SHA256
    ):
        raise ValueError("Unknown historical E0 origin")
    if obj["config"] != {
        "input_dim": 78,
        "hidden": 64,
        "dropout": 0.1,
        "sequence_length": 60,
    }:
        raise ValueError("Incompatible E0 configuration")
    if not all(
        type(k) is str and type(v) is torch.Tensor and torch.isfinite(v).all()
        for k, v in obj["model"].items()
    ):
        raise ValueError("Invalid E0 model tensors")
    stats = obj["feature_stats"]
    if (
        set(stats) != {"mean", "std"}
        or any(
            type(v) is not torch.Tensor
            or v.shape != (78,)
            or not torch.isfinite(v).all()
            for v in stats.values()
        )
        or not (stats["std"] > 0).all()
    ):
        raise ValueError("Invalid E0 feature statistics")
    model = TCNVerifier(78, 64, 0.1)
    model.load_state_dict(obj["model"], strict=True)
    model.eval().requires_grad_(False).to(device)
    before = state_digest(model)
    # Preserve historical z-score then mean-signature semantics. The reference
    # order is canonical in the caller; no signature from another claim is present.
    q = (torch.from_numpy(query) - stats["mean"]) / stats["std"]
    r = (torch.from_numpy(references) - stats["mean"]) / stats["std"]
    r = r.mean(dim=0)
    if not torch.isfinite(q).all() or not torch.isfinite(r).all():
        raise ValueError("Nonfinite normalized E0 features")
    torch.use_deterministic_algorithms(True)
    with torch.inference_mode():
        logits = model(q[None].to(device), r[None].to(device))[0]
        repeated = model(q[None].to(device), r[None].to(device))[0]
    if not torch.isfinite(logits).all() or not torch.equal(logits, repeated):
        raise ValueError("Nonfinite or nondeterministic E0 output")
    after = state_digest(model)
    if before != after:
        raise RuntimeError("Frozen E0 state changed")
    return float((logits[1] - logits[0]).cpu()), {
        "state_before": before,
        "state_after": after,
        "training": model.training,
        "frozen": all(not p.requires_grad for p in model.parameters()),
        "parameters": sum(p.numel() for p in model.parameters()),
        "repeat_max_delta": float((logits - repeated).abs().max().cpu()),
        "device": device,
        "tf32": False,
    }


def extract(video, model_path, model_sha256):
    from utils.pose_extraction import GaitFeatureExtractor
    from gait_verification.media import sha256_file

    if sha256_file(model_path) != model_sha256:
        raise ValueError("MediaPipe model checksum mismatch")

    class LocalExtractor(GaitFeatureExtractor):
        def _get_pose_model(self, *args):
            return str(model_path)  # no automatic model download

        def extract_pose_from_frame(self, frame):
            pose = super().extract_pose_from_frame(frame)
            if pose is not None and (
                pose.shape != (33, 3) or not np.isfinite(pose).all()
            ):
                raise ValueError("MISSING_ANATOMICAL_LANDMARK")
            return pose

    extractor = LocalExtractor(sequence_length=60)
    try:
        with redirect_stdout(io.StringIO()):
            result = extractor.process_video(video)
    finally:
        extractor.landmarker.close()
    if result is None:
        return None
    # Reuse historical detected-pose resampling and exact vectorized feature math.
    return features(result["pose_sequence"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("features", "score", "extract"))
    p.add_argument("--input")
    p.add_argument("--output")
    p.add_argument("--checkpoint")
    p.add_argument("--sha256")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--video")
    a = p.parse_args()
    if a.mode == "score":
        with np.load(a.input, allow_pickle=False) as data:
            s, audit = score(
                a.checkpoint, a.sha256, a.device, data["query"], data["references"]
            )
        print(json.dumps({"score": s, "audit": audit}, allow_nan=False))
    else:
        if a.mode == "features":
            with np.load(a.input, allow_pickle=False) as data:
                f = features(data["pose"])
        else:
            f = extract(a.video, a.checkpoint, a.sha256)
        if f is None:
            print(json.dumps({"status": "INSUFFICIENT_GAIT"}))
        else:
            np.save(a.output, f, allow_pickle=False)
            print(json.dumps({"status": "FEATURED"}))


if __name__ == "__main__":
    main()
