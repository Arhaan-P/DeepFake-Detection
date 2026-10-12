"""Isolated, CUDA-only strict pretrained inference; never constructs an optimizer."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gait_verification.backends.fastpose_contract import (
    official_config,
    validate_source,
)
from gait_verification.media import sha256_file


def state_digest(model):
    """Include parameters and buffers without retaining a second GPU model."""
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_model(source, checkpoint, checksum, variant):
    # Recheck at the actual loading boundary, even if caller already checked.
    source = validate_source(source)
    if not checksum or sha256_file(checkpoint) != checksum:
        raise ValueError("FastPoseGait checkpoint checksum mismatch")
    config = official_config(source, variant)
    sys.path.insert(0, str(source / "fastposegait"))
    module = importlib.import_module(
        "modeling.models.gpgait++"
        if variant.startswith("gpgaitpp")
        else "modeling.models.gpgait"
    )
    cls = getattr(module, config["model"])
    model = cls.__new__(cls)
    torch.nn.Module.__init__(model)
    model.build_network(config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or set(state) != {"model"}:
        raise ValueError("Expected a model-only checkpoint")
    if not all(
        isinstance(k, str) and type(v) is torch.Tensor
        for k, v in state["model"].items()
    ):
        raise ValueError("Expected tensor-only model state")
    model.load_state_dict(state["model"], strict=True)
    return model.eval().requires_grad_(False), config


def validate_input(array):
    if (
        array.ndim != 5
        or array.shape[0] != 1
        or array.shape[1] != 6
        or array.shape[2] < 31
        or array.shape[3:] != (17, 1)
        or not np.isfinite(array).all()
    ):
        raise ValueError("Invalid GPGait input tensor")


def main():
    parser = argparse.ArgumentParser()
    for name in ("source", "checkpoint", "sha256", "input", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument(
        "--variant", choices=("gpgait", "gpgaitpp", "gpgaitpp_casia"), default="gpgait"
    )
    parser.add_argument("--audit-output")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FastPoseGait inference")
    torch.set_num_threads(2)
    array = np.load(args.input, allow_pickle=False)
    validate_input(array)
    started = time.perf_counter()
    model, config = load_model(args.source, args.checkpoint, args.sha256, args.variant)
    before = state_digest(model) if args.audit_output else None
    torch.cuda.reset_peak_memory_stats()
    model = model.to("cuda:0")
    x = torch.from_numpy(array.astype(np.float32)).to("cuda:0")
    seq_lengths = torch.tensor([[array.shape[2]]], dtype=torch.int32)
    torch.cuda.synchronize()
    forward_start = time.perf_counter()
    with torch.inference_mode():
        embedding = model(([x], None, None, None, seq_lengths))["inference_feat"][
            "embeddings"
        ]
        repeat = (
            model(([x], None, None, None, seq_lengths))["inference_feat"]["embeddings"]
            if args.audit_output
            else embedding
        )
    torch.cuda.synchronize()
    forward_seconds = time.perf_counter() - forward_start
    if (
        embedding.shape != (1, 256, 19)
        or not embedding.is_cuda
        or not bool(torch.isfinite(embedding).all())
    ):
        raise ValueError("Invalid GPGait inference output")
    vector = embedding.reshape(-1).cpu().numpy().astype(np.float32)
    if np.linalg.norm(vector) < 1e-12:
        raise ValueError("Zero GPGait embedding")
    if args.audit_output:
        after = state_digest(model)
        delta = float((embedding - repeat).abs().max())
        frozen = all(not p.requires_grad for p in model.parameters())
        if (
            before != after
            or not frozen
            or not torch.allclose(embedding, repeat, atol=1e-7, rtol=0)
        ):
            raise RuntimeError("Frozen/repeatability audit failed")
        report = dict(
            variant=args.variant,
            config=config,
            input_shape=list(array.shape),
            output_shape=list(embedding.shape),
            state_entries=len(model.state_dict()),
            parameters=sum(p.numel() for p in model.parameters()),
            state_sha256_before=before,
            state_sha256_after=after,
            frozen=frozen,
            training=model.training,
            repeat_max_delta=delta,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            device=str(x.device),
            gpu=torch.cuda.get_device_name(0),
            two_forward_seconds=forward_seconds,
            load_and_audit_seconds=time.perf_counter() - started,
        )
        with open(args.audit_output, "x") as f:
            json.dump(report, f, indent=2, allow_nan=False)
    with open(args.output, "xb") as output:
        np.save(output, vector, allow_pickle=False)
    print(f"{args.variant} CUDA WORKER: PASSED ({vector.size} dimensions)")


if __name__ == "__main__":
    main()
