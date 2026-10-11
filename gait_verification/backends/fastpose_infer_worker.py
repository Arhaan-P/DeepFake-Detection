"""Isolated, CUDA-only inference using the pinned FastPoseGait architecture."""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPGait inference")

    source = Path(args.source).resolve()
    sys.path.insert(0, str(source / "fastposegait"))

    from modeling.models.gpgait import gpgait

    array = np.load(args.input, allow_pickle=False)

    if (
        array.ndim != 5
        or array.shape[0] != 1
        or array.shape[1] != 6
        or array.shape[2] < 31
        or array.shape[3:] != (17, 1)
        or not np.isfinite(array).all()
    ):
        raise ValueError("Invalid GPGait input tensor")

    config = {
        "in_channels": [2, 64, 128, 128, 128, 128, 256, 256],
        "num_class": 20000,
        "share_num": 3,
    }

    # Bypass BaseModel.__init__, which starts the training/test framework.
    model = gpgait.__new__(gpgait)
    torch.nn.Module.__init__(model)
    model.build_network(config)

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    if set(checkpoint) != {"model"}:
        raise ValueError("Expected a model-only checkpoint")

    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint

    model = model.to("cuda").eval().requires_grad_(False)

    x = torch.from_numpy(array.astype(np.float32)).to("cuda")
    seq_lengths = torch.tensor([[array.shape[2]]], dtype=torch.int32)

    with torch.inference_mode():
        result = model(([x], None, None, None, seq_lengths))
        embedding = result["inference_feat"]["embeddings"]

    if (
        embedding.shape != (1, 256, 19)
        or not embedding.is_cuda
        or not bool(torch.isfinite(embedding).all())
    ):
        raise ValueError("Invalid GPGait inference output")

    vector = embedding.reshape(-1).cpu().numpy().astype(np.float32)

    if np.linalg.norm(vector) < 1e-12:
        raise ValueError("Zero GPGait embedding")

    with open(args.output, "xb") as output:
        np.save(output, vector, allow_pickle=False)

    print("GPGAIT WORKER: PASSED")
    print("Device:", x.device)
    print("Embedding dimensions:", vector.size)


if __name__ == "__main__":
    main()
