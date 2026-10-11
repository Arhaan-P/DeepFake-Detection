"""Engineering smoke only: real weights + synthetic poses, or an explicit pose cache."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from gait_verification.backends.gaitpt import GaitPTEncoder
from gait_verification.contracts import PoseSequence
from gait_verification.pose import load_pose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--pose-cache")
    args = parser.parse_args()
    torch.set_num_threads(2)
    encoder = GaitPTEncoder(args.source, args.checkpoint, args.sha256)
    if args.pose_cache:
        sequence = load_pose(args.pose_cache)
    else:
        n = 90
        xy = (
            np.broadcast_to(
                np.stack([100 + np.arange(17) * 3, 50 + np.arange(17) * 5], -1),
                (n, 17, 2),
            )
            .copy()
            .astype(float)
        )
        xy[..., 0] += np.sin(np.arange(n)[:, None] / 10)
        sequence = PoseSequence(
            xy,
            np.ones((n, 17)),
            np.ones((n, 17), bool),
            np.arange(n) / 30,
            "coco17",
            640,
            480,
            "synthetic-take",
            "synthetic-source",
            "synthetic",
            np.arange(n),
            np.zeros(n, int),
            np.tile([100, 50, 200, 200], (n, 1)),
        )
    before = {
        name: tensor.clone() for name, tensor in encoder.model.state_dict().items()
    }
    a, b = encoder.encode(sequence), encoder.encode(sequence)
    unchanged = all(
        torch.equal(tensor, before[name])
        for name, tensor in encoder.model.state_dict().items()
    )
    assert unchanged and all(not p.requires_grad for p in encoder.model.parameters())
    assert np.allclose(a.values, b.values, atol=1e-7) and np.isclose(
        np.linalg.norm(a.values), 1
    )
    print(
        json.dumps(
            {
                "fixture": "historical_pose_cache"
                if args.pose_cache
                else "synthetic_anatomical_markers_NOT_source_dataset",
                "metadata": asdict(encoder.metadata),
                "embedding_shape": list(a.values.shape),
                "embedding_norm": float(np.linalg.norm(a.values)),
                "max_repeat_delta": float(np.max(np.abs(a.values - b.values))),
                "parameters": sum(p.numel() for p in encoder.model.parameters()),
                "weights_unchanged": unchanged,
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
