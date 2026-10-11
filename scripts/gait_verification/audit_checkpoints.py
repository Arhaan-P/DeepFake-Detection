"""Read-only checkpoint inventory; safe deserialization failures remain blockers."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gait_verification.media import sha256_file


def inspect(path):
    import torch

    path = Path(path)
    row = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "model_weights_loaded": False,
        "forward_pass": False,
    }
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
        row["deserialization"] = "OK"
        row["top_level_keys"] = list(obj)[:12]
    except Exception as exc:
        row["deserialization"] = "BLOCKED"
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoints", nargs="+")
    args = p.parse_args()
    print(json.dumps([inspect(path) for path in args.checkpoints], indent=2))
