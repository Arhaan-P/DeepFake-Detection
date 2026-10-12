"""Isolated official FastPoseGait preprocessing (its top-level utils/data collide with E0)."""

import argparse
from pathlib import Path
import sys
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dataset", choices=("GREW", "CASIA-B"), default="GREW")
    args = p.parse_args()
    sys.path.insert(0, str(Path(args.source).resolve() / "fastposegait"))
    from data.transform import Affine, RescaleCenter, HOD_MultiInput, SkeletonInput

    native = np.load(args.input, allow_pickle=False)
    transforms = (RescaleCenter(), HOD_MultiInput(), SkeletonInput())
    if args.dataset == "GREW":
        transforms = (Affine(),) + transforms
    for transform in transforms:
        native = transform(native)
    with open(args.output, "xb") as f:
        np.save(f, native, allow_pickle=False)


if __name__ == "__main__":
    main()
