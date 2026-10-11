"""Audited challenger contract. Availability is explicit until weights are validated."""

from pathlib import Path
import subprocess
import numpy as np
from ..skeleton_contracts import coco_arrays, assess_quality, InsufficientGait

REVISION = "d3ce631971a01d806eeb3a154774d482c0951813"


def prepare_official_grew(sequence, source_dir):
    source = Path(source_dir).resolve()
    rev = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if (
        rev != REVISION
        or subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain"], text=True
        ).strip()
    ):
        raise ValueError("Requires clean audited FastPoseGait revision")
    quality = assess_quality(sequence)
    if not quality.usable:
        raise InsufficientGait(",".join(quality.reasons))
    xy, confidence, _ = coco_arrays(sequence)
    if (np.ptp(xy[..., 1], axis=1) <= 0).any():
        raise ValueError("Degenerate skeleton height")
    native = np.concatenate([xy, confidence[..., None]], axis=-1).astype(np.float32)
    # Official source imports top-level `utils` and `data`; isolate namespace.
    import sys
    import tempfile

    with tempfile.TemporaryDirectory(prefix="gait-v2-transform-") as work:
        input_path, output_path = Path(work) / "input.npy", Path(work) / "output.npy"
        np.save(input_path, native, allow_pickle=False)
        worker = Path(__file__).with_name("fastpose_worker.py")
        subprocess.run(
            [
                sys.executable,
                str(worker),
                "--source",
                str(source),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        native = np.load(output_path, allow_pickle=False)
    if native.shape != (6, len(xy), 17, 1) or not np.isfinite(native).all():
        raise ValueError("Invalid HOD output")
    return native[None].astype(np.float32)


class GPGaitEncoder:
    """Fail closed; do not silently substitute random weights or GaitPT embeddings."""

    embedding_dim = None

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "BLOCKED: FastPoseGait checkpoint loading and isolated inference not validated; see Phase 0 report"
        )
