"""Small frozen E0 interface; legacy imports/deserialization stay outside V2."""

import json
from contextlib import nullcontext
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from .media import sha256_file
from .skeleton_contracts import InsufficientGait

JOINTS = (11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32)
JOINT_NAMES = (
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)
AGGREGATION = "mean-zscored-reference-feature-signature-v1"
SCORE_SEMANTICS = "class1-minus-class0-logit; higher supports claim; uncalibrated"


def worker(mode, *args):
    script = (
        Path(__file__).resolve().parents[1] / "scripts/gait_verification/e0_worker.py"
    )
    result = subprocess.run(
        [sys.executable, str(script), mode, *map(str, args)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode:
        raise RuntimeError("E0_WORKER_FAILED")  # no paths or private data in errors
    return json.loads(result.stdout)


def validate_features(values):
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (60, 78) or not np.isfinite(values).all():
        raise ValueError("INVALID_E0_60X78_FEATURES")
    return values


def native_features(pose, observed):
    """Full MediaPipe33 normalized xyz; no COCO mapping, padding or gap filling."""
    pose, observed = np.asarray(pose), np.asarray(observed)
    if pose.ndim != 3 or pose.shape[1:] != (33, 3) or len(pose) < 10:
        raise ValueError("E0 requires at least ten native MediaPipe33 xyz poses")
    if observed.shape != pose.shape[:2] or observed.dtype != bool:
        raise ValueError("E0 requires an explicit anatomical observation mask")
    if not observed[:, JOINTS].all():
        raise InsufficientGait("MISSING_ANATOMICAL_LANDMARK")
    if not np.isfinite(pose[:, JOINTS]).all():
        raise ValueError("NONFINITE_E0_POSE")
    with tempfile.TemporaryDirectory() as td:
        inp, out = Path(td) / "input.npz", Path(td) / "features.npy"
        np.savez(inp, pose=pose)
        worker("features", "--input", inp, "--output", out)
        return validate_features(np.load(out, allow_pickle=False))


class FrozenE0:
    """Hash-pinned historical temporal head; no threshold, fitting or gallery."""

    def __init__(self, checkpoint, checksum, device="cuda"):
        self.checkpoint = Path(checkpoint)
        if sha256_file(self.checkpoint) != checksum:
            raise ValueError("E0 checkpoint checksum mismatch")
        if device not in ("cpu", "cuda"):
            raise ValueError("Explicit cpu or cuda device required")
        self.checksum, self.device = checksum, device
        self.model_id = f"e0:{checksum}:native60x78-v1:{AGGREGATION}"

    def score(self, query, references):
        q = validate_features(query)
        refs = [validate_features(r) for r in references]
        if not refs:
            raise ValueError("E0 requires independent claimed references")
        # Canonical tensor ordering makes averaging independent of request order.
        refs.sort(key=lambda r: r.tobytes())
        from .dataset_processing import default_gpu_lock_dir, gpu_slot

        slot = (
            gpu_slot(default_gpu_lock_dir()) if self.device == "cuda" else nullcontext()
        )
        with slot, tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "pair.npz"
            np.savez(inp, query=q, references=np.stack(refs))
            result = worker(
                "score",
                "--input",
                inp,
                "--checkpoint",
                self.checkpoint,
                "--sha256",
                self.checksum,
                "--device",
                self.device,
            )
        score = result["score"]
        if not isinstance(score, (int, float)) or not np.isfinite(score):
            raise ValueError("NONFINITE_E0_SCORE")
        return float(score), result["audit"]
