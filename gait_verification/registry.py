"""Build the frozen encoders and RTMPose runtime from the pinned model registry.

No paths are hardcoded: roots are arguments, the registry is the repository's own
`configs/gait_verification/models.json`. Checksums come from that registry; every
encoder refuses a mismatching checkpoint, a dirty/foreign source revision, and CPU.
"""

import json
from pathlib import Path
from .pose_quality import WindowQualityPolicy, WindowedEncoder

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("gaitpt", "gpgait", "gpgaitpp", "gpgaitpp_casia")
# CLI model name -> entry in configs/gait_verification/models.json
REGISTRY_KEYS = {
    "gaitpt": "gaitpt_grew",
    "gpgait": "gpgait",
    "gpgaitpp": "gpgaitpp_grew",
    "gpgaitpp_casia": "gpgaitpp_casia",
}


def load_json(relative):
    return json.loads((ROOT / relative).read_text())


def load_quality_policy(config="configs/gait_verification/phase1a.json"):
    return WindowQualityPolicy(**load_json(config)["quality"])


def model_entry(name):
    """Registry facts for one candidate (no weights loaded)."""
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}; choose from {MODELS}")
    return load_json("configs/gait_verification/models.json")[REGISTRY_KEYS[name]]


def load_encoder(name, source_root, checkpoint_root, policy=None):
    """Frozen CUDA encoder wrapped in the versioned window-aggregation recipe."""
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}; choose from {MODELS}")
    from .backends.gaitpt import GaitPTEncoder
    from .backends.gpgait import (
        GPGaitEncoder,
        GPGaitPlusPlusCASIAEncoder,
        GPGaitPlusPlusEncoder,
    )

    registry = load_json("configs/gait_verification/models.json")
    cls, source, filename, sha = {
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
        "gpgaitpp": (
            GPGaitPlusPlusEncoder,
            "FastPoseGait",
            "gpgaitpp-grew-model-only.pt",
            registry["gpgaitpp_grew"]["model_only_sha256"],
        ),
        "gpgaitpp_casia": (
            GPGaitPlusPlusCASIAEncoder,
            "FastPoseGait",
            "gpgaitpp-casia-model-only.pt",
            registry["gpgaitpp_casia"]["model_only_sha256"],
        ),
    }[name]
    encoder = cls(Path(source_root) / source, Path(checkpoint_root) / filename, sha)
    return WindowedEncoder(encoder, policy or load_quality_policy())


def load_rtmpose(rtmpose_dir, config="configs/gait_verification/phase1a.json"):
    """Checksum-pinned CUDA RTMPose (explicit local ONNX files, no downloads)."""
    from .pose import RTMPoseDetector

    spec = load_json(config)["rtmpose"]
    directory = Path(rtmpose_dir).expanduser()
    return RTMPoseDetector(
        str(directory / spec["detector"]["filename"]),
        str(directory / spec["pose"]["filename"]),
        expected_detector_sha256=spec["detector"]["sha256"],
        expected_pose_sha256=spec["pose"]["sha256"],
    )
