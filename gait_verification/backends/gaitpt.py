"""Load the unmodified official model, without its training/data-loading entrypoint."""

from pathlib import Path
from types import SimpleNamespace
import hashlib
import importlib.util
import json
import subprocess
import numpy as np
from ..contracts import ModelMetadata, GaitEmbedding
from ..media import sha256_file
from ..skeleton_contracts import gaitpt_grew_inputs

REVISION = "4977618a36ef77ee9a2863c9f8e27ab862894d70"


def load_source(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GaitPTEncoder:
    min_frames = 31  # Official GREW loader excludes sequences <=30 frames.

    def __init__(self, source_dir, checkpoint, expected_sha256, device="cuda"):
        import torch
        import yaml

        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no CPU substitution")

        source = Path(source_dir).resolve()
        revision = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        if (
            revision != REVISION
            or subprocess.check_output(
                ["git", "-C", str(source), "status", "--porcelain"], text=True
            ).strip()
        ):
            raise ValueError("Requires clean audited GaitPT revision")
        checkpoint = Path(checkpoint)
        checksum = sha256_file(checkpoint)
        if not expected_sha256 or checksum != expected_sha256:
            raise ValueError("Checkpoint checksum mismatch")
        with open(source / "configs/gaitpt_config.yaml") as f:
            config = yaml.safe_load(f)
        config["sequence_length"] = 30
        recipe = "grew-two-center30-height-v1"
        fingerprint = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()[:12]
        model_id = f"gaitpt:{checksum}:{recipe}:{fingerprint}"
        self.metadata = ModelMetadata(
            model_id,
            "https://github.com/UbiquitousCognitiveSystems/GaitPT",
            revision,
            checksum,
            checkpoint.stat().st_size,
            "GREW",
            recipe,
            "CC-BY-NC-ND-4.0",
            "BLOCKED",
        )  # Source-dataset sample + visual Android validation still required.
        model_class = load_source(source / "gaitpt.py", "_official_gaitpt_v2").GaitPT
        model = model_class(SimpleNamespace(**config))
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        # Official checkpoint is DataParallel.state_dict(); accept only that exact form.
        state = {k.removeprefix("module."): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        self.model = model.to(device).eval().requires_grad_(False)
        self.device = device
        self.embedding_dim = None

    def encode(self, sequence):
        import torch

        batch = gaitpt_grew_inputs(sequence)
        self.model.eval()
        with torch.inference_mode():
            z = self.model(torch.from_numpy(batch).to(self.device)).cpu().numpy()
        if z.ndim != 2 or z.shape[0] != 2 or not np.isfinite(z).all():
            raise ValueError("Unexpected official embedding output")
        if self.embedding_dim is not None and z.shape[1] != self.embedding_dim:
            raise ValueError("Embedding dimension changed")
        self.embedding_dim = z.shape[1]
        norms = np.linalg.norm(z, axis=1, keepdims=True)
        if (norms < 1e-12).any():
            raise ValueError("Zero window embedding")
        # Enrollment compatibility binds pose estimator and raw/stabilized recipe
        # as well as checkpoint/native preprocessing. Display rotation itself is
        # per-file metadata; the orientation convention is constant.
        pose_recipe = sequence.backend_id.split(":display-rotation=", 1)[0]
        pipeline = hashlib.sha256(
            f"{pose_recipe}:display-orientation-v1:{sequence.variant_id}".encode()
        ).hexdigest()[:16]
        result = GaitEmbedding(
            (z / norms).mean(0),
            self.metadata.model_id + ":pipeline=" + pipeline,
            sequence.original_take_id,
            sequence.source_sha256,
        )
        return GaitEmbedding(
            result.normalized(),
            result.model_id,
            result.original_take_id,
            result.source_sha256,
        )
