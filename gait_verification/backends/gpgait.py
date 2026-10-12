"""Audited challenger contract. Availability is explicit until weights are validated."""

from pathlib import Path
import subprocess
import numpy as np
from ..skeleton_contracts import coco_arrays, assess_quality, InsufficientGait

REVISION = "d3ce631971a01d806eeb3a154774d482c0951813"


def prepare_official_grew(sequence, source_dir):
    return _prepare_official(sequence, source_dir, "GREW")


def prepare_official_casia(sequence, source_dir):
    return _prepare_official(sequence, source_dir, "CASIA-B")


def _prepare_official(sequence, source_dir, dataset):
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
                "--dataset",
                dataset,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        native = np.load(output_path, allow_pickle=False)
    if native.shape != (6, len(xy), 17, 1) or not np.isfinite(native).all():
        raise ValueError("Invalid HOD output")
    return native[None].astype(np.float32)


class GPGaitEncoder:
    """Frozen, CUDA-only GPGait verification encoder."""

    embedding_dim = 4864
    variant = "gpgait"
    min_frames = 31  # V2 quality floor; official evaluation uses all_ordered.

    def __init__(self, source_dir, checkpoint, expected_sha256, device="cuda"):
        import hashlib
        import json
        import torch
        from ..contracts import ModelMetadata
        from ..media import sha256_file

        if device not in ("cuda", "cuda:0"):
            raise ValueError("GPGait requires CUDA device 0")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")

        self.source = Path(source_dir).resolve()
        revision = subprocess.check_output(
            ["git", "-C", str(self.source), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(self.source), "status", "--porcelain"],
            text=True,
        ).strip()
        if revision != REVISION or dirty:
            raise ValueError("Requires clean audited FastPoseGait revision")

        self.checkpoint = Path(checkpoint).resolve()
        checksum = sha256_file(self.checkpoint)
        if not expected_sha256 or checksum != expected_sha256:
            raise ValueError("GPGait checkpoint checksum mismatch")

        recipe = "grew-affine-rescalecenter-hod-skeleton-v1"
        config = {
            "in_channels": [2, 64, 128, 128, 128, 128, 256, 256],
            "num_class": 20000,
            "share_num": 3,
            "embedding_shape": [256, 19],
        }
        if self.variant == "gpgaitpp":
            config["mask"] = False
        elif self.variant == "gpgaitpp_casia":
            recipe = "casia-rescalecenter-hod-skeleton-v1"
            config.update(in_channels=[2, 64, 64, 128, 256], num_class=74, mask=True)
        fingerprint = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()[:12]

        self.metadata = ModelMetadata(
            model_id=f"{self.variant}:{checksum}:{recipe}:{fingerprint}",
            repository="https://github.com/BNU-IVC/FastPoseGait",
            revision=revision,
            checkpoint_sha256=checksum,
            checkpoint_bytes=self.checkpoint.stat().st_size,
            training_dataset="CASIA-B" if self.variant == "gpgaitpp_casia" else "GREW",
            input_contract=recipe,
            license="FastPoseGait academic/noncommercial; deployment rights unverified",
            readiness="BLOCKED",
        )
        self.device = "cuda:0"

    def encode(self, sequence, *, audit_path=None):
        import hashlib
        import sys
        import tempfile
        from ..contracts import GaitEmbedding

        tensor = (
            prepare_official_casia(sequence, self.source)
            if self.variant == "gpgaitpp_casia"
            else prepare_official_grew(sequence, self.source)
        )

        with tempfile.TemporaryDirectory(prefix="gpgait-inference-") as work:
            input_path = Path(work) / "input.npy"
            output_path = Path(work) / "embedding.npy"
            np.save(input_path, tensor, allow_pickle=False)

            worker = Path(__file__).with_name("fastpose_infer_worker.py")
            command = [
                sys.executable,
                str(worker),
                "--source",
                str(self.source),
                "--checkpoint",
                str(self.checkpoint),
                "--sha256",
                self.metadata.checkpoint_sha256,
                "--variant",
                self.variant,
                "--input",
                str(input_path),
                "--output",
                str(output_path),
            ]
            if audit_path is not None:
                command += ["--audit-output", str(Path(audit_path).resolve())]
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=180,
            )

            vector = np.load(output_path, allow_pickle=False)

        if vector.shape != (self.embedding_dim,) or not np.isfinite(vector).all():
            raise ValueError("Invalid GPGait embedding")

        pose_recipe = sequence.backend_id.split(":display-rotation=", 1)[0]
        pipeline = hashlib.sha256(
            f"{pose_recipe}:display-orientation-v1:{sequence.variant_id}".encode()
        ).hexdigest()[:16]

        embedding = GaitEmbedding(
            values=vector,
            model_id=self.metadata.model_id + ":pipeline=" + pipeline,
            original_take_id=sequence.original_take_id,
            source_sha256=sequence.source_sha256,
        )

        return GaitEmbedding(
            values=embedding.normalized(),
            model_id=embedding.model_id,
            original_take_id=embedding.original_take_id,
            source_sha256=embedding.source_sha256,
        )


class GPGaitPlusPlusEncoder(GPGaitEncoder):
    """Independent official GREW GPGait++ candidate, never a CASIA substitute."""

    variant = "gpgaitpp"


class GPGaitPlusPlusCASIAEncoder(GPGaitEncoder):
    """CASIA-B engineering candidate; not a matched GREW-trained comparator."""

    variant = "gpgaitpp_casia"
