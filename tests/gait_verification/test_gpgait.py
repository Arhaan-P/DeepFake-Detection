"""Regression tests for the isolated GPGait encoder interface."""

import hashlib
import subprocess

import numpy as np
import pytest

from gait_verification.contracts import PoseSequence
from gait_verification.backends import gpgait


def make_sequence():
    n = 40
    return PoseSequence(
        xy=np.ones((n, 17, 2), dtype=np.float64) * 100,
        confidence=np.ones((n, 17), dtype=np.float32),
        observed=np.ones((n, 17), dtype=bool),
        timestamps=np.arange(n, dtype=np.float64) / 30,
        layout="coco17",
        width=1080,
        height=1920,
        original_take_id="test-original",
        source_sha256="test-source-hash",
        backend_id="rtmpose-test:display-rotation=0",
        frame_indices=np.arange(n),
        track_ids=np.ones(n, dtype=np.int64),
        bboxes=np.zeros((n, 4)),
    )


@pytest.fixture
def encoder_setup(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"test-checkpoint-fixture")
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    def git_response(command, **kwargs):
        if "rev-parse" in command:
            return gpgait.REVISION + "\n"
        if "status" in command:
            return ""
        raise AssertionError("Unexpected git command")

    monkeypatch.setattr(gpgait.subprocess, "check_output", git_response)

    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    return tmp_path, checkpoint, checksum


def test_cuda_only(encoder_setup):
    source, checkpoint, checksum = encoder_setup

    with pytest.raises(ValueError, match="CUDA"):
        gpgait.GPGaitEncoder(
            source, checkpoint, checksum, device="cpu"
        )


def test_checkpoint_checksum_must_match(encoder_setup):
    source, checkpoint, _ = encoder_setup

    with pytest.raises(ValueError, match="checksum mismatch"):
        gpgait.GPGaitEncoder(
            source, checkpoint, "0" * 64, device="cuda"
        )


def test_embedding_is_normalized_and_provenance_preserved(
    encoder_setup, monkeypatch
):
    source, checkpoint, checksum = encoder_setup

    monkeypatch.setattr(
        gpgait,
        "prepare_official_grew",
        lambda sequence, source_dir: np.ones(
            (1, 6, 40, 17, 1), dtype=np.float32
        ),
    )

    def fake_worker(command, **kwargs):
        output_path = command[command.index("--output") + 1]
        vector = np.arange(1, 4865, dtype=np.float32)
        np.save(output_path, vector, allow_pickle=False)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(gpgait.subprocess, "run", fake_worker)

    encoder = gpgait.GPGaitEncoder(
        source, checkpoint, checksum, device="cuda"
    )
    sequence = make_sequence()
    result = encoder.encode(sequence)

    assert result.values.shape == (4864,)
    assert np.isfinite(result.values).all()
    assert np.isclose(np.linalg.norm(result.values), 1.0, atol=1e-6)
    assert result.original_take_id == sequence.original_take_id
    assert result.source_sha256 == sequence.source_sha256
    assert result.model_id.startswith(encoder.metadata.model_id + ":pipeline=")


def test_invalid_worker_embedding_is_rejected(encoder_setup, monkeypatch):
    source, checkpoint, checksum = encoder_setup

    monkeypatch.setattr(
        gpgait,
        "prepare_official_grew",
        lambda sequence, source_dir: np.ones(
            (1, 6, 40, 17, 1), dtype=np.float32
        ),
    )

    def fake_worker(command, **kwargs):
        output_path = command[command.index("--output") + 1]
        np.save(
            output_path,
            np.full(4864, np.nan, dtype=np.float32),
            allow_pickle=False,
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(gpgait.subprocess, "run", fake_worker)

    encoder = gpgait.GPGaitEncoder(
        source, checkpoint, checksum, device="cuda"
    )

    with pytest.raises(ValueError, match="Invalid GPGait embedding"):
        encoder.encode(make_sequence())
