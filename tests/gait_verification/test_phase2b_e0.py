"""Phase 2B engineering fixtures; no training, fitting or biometric claims."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from gait_verification.e0 import FrozenE0, native_features
from gait_verification.e0_checkpoint import extract_e0, SHIPPED_SHA256

REPO = Path(__file__).resolve().parents[2]
SHIPPED = REPO / "outputs/checkpoints/checkpoint_epoch_46_best.pth"


def poses():
    return np.random.default_rng(42).random((83, 33, 3))


def test_feature_parity_joint_order_and_last_difference():
    from utils.pose_extraction import GaitFeatureExtractor

    p = poses()
    old = GaitFeatureExtractor.__new__(GaitFeatureExtractor)
    old.sequence_length = 60
    resampled = old._normalize_sequence_length(p)
    f = old.compute_gait_features(resampled)
    expected = np.concatenate(
        [
            f["normalized_coords"].reshape(60, 36),
            f["joint_angles"],
            f["velocities"].reshape(60, 36),
        ],
        axis=1,
    ).astype(np.float32)
    actual = native_features(p, np.ones((83, 33), bool))
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-6)
    assert actual.shape == (60, 78)
    np.testing.assert_array_equal(actual[-1, 42:], actual[-2, 42:])
    order = [11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    hip = resampled[:, [23, 24]].mean(axis=1)
    np.testing.assert_allclose(
        actual[:, :36], (resampled[:, order] - hip[:, None]).reshape(60, 36), atol=1e-7
    )


@pytest.mark.parametrize("joint", [29, 30, 31, 32])
def test_missing_heel_toe_is_refused(joint):
    from gait_verification.skeleton_contracts import InsufficientGait

    mask = np.ones((83, 33), bool)
    mask[4, joint] = False
    with pytest.raises(InsufficientGait, match="LANDMARK"):
        native_features(poses(), mask)


@pytest.mark.parametrize(
    "bad", [np.zeros((60, 17, 2)), np.full((60, 33, 3), np.nan), np.zeros((9, 33, 3))]
)
def test_invalid_native_input_refused(bad):
    with pytest.raises(ValueError):
        native_features(bad, np.ones(bad.shape[:2], bool))


@pytest.fixture(scope="module")
def extracted(tmp_path_factory):
    out = tmp_path_factory.mktemp("e0") / "frozen.pt"
    evidence = extract_e0(SHIPPED, out, SHIPPED_SHA256)
    return out, evidence


def test_safe_extract_preserves_original_and_stats(extracted):
    out, evidence = extracted
    assert hashlib.sha256(SHIPPED.read_bytes()).hexdigest() == SHIPPED_SHA256
    b = torch.load(out, map_location="cpu", weights_only=True)
    assert b["feature_stats"]["mean"].shape == (78,)
    assert set(b["feature_stats"]) == {"mean", "std"}
    assert evidence["head_parameters"] == 133058
    assert b["training_origin"] == "historical-local-13-subject-shipped"
    assert "threshold" not in b and "signatures" not in b


def test_unaudited_checkpoint_refused_before_loading(tmp_path, monkeypatch):
    p = tmp_path / "unsafe.pt"
    torch.save({"danger": np.zeros(1)}, p)
    monkeypatch.setattr(
        torch, "load", lambda *a, **k: pytest.fail("must refuse before load")
    )
    with pytest.raises(ValueError, match="audited"):
        extract_e0(p, tmp_path / "out.pt", hashlib.sha256(p.read_bytes()).hexdigest())


def _unsafe_marker(path):
    Path(path).write_text("executed")


class UnsafeFixture:
    def __init__(self, marker):
        self.marker = str(marker)

    def __reduce__(self):
        return _unsafe_marker, (self.marker,)


def test_inference_never_executes_unsafe_checkpoint(tmp_path):
    marker, path = tmp_path / "marker", tmp_path / "unsafe.pt"
    torch.save(UnsafeFixture(marker), path)
    backend = FrozenE0(
        path, hashlib.sha256(path.read_bytes()).hexdigest(), device="cpu"
    )
    with pytest.raises(RuntimeError):
        backend.score(np.zeros((60, 78)), [np.zeros((60, 78))])
    assert not marker.exists()


def test_frozen_repeat_orientation_parity(extracted):
    from models.full_pipeline import create_model

    out, evidence = extracted
    cfg = json.loads((REPO / "outputs/model_config.json").read_text())
    b = torch.load(out, map_location="cpu", weights_only=True)
    old = create_model(cfg).eval().requires_grad_(False)
    # Restore only the decisive path; auxiliary modules cannot affect these logits.
    old.diff_conv.load_state_dict(
        {
            k.removeprefix("head.conv."): v
            for k, v in b["model"].items()
            if k.startswith("head.conv.")
        },
        strict=True,
    )
    old.diff_classifier.load_state_dict(
        {
            k.removeprefix("head.classifier."): v
            for k, v in b["model"].items()
            if k.startswith("head.classifier.")
        },
        strict=True,
    )
    q, r = (
        native_features(poses(), np.ones((83, 33), bool)),
        native_features(poses()[::-1].copy(), np.ones((83, 33), bool)),
    )
    stats = b["feature_stats"]
    v = (torch.from_numpy(q)[None] - stats["mean"]) / stats["std"]
    c = (torch.from_numpy(r)[None] - stats["mean"]) / stats["std"]
    with torch.inference_mode():
        logits = old(v, c)["verification"]["logits"][0]
    backend = FrozenE0(out, evidence["model_only_sha256"], device="cpu")
    score, audit = backend.score(q, [r])
    repeated, again = backend.score(q, [r])
    assert score == repeated
    assert audit["state_before"] == audit["state_after"] == again["state_after"]
    assert audit["frozen"] and not audit["training"]
    assert score == pytest.approx(float(logits[1] - logits[0]), abs=1e-5)
    assert torch.sigmoid(torch.tensor(score)).item() == pytest.approx(
        torch.softmax(logits, 0)[1].item(), abs=1e-6
    )


def test_missing_and_wrong_checkpoint(extracted, tmp_path):
    with pytest.raises(FileNotFoundError):
        FrozenE0(tmp_path / "missing", "0" * 64)
    with pytest.raises(ValueError, match="checksum"):
        FrozenE0(extracted[0], "0" * 64)


def test_multiple_reference_mean_is_order_independent(extracted):
    q = native_features(poses(), np.ones((83, 33), bool))
    refs = [q + 0.01, q - 0.02]
    backend = FrozenE0(extracted[0], extracted[1]["model_only_sha256"], device="cpu")
    a, _ = backend.score(q, refs)
    b, _ = backend.score(q, refs[::-1])
    assert a == b
    # Historical signatures average AFTER z-scoring; assert parity with that
    # decisive path rather than a raw-space mean with different float32 rounding.
    from models.verifier_variants import TCNVerifier

    bundle = torch.load(extracted[0], weights_only=True)
    model = TCNVerifier(78).eval().requires_grad_(False)
    model.load_state_dict(bundle["model"], strict=True)
    mu, sd = bundle["feature_stats"]["mean"], bundle["feature_stats"]["std"]
    v = (torch.from_numpy(q) - mu) / sd
    signature = ((torch.from_numpy(np.stack(refs)) - mu) / sd).mean(0)
    with torch.inference_mode():
        logits = model(v[None], signature[None])[0]
    assert a == pytest.approx(float(logits[1] - logits[0]), abs=1e-5)


def test_incompatible_strict_state_refused(extracted, tmp_path):
    obj = torch.load(extracted[0], weights_only=True)
    del obj["model"]["head.classifier.3.bias"]
    p = tmp_path / "bad.pt"
    torch.save(obj, p)
    backend = FrozenE0(p, hashlib.sha256(p.read_bytes()).hexdigest(), device="cpu")
    q = np.zeros((60, 78), np.float32)
    with pytest.raises(RuntimeError):
        backend.score(q, [q])


@pytest.mark.parametrize("bad", ["stats", "config", "origin"])
def test_incompatible_inference_envelope_refused(extracted, tmp_path, bad):
    obj = torch.load(extracted[0], weights_only=True)
    if bad == "stats":
        obj["feature_stats"]["std"][0] = 0
    elif bad == "config":
        obj["config"]["hidden"] = 128
    else:
        obj["training_origin"] = "unknown"
    path = tmp_path / "bad.pt"
    torch.save(obj, path)
    backend = FrozenE0(
        path, hashlib.sha256(path.read_bytes()).hexdigest(), device="cpu"
    )
    with pytest.raises(RuntimeError):
        backend.score(np.zeros((60, 78)), [np.zeros((60, 78))])


@pytest.mark.skipif(
    os.environ.get("GAIT_V2_CUDA_TESTS") != "1",
    reason="opt-in real E0 CUDA engineering",
)
def test_real_e0_cuda_parity_and_repeat(extracted):
    q = np.random.default_rng(7).normal(0, 0.02, (60, 78)).astype(np.float32)
    q[:, 36:42] = 130
    cpu, _ = FrozenE0(
        extracted[0], extracted[1]["model_only_sha256"], device="cpu"
    ).score(q, [q + 0.001])
    cuda, evidence = FrozenE0(
        extracted[0], extracted[1]["model_only_sha256"], device="cuda"
    ).score(q, [q + 0.001])
    # Different float32 reduction orders across CPU/CUDA need a relative term;
    # same-device repeats remain bit-identical (asserted independently below).
    assert cuda == pytest.approx(cpu, rel=1e-6, abs=1e-5)
    assert evidence["device"] == "cuda" and evidence["parameters"] == 133058
    assert (
        evidence["repeat_max_delta"] == 0
        and evidence["state_before"] == evidence["state_after"]
    )
