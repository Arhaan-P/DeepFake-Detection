"""Phase 1B contracts and security tests; mocked weights are not biometric evidence."""

from dataclasses import replace
import hashlib
import io
from pathlib import Path
import pickle
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gait_verification.backends import gpgait, gaitpt
from gait_verification.backends.checkpoint_extract import (
    extract_model_only,
    restricted_model_pickle,
    static_pickle_audit,
)
from gait_verification.backends.fastpose_infer_worker import load_model, validate_input
from gait_verification.contracts import (
    GaitEmbedding,
    QualityAssessment,
    Status,
)
from gait_verification.enrollments import (
    EnrollmentTemplate,
    MemoryEnrollmentStore,
    validate_templates,
)
from gait_verification.pose_quality import (
    WindowedEncoder,
    slice_pose,
)
from gait_verification.skeleton_contracts import InsufficientGait, gaitpt_grew_inputs
from gait_verification.verifier import ClaimVerifier, VerificationPolicy
from test_foundation import recording
import test_gpgait as gpgait_tests
from test_gpgait import make_sequence

from test_pilot import moving_pose

encoder_setup = gpgait_tests.encoder_setup


@pytest.mark.parametrize("cls", [gpgait.GPGaitEncoder, gpgait.GPGaitPlusPlusEncoder])
def test_cuda_unavailable_fails_before_loading(encoder_setup, monkeypatch, cls):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        cls(*encoder_setup)


@pytest.mark.parametrize("response", ["wrong-revision", "dirty"])
def test_revision_and_dirty_source_rejected(encoder_setup, monkeypatch, response):
    def git(command, **kwargs):
        if "rev-parse" in command:
            return gpgait.REVISION if response == "dirty" else response
        return " M model.py" if response == "dirty" else ""

    monkeypatch.setattr(subprocess, "check_output", git)
    with pytest.raises(ValueError, match="audited"):
        gpgait.GPGaitPlusPlusEncoder(*encoder_setup)


def test_independent_backbone_ids_and_worker_failures(encoder_setup, monkeypatch):
    a = gpgait.GPGaitEncoder(*encoder_setup)
    b = gpgait.GPGaitPlusPlusEncoder(*encoder_setup)
    assert a.metadata.model_id != b.metadata.model_id
    monkeypatch.setattr(
        gpgait, "prepare_official_grew", lambda *a: np.ones((1, 6, 40, 17, 1))
    )

    def fail(cmd, **kwargs):
        assert "--sha256" in cmd and "gpgaitpp" in cmd and kwargs["timeout"] == 180
        raise subprocess.CalledProcessError(1, cmd, stderr="private worker traceback")

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        b.encode(make_sequence())
    # Exercise service failure mapping with a test-only READY wrapper.
    b.metadata = replace(b.metadata, readiness="READY")

    class Store:
        calls = []

        def get_claim(self, claim):
            self.calls.append(claim)
            return (object(),)

    s = Store()
    result = ClaimVerifier(
        s, b, SimpleNamespace(extract=lambda _: moving_pose())
    ).verify(__file__, "S1")
    assert result.status == Status.PROCESSING_ERROR and result.verdict is None
    assert s.calls == ["S1"] and "private" not in str(result.public_dict())


@pytest.mark.parametrize(
    "shape",
    [(1, 5, 31, 17, 1), (1, 6, 30, 17, 1), (1, 6, 31, 18, 1), (2, 6, 31, 17, 1)],
)
def test_worker_wrong_tensor_contract(shape):
    with pytest.raises(ValueError, match="input"):
        validate_input(np.ones(shape))


def test_worker_nan_and_minimum_input():
    x = np.ones((1, 6, 31, 17, 1))
    validate_input(x)
    x[0, 0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        validate_input(x)


@pytest.mark.parametrize("fault", ["short", "missing", "cropped", "tracking", "gap"])
def test_invalid_pose_rejected_before_fastpose_worker(
    encoder_setup, monkeypatch, fault
):
    s = moving_pose()
    if fault == "short":
        s = slice_pose(s, 0, 30)
    elif fault == "missing":
        s.observed[20, 15] = False
    elif fault == "cropped":
        s.xy[20, 15, 1] = s.height
    elif fault == "tracking":
        s.track_ids[60:] = 1
    else:
        s.timestamps[60:] += 2

    def forbidden(*a, **kw):
        raise AssertionError("Rejected input reached worker")

    monkeypatch.setattr(subprocess, "run", forbidden)
    with pytest.raises(InsufficientGait):
        gpgait.prepare_official_grew(s, encoder_setup[0])


def test_native_scores_masks_and_pts_unchanged_by_preprocessing():
    s = moving_pose(31)
    s.confidence[:] = 1.12
    before = [a.copy() for a in (s.xy, s.confidence, s.observed, s.timestamps)]
    assert gaitpt_grew_inputs(s).shape == (2, 30, 18, 2)
    for a, b in zip(before, (s.xy, s.confidence, s.observed, s.timestamps)):
        np.testing.assert_array_equal(a, b)


def test_window_encoder_minimum_is_not_assumed_universal():
    e = SimpleNamespace(min_frames=60)
    with pytest.raises(ValueError, match="encoder minimum"):
        WindowedEncoder(e)


@pytest.mark.parametrize(
    "mismatch",
    ["gpgaitpp:checkpoint", "gaitpt:changed-preprocessing", "gaitpt:changed-pose"],
)
def test_cross_model_or_recipe_enrollment_rejected(mismatch):
    a = GaitEmbedding(np.array([1.0, 0.0]), "gaitpt:checkpoint", "r1", "r1-hash")
    b = replace(a, model_id=mismatch, original_take_id="r2", source_sha256="r2-hash")
    store = MemoryEnrollmentStore()
    # Phase 1C: one claim may hold several recipes, but they are never pooled.
    with pytest.raises(ValueError, match="Mixed enrollment recipes"):
        validate_templates(
            "S1",
            [
                EnrollmentTemplate("S1", a, recording()),
                EnrollmentTemplate("S1", b, recording(take="r2")),
            ],
        )
    store.enroll("S1", [EnrollmentTemplate("S1", a, recording())])
    v = ClaimVerifier(
        store,
        policy=VerificationPolicy(a.model_id, 0.5, test_only=True),
        allow_test_policy=True,
    )
    result = v.verify_embedding(b, "S1", QualityAssessment(True))
    assert result.status == Status.CLAIM_NOT_ENROLLED and result.verdict is None


def test_unaudited_checkpoint_rejected_without_deserialization(tmp_path, monkeypatch):
    p = tmp_path / "untrusted.pt"
    p.write_bytes(b"not a checkpoint")
    monkeypatch.setattr(
        torch, "load", lambda *a, **kw: pytest.fail("must not deserialize")
    )
    with pytest.raises(ValueError, match="audited"):
        extract_model_only(
            p, tmp_path / "out.pt", hashlib.sha256(p.read_bytes()).hexdigest()
        )
    assert not (tmp_path / "out.pt").exists()


def test_prefix_extraction_does_not_execute_training_tail():
    # This serializes a global reference but never executes it. The original
    # object would fail restricted loading; the tail is never deserialized here.
    state = torch.nn.Linear(2, 2).state_dict()
    import zipfile

    b = io.BytesIO()
    torch.save({"model": state, "optimizer": {"unsafe": getattr}}, b)
    with zipfile.ZipFile(b) as z:
        data = z.read("archive/data.pkl")
    import pickletools

    end = next(
        p
        for o, a, p in pickletools.genops(data)
        if o.name == "BINUNICODE" and a == "optimizer"
    )
    spec = dict(pickle_sha256=hashlib.sha256(data).hexdigest(), model_end=end)
    prefix = restricted_model_pickle(data, spec)
    assert b"getattr" not in prefix
    with pytest.raises(pickle.UnpicklingError):
        torch.load(io.BytesIO(b.getvalue()), weights_only=True)
    clean = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(b.getvalue())) as original:
        with zipfile.ZipFile(clean, "w") as archive:
            for name in original.namelist():
                archive.writestr(
                    name, prefix if name == "archive/data.pkl" else original.read(name)
                )
    clean.seek(0)
    recovered = torch.load(clean, weights_only=True)
    assert set(recovered) == {"model"}
    assert all(torch.equal(v, recovered["model"][k]) for k, v in state.items())
    with pytest.raises(ValueError, match="pickle bytes"):
        restricted_model_pickle(data + b"x", spec)
    with pytest.raises(ValueError, match="boundary"):
        restricted_model_pickle(data, dict(spec, model_end=end - 1))
    bad = pickle.dumps({"model": getattr, "optimizer": {}}, protocol=2)
    bad_end = next(p for o, a, p in pickletools.genops(bad) if a == "optimizer")
    with pytest.raises(ValueError):
        restricted_model_pickle(
            bad, dict(pickle_sha256=hashlib.sha256(bad).hexdigest(), model_end=bad_end)
        )


def test_static_audit_locates_boundary_without_deserializing(tmp_path, monkeypatch):
    path = tmp_path / "training.pt"
    torch.save(
        {"model": torch.nn.Linear(2, 2).state_dict(), "optimizer": {"x": getattr}},
        path,
    )
    monkeypatch.setattr(
        torch, "load", lambda *a, **kw: pytest.fail("must not deserialize")
    )
    audit = static_pickle_audit(path)
    assert audit["head"][4] == ("BINUNICODE", "model")
    assert ["model", "optimizer"] == [a for _, a in audit["key_strings"]]
    assert audit["candidate_model_end"] == audit["key_strings"][1][0]
    assert "__builtin__ getattr" in audit["all_globals"]
    assert audit["model_section_globals_allowed"] is True


def test_worker_checkpoint_rechecked_before_import(tmp_path, monkeypatch):
    import gait_verification.backends.fastpose_infer_worker as worker

    p = tmp_path / "weights.pt"
    p.write_bytes(b"changed after parent construction")
    monkeypatch.setattr(worker, "validate_source", lambda p: p)
    with pytest.raises(ValueError, match="checksum"):
        load_model(tmp_path, p, "0" * 64, "gpgaitpp")


def test_gaitpt_strict_frozen_initialization(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/gaitpt_config.yaml").write_text("projection_size: 2\n")
    p = tmp_path / "weights.pt"
    model = torch.nn.Linear(2, 2)
    torch.save({"module." + k: v for k, v in model.state_dict().items()}, p)
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda c, **kw: gaitpt.REVISION if "rev-parse" in c else "",
    )
    monkeypatch.setattr(
        gaitpt,
        "load_source",
        lambda *a: SimpleNamespace(GaitPT=lambda config: torch.nn.Linear(2, 2)),
    )
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    e = gaitpt.GaitPTEncoder(tmp_path, p, sha, device="cpu")
    assert not e.model.training and all(
        not v.requires_grad for v in e.model.parameters()
    )
    for k, v in model.state_dict().items():
        assert torch.equal(e.model.state_dict()[k], v)
    torch.save({"module.wrong": torch.ones(1)}, p)
    with pytest.raises(RuntimeError, match="state_dict"):
        gaitpt.GaitPTEncoder(
            tmp_path, p, hashlib.sha256(p.read_bytes()).hexdigest(), device="cpu"
        )


def test_gaitpt_default_cuda_never_falls_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA unavailable"):
        gaitpt.GaitPTEncoder("missing-source", "missing-checkpoint", "hash")


def test_fastpose_exact_config_and_strict_tensor_loading(tmp_path, monkeypatch):
    import gait_verification.backends.fastpose_infer_worker as worker

    class FixtureModel(torch.nn.Module):
        def build_network(self, config):
            assert config == {"model": "GPGait_PlusPlus", "mask": False}
            self.layer = torch.nn.Linear(2, 3)

    monkeypatch.setattr(worker, "validate_source", lambda _: tmp_path)
    monkeypatch.setattr(
        worker,
        "official_config",
        lambda *a: {"model": "GPGait_PlusPlus", "mask": False},
    )
    monkeypatch.setattr(
        worker.importlib,
        "import_module",
        lambda name: SimpleNamespace(GPGait_PlusPlus=FixtureModel),
    )
    fixture = FixtureModel()
    fixture.build_network({"model": "GPGait_PlusPlus", "mask": False})
    path = tmp_path / "fixture.pt"
    torch.save({"model": fixture.state_dict()}, path)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    model, config = worker.load_model(tmp_path, path, checksum, "gpgaitpp")
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    for k, v in fixture.state_dict().items():
        assert torch.equal(model.state_dict()[k], v)
    state = dict(fixture.state_dict())
    del state["layer.bias"]
    torch.save({"model": state}, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        worker.load_model(
            tmp_path, path, hashlib.sha256(path.read_bytes()).hexdigest(), "gpgaitpp"
        )
    torch.save({"model": fixture.state_dict(), "optimizer": {}}, path)
    with pytest.raises(ValueError, match="model-only"):
        worker.load_model(
            tmp_path, path, hashlib.sha256(path.read_bytes()).hexdigest(), "gpgaitpp"
        )


def test_worker_timeout_cannot_return_embedding(encoder_setup, monkeypatch):
    encoder = gpgait.GPGaitPlusPlusEncoder(*encoder_setup)
    monkeypatch.setattr(
        gpgait, "prepare_official_grew", lambda *a: np.ones((1, 6, 40, 17, 1))
    )

    def timeout(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        encoder.encode(make_sequence())


PINNED_FASTPOSE = (
    Path(__file__).resolve().parents[2]
    / ".cache/gait_verification/sources/FastPoseGait"
)


@pytest.mark.skipif(
    not PINNED_FASTPOSE.is_dir(), reason="pinned FastPoseGait source not restored"
)
@pytest.mark.parametrize(
    "cls",
    [
        gpgait.GPGaitEncoder,
        gpgait.GPGaitPlusPlusEncoder,
        gpgait.GPGaitPlusPlusCASIAEncoder,
    ],
)
def test_model_id_fingerprint_matches_pinned_official_config(encoder_setup, cls):
    # The worker builds from the official YAML; the model ID must fingerprint
    # exactly that model_cfg, so the hand-held replica cannot drift silently.
    import json

    from gait_verification.backends.fastpose_contract import official_config

    official = official_config(PINNED_FASTPOSE, cls.variant)
    expected = {k: v for k, v in official.items() if k != "model"}
    expected["embedding_shape"] = [256, 19]
    digest = hashlib.sha256(json.dumps(expected, sort_keys=True).encode())
    encoder = cls(*encoder_setup)
    assert encoder.metadata.model_id.split(":")[-1] == digest.hexdigest()[:12]


def test_casia_and_grew_have_distinct_native_recipes(encoder_setup, monkeypatch):
    casia = gpgait.GPGaitPlusPlusCASIAEncoder(*encoder_setup)
    grew = gpgait.GPGaitPlusPlusEncoder(*encoder_setup)
    assert casia.metadata.training_dataset == "CASIA-B"
    assert grew.metadata.training_dataset == "GREW"
    assert casia.metadata.input_contract != grew.metadata.input_contract
    assert casia.metadata.model_id != grew.metadata.model_id
    seen = []

    def worker(cmd, **kwargs):
        output = cmd[cmd.index("--output") + 1]
        if "--dataset" in cmd:
            seen.append(cmd[cmd.index("--dataset") + 1])
            np.save(output, np.ones((6, 240, 17, 1)), allow_pickle=False)
        else:
            assert cmd[cmd.index("--variant") + 1] == "gpgaitpp_casia"
            np.save(output, np.ones(4864), allow_pickle=False)

    monkeypatch.setattr(subprocess, "run", worker)
    s = moving_pose()
    before = s.confidence.copy()
    embedding = casia.encode(s)
    assert seen == ["CASIA-B"] and np.isclose(np.linalg.norm(embedding.values), 1)
    np.testing.assert_array_equal(before, s.confidence)
