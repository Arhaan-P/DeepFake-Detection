"""Phase 2C engineering regressions; no CUDA, fitting or biometric evidence."""

import json

import numpy as np
import pytest
import test_phase1d as fixtures

from gait_verification import dataset_processing, e0
from gait_verification.common_eval import (
    ARCHITECTURE_POLICY,
    compare_common_runs,
    run_common_evaluation,
    quality_asymmetry,
)
from gait_verification.contracts import Status
from gait_verification.media import sha256_file
from gait_verification.offline_eval import compare_runs
from gait_verification.registry import load_json
from scripts.gait_verification.evaluate_claims import make_scorer

synthetic_eval = fixtures.synthetic_eval


def test_quality_gate_comparison_ignores_dictionary_key_order():
    gate = {"version": "synthetic-v1", "backend": "synthetic", "provisional": True}
    reordered = dict(reversed(list(gate.items())))
    assert (
        quality_asymmetry(
            [
                {"label": "A", "quality_gate": gate},
                {"label": "B", "quality_gate": reordered},
            ]
        )["unequal_gates"]
        is False
    )


def test_e0_cuda_score_respects_occupied_default_gpu_slot(tmp_path, monkeypatch):
    # The default slot is isolated from other audit processes. No CUDA API or
    # actual model worker runs: the dispatch boundary returns a synthetic score.
    monkeypatch.setattr(
        dataset_processing.tempfile, "gettempdir", lambda: str(tmp_path)
    )
    checkpoint = tmp_path / "stub-checkpoint.pt"
    checkpoint.write_bytes(b"synthetic worker boundary")
    backend = e0.FrozenE0(checkpoint, sha256_file(checkpoint), device="cuda")
    dispatched = []

    def stub_worker(*args):
        dispatched.append(args)
        return {"score": 0.0, "audit": {"synthetic": True}}

    monkeypatch.setattr(e0, "worker", stub_worker)
    with dataset_processing.gpu_slot(dataset_processing.default_gpu_lock_dir(), 1):
        with pytest.raises(dataset_processing.GpuBusy):
            backend.score(np.zeros((60, 78)), [np.zeros((60, 78))])
    assert dispatched == []


def _systems(registry, split, root):
    checksum = load_json("configs/gait_verification/e0.json")["model_only_sha256"]
    native, _ = make_scorer(
        registry,
        split,
        root,
        {
            "model": "e0",
            "feature_store": root / "features",
            "checkpoint": root / "unused-checkpoint.pt",
            "sha256": checksum,
        },
    )
    pretrained, _ = make_scorer(
        registry,
        split,
        root,
        {
            "model": "gaitpt",
            "enrollments": root / "unused-enrollments",
            "query_store": root / "unused-queries",
        },
    )
    return native, pretrained


@pytest.mark.parametrize("index,backend", [(0, "mediapipe"), (1, "rtmpose")])
def test_system_report_discloses_versioned_provisional_quality_gate(
    synthetic_eval, tmp_path, index, backend
):
    registry, split, *_ = synthetic_eval
    system = _systems(registry, split, tmp_path)[index]
    gate = system.get("quality_gate")
    assert isinstance(gate, dict), (
        "A recipe digest alone does not disclose the quality gate"
    )
    assert any("version" in key and value for key, value in gate.items())
    assert gate["provisional"] is True
    assert backend in json.dumps(gate).lower()


@pytest.mark.parametrize("compare", ["paired", "common"])
def test_comparison_explicitly_reports_e0_rtmpose_quality_asymmetry(
    synthetic_eval, tmp_path, compare
):
    registry, split, manifest, *_ = synthetic_eval
    systems = _systems(registry, split, tmp_path)

    def synthetic_score(request):
        return Status.POLICY_NOT_READY, "SYNTHETIC_SCORE", 0.0, None, True

    runs = run_common_evaluation(
        manifest,
        registry,
        split,
        [(system, synthetic_score) for system in systems],
        "SYNTHETIC_FIXTURE",
        target_fars=(0.05,),
        bootstrap_repetitions=2,
    )
    if compare == "paired":
        report = compare_runs(
            *runs, target_far=0.05, comparison_policy=ARCHITECTURE_POLICY, repetitions=2
        )
    else:
        report = compare_common_runs(runs, 0.05, ARCHITECTURE_POLICY, repetitions=2)
    asymmetry = report.get("quality_asymmetry")
    assert isinstance(asymmetry, dict), (
        "Training-population caveats do not disclose gate asymmetry"
    )
    assert asymmetry["unequal_gates"] is True
    gates = asymmetry["per_system"]
    assert set(gates) == {"e0", "gaitpt"}
    assert "mediapipe" in json.dumps(gates["e0"]).lower()
    assert "rtmpose" in json.dumps(gates["gaitpt"]).lower()
    assert "quality" in asymmetry["interpretation"].lower()
