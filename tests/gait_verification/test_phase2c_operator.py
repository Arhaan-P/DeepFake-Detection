"""Full CLI acceptance with generated media and explicitly injected pose/features.

The optional CUDA case uses real GaitPT and E0 weights. The patterns in generated
RGB files are never claimed to contain a detected walker. Default encoder and all
pose/native-feature observations are deliberate synthetic fixtures.
"""

from dataclasses import replace
import json
import os
import sys

import numpy as np
import pytest

from gait_verification import e0, registry as loaders
from gait_verification.dataset_processing import ModelSpec, ProcessingConfig
from gait_verification.e0_processing import E0FeatureStore, feature_recipe, process_e0
from gait_verification.offline_eval import read_run
from gait_verification.pose_quality import WindowedEncoder
from scripts.gait_verification import enroll_subject, evaluate_claims
from test_phase1c import FakeDetector, StatEncoder
from test_phase2 import build_world
import test_phase2b_e0 as e0_fixtures

extracted = e0_fixtures.extracted


@pytest.mark.parametrize("real_cuda", [False, True])
def test_complete_operator_workflow(tmp_path, monkeypatch, extracted, real_cuda):
    if real_cuda and os.environ.get("GAIT_V2_CUDA_TESTS") != "1":
        pytest.skip("opt-in real frozen CUDA workflow")
    w = build_world(tmp_path)
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(w.registry.document))
    split_draft, split_path, trials_path = [
        tmp_path / f"{name}.json" for name in ("draft", "split", "trials")
    ]

    def invoke(module, *args, code=0):
        monkeypatch.setattr(sys, "argv", [module.__name__, *map(str, args)])
        assert module.main() == code

    invoke(
        evaluate_claims,
        "validate",
        "--registry",
        registry_path,
        "--data-root",
        w.data,
        "--measure",
        "--out",
        tmp_path / "validate.json",
    )
    invoke(
        evaluate_claims,
        "split",
        "--registry",
        registry_path,
        "--policy-version",
        "SYNTHETIC-AUDIT-ONLY",
        "--seed",
        109,
        "--development",
        2,
        "--calibration",
        0,
        "--locked-final",
        0,
        "--out",
        split_draft,
    )
    invoke(
        evaluate_claims,
        "lock",
        "--split",
        split_draft,
        "--locked-by",
        "SYNTHETIC-AUDIT-ONLY",
        "--out",
        split_path,
    )
    invoke(
        evaluate_claims,
        "trials",
        "--registry",
        registry_path,
        "--split",
        split_path,
        "--role",
        "development",
        "--policy-version",
        "SYNTHETIC-AUDIT-ONLY",
        "--seed",
        211,
        "--impostors-per-query",
        1,
        "--out",
        trials_path,
    )
    split = json.loads(split_path.read_text())
    manifest = json.loads(trials_path.read_text())
    assert manifest["counts"]["trials"] == 8
    original_hashes = {p: p.read_bytes() for p in w.paths.values()}

    # Injection occurs at the observation boundary; backend ID openly names the fake.
    monkeypatch.setattr(loaders, "load_rtmpose", lambda *args: FakeDetector())
    monkeypatch.setattr(enroll_subject, "load_rtmpose", lambda *args: FakeDetector())
    key = loaders.model_entry("gaitpt")["model_id"]

    class SyntheticEncoder(StatEncoder):
        metadata = replace(StatEncoder.metadata, model_id=key)

        def encode(self, sequence):
            return replace(super().encode(sequence), model_id=key)

    if not real_cuda:
        monkeypatch.setattr(
            loaders,
            "load_encoder",
            lambda *args: WindowedEncoder(SyntheticEncoder(), w.policy),
        )
        monkeypatch.setattr(
            enroll_subject,
            "load_encoder",
            lambda *args: WindowedEncoder(SyntheticEncoder(), w.policy),
        )
        original_frozen = e0.FrozenE0
        monkeypatch.setattr(
            e0,
            "FrozenE0",
            lambda path, checksum: original_frozen(path, checksum, device="cpu"),
        )
        # A synthetic observation stage does not require a GPU. Real CUDA case
        # leaves this preflight and both real frozen scorers unchanged.
        monkeypatch.setattr(
            evaluate_claims,
            "processing_config",
            lambda args: ProcessingConfig(
                FakeDetector.backend_id,
                w.policy,
                (
                    ModelSpec(
                        "gaitpt",
                        key,
                        lambda: WindowedEncoder(SyntheticEncoder(), w.policy),
                    ),
                ),
                ("pose", "quality", "embed"),
            ),
        )
    else:
        original_config = evaluate_claims.processing_config
        monkeypatch.setattr(
            evaluate_claims,
            "processing_config",
            lambda args: replace(
                original_config(args), pose_backend_id=FakeDetector.backend_id
            ),
        )

    from gait_verification import dataset_processing

    # run_processing's default is bound at definition; CLI passes its own actual
    # probe below. No actual CUDA claim is made for the synthetic default case.
    original_runner = dataset_processing.run_processing
    if not real_cuda:
        monkeypatch.setattr(
            dataset_processing,
            "run_processing",
            lambda *args, **kwargs: original_runner(
                *args, **kwargs, cuda_probe=lambda: True
            ),
        )
    selection = [
        "--registry",
        registry_path,
        "--split",
        split_path,
        "--data-root",
        w.data,
        "--work-dir",
        w.work,
        "--model",
        "gaitpt",
    ]
    for asset_id in w.ids:
        selection += ["--asset", asset_id]
    invoke(
        evaluate_claims,
        "plan",
        *selection,
        "--measure",
        "--out",
        tmp_path / "plan.json",
    )
    assert not w.work.exists()
    plan = json.loads((tmp_path / "plan.json").read_text())
    invoke(evaluate_claims, "process", *selection, "--expect-plan-id", plan["plan_id"])
    before = {
        p: p.read_bytes()
        for p in w.work.rglob("*")
        if p.is_file() and p.suffix != ".jsonl" and "runs" not in p.parts
    }
    invoke(evaluate_claims, "process", *selection, "--expect-plan-id", plan["plan_id"])
    assert all(p.read_bytes() == b for p, b in before.items())
    invoke(evaluate_claims, "report", *selection, "--out", tmp_path / "coverage.json")

    enrollments = tmp_path / "enrollments"
    invoke(
        enroll_subject,
        "--enrollments",
        enrollments,
        "batch",
        "--capture-registry",
        registry_path,
        "--split",
        split_path,
        "--data-root",
        w.data,
        "--assets",
        *w.ids,
        "--models",
        "gaitpt",
        "--pose-cache-dir",
        w.work / "pose",
        "--report-out",
        tmp_path / "enroll.json",
    )
    query_dir = w.work / plan["embedding_stores"]["gaitpt"]
    native = E0FeatureStore(tmp_path / "e0-features", feature_recipe())
    native_inputs = {
        p: np.random.default_rng(i).normal(size=(60, 78)).astype(np.float32)
        for i, p in enumerate(w.paths.values())
    }
    prepared = process_e0(
        w.registry, split, w.data, native, w.ids, lambda path: native_inputs[path]
    )
    assert prepared["status"] == "COMPLETE"
    invoke(
        evaluate_claims,
        "e0-features",
        "--registry",
        registry_path,
        "--split",
        split_path,
        "--data-root",
        w.data,
        "--e0-feature-store",
        native.root,
        *sum((["--asset", a] for a in w.ids), []),
        "--out",
        tmp_path / "e0-resume.json",
    )
    checkpoint, extraction = extracted
    checksum = extraction["model_only_sha256"]
    systems = tmp_path / "systems.json"
    systems.write_text(
        json.dumps(
            {
                "systems": [
                    {
                        "model": "e0",
                        "feature_store": str(native.root),
                        "checkpoint": str(checkpoint),
                        "sha256": checksum,
                    },
                    {
                        "model": "gaitpt",
                        "enrollments": str(enrollments),
                        "query_store": str(query_dir),
                    },
                ]
            }
        )
    )
    invoke(
        evaluate_claims,
        "run-common",
        "--registry",
        registry_path,
        "--split",
        split_path,
        "--trials",
        trials_path,
        "--data-root",
        w.data,
        "--systems",
        systems,
        "--evidence-class",
        "SYNTHETIC_FIXTURE",
        "--target-far",
        0.05,
        "--bootstrap",
        10,
        "--comparison-policy",
        "historical-e0-versus-grew-architecture-v1",
        "--out",
        tmp_path / "common",
    )
    comparison = json.loads((tmp_path / "common/comparison.json").read_text())
    assert comparison["common_scoreable_trials"] == 8
    assert comparison["quality_asymmetry"]["unequal_gates"]
    for model in ("e0", "gaitpt"):
        report, outcomes = read_run(tmp_path / "common" / model)
        assert report["coverage"] == {"operational": 0.0, "quality": 1.0, "score": 1.0}
        assert [r["trial_id"] for r in outcomes] == [
            r["trial_id"] for r in manifest["trials"]
        ]
        assert all(
            r["status"] == "POLICY_NOT_READY" and r["verdict"] is None for r in outcomes
        )
        assert (
            tmp_path / "common" / model / "outcomes.jsonl"
        ).stat().st_mode & 0o777 == 0o600
    assert all(p.read_bytes() == b for p, b in original_hashes.items())
