"""Reproducible CLI engineering workflow, including missing-resource reports."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from gait_verification.e0_checkpoint import SHIPPED_SHA256
from gait_verification.e0_processing import E0FeatureStore, feature_recipe, process_e0
from gait_verification.offline_eval import read_run
import test_phase2b_integration as integration

world = integration.world

REPO = Path(__file__).resolve().parents[2]
CLI = REPO / "scripts/gait_verification/evaluate_claims.py"


def cli(*args):
    return subprocess.run(
        [sys.executable, str(CLI), *map(str, args)],
        cwd=REPO,
        capture_output=True,
        text=True,
    )


def test_cli_audit_validation_and_exclusive_writes(tmp_path):
    checkpoint, audit = tmp_path / "head.pt", tmp_path / "audit.json"
    out = cli("e0-audit", "--extract-to", checkpoint, "--out", audit)
    assert out.returncode == 0, out.stdout + out.stderr
    assert json.loads(audit.read_text())["original_sha256"] == SHIPPED_SHA256
    assert cli("e0-audit", "--extract-to", checkpoint, "--out", audit).returncode == 3
    native = tmp_path / "native.npz"
    np.savez(
        native,
        pose=np.random.default_rng(9).random((65, 33, 3)),
        observed=np.ones((65, 33), bool),
    )
    validation = tmp_path / "features.json"
    out = cli("e0-validate-features", "--native-pose", native, "--out", validation)
    assert out.returncode == 0, out.stdout + out.stderr
    assert json.loads(validation.read_text())["shape"] == [60, 78]


def test_cli_feature_resume_run_and_common_report(world, tmp_path):
    w = world
    registry, split, trials = (
        tmp_path / f"{name}.json" for name in ("registry", "split", "trials")
    )
    for path, doc in [
        (registry, w.registry.document),
        (split, w.split),
        (trials, w.doc),
    ]:
        path.write_text(json.dumps(doc))
    store = E0FeatureStore(tmp_path / "native", feature_recipe())
    process_e0(w.registry, w.split, w.root, store, w.assets, w.extractor)
    args = [
        "e0-features",
        "--registry",
        registry,
        "--split",
        split,
        "--data-root",
        w.root,
        "--e0-feature-store",
        store.root,
        "--pose-model",
        tmp_path / "absent-pose.task",
    ]
    for asset in w.assets:
        args += ["--asset", asset]
    out = cli(*args, "--out", tmp_path / "resume.json")
    assert out.returncode == 0, out.stdout + out.stderr  # cached run loads no detector
    checkpoint = tmp_path / "absent-e0.pt"
    checksum = "f65c44cde5af178b154efbcf47b9aa18dc67c37db19b63e25d27b9a781be65fd"
    out = cli(
        "run",
        "--model",
        "e0",
        "--registry",
        registry,
        "--split",
        split,
        "--trials",
        trials,
        "--data-root",
        w.root,
        "--e0-feature-store",
        store.root,
        "--e0-checkpoint",
        checkpoint,
        "--e0-sha256",
        checksum,
        "--evidence-class",
        "SYNTHETIC_FIXTURE",
        "--target-far",
        "0.05",
        "--bootstrap",
        "10",
        "--out",
        tmp_path / "e0-run",
    )
    assert out.returncode == 0, out.stdout + out.stderr
    report, rows = read_run(tmp_path / "e0-run")
    assert report["counts"]["submitted_trials"] == len(w.doc["trials"])
    assert all(r["reason"] == "E0_RESOURCE_MISSING" for r in rows)
    systems = tmp_path / "systems.json"
    systems.write_text(
        json.dumps(
            {
                "systems": [
                    {
                        "model": "e0",
                        "feature_store": str(store.root),
                        "checkpoint": str(checkpoint),
                        "sha256": checksum,
                    },
                    {
                        "model": "gaitpt",
                        "enrollments": str(tmp_path / "enrollments"),
                        "query_store": str(tmp_path / "queries"),
                    },
                    {
                        "model": "gpgaitpp_casia",
                        "enrollments": str(tmp_path / "enrollments"),
                        "query_store": str(tmp_path / "queries-casia"),
                    },
                ]
            }
        )
    )
    out = cli(
        "run-common",
        "--registry",
        registry,
        "--split",
        split,
        "--trials",
        trials,
        "--data-root",
        w.root,
        "--systems",
        systems,
        "--evidence-class",
        "SYNTHETIC_FIXTURE",
        "--target-far",
        "0.05",
        "--bootstrap",
        "10",
        "--comparison-policy",
        "historical-e0-versus-grew-architecture-v1",
        "--out",
        tmp_path / "common",
    )
    assert out.returncode == 0, out.stdout + out.stderr
    comparison = json.loads((tmp_path / "common/comparison.json").read_text())
    assert comparison["common_scoreable_trials"] == 0
    assert (
        comparison["separate_ablations"][0]["system"]["training_dataset"] == "CASIA-B"
    )
    for model in ("e0", "gaitpt", "gpgaitpp_casia"):
        _, outcomes = read_run(tmp_path / "common" / model)
        assert [r["trial_id"] for r in outcomes] == [
            r["trial_id"] for r in w.doc["trials"]
        ]
        assert all(r["verdict"] is None for r in outcomes)
