"""Complete 16-source fixture. RGB patterns are media tests; observations injected.

No gait is extracted successfully from a color pattern. Synthetic anatomically
ordered COCO observations and separately synthetic native E0 features are explicit.
"""

from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import stat
import sys

import numpy as np
import pytest

from gait_verification.dataset_processing import ModelSpec, ProcessingConfig
from gait_verification.pose_quality import WindowQualityPolicy, WindowedEncoder
from gait_verification.two_subject_pilot import PilotWorkflow
from gait_verification.pilot_intake import EVIDENCE
from test_phase1c import FakeDetector, StatEncoder
import test_phase2b_e0 as e0_fixtures
import test_pilot_intake as intake_fixtures

sheet = intake_fixtures.sheet
extracted = e0_fixtures.extracted


def configured(tmp_path, sheet):
    source, doc = sheet
    data, work = tmp_path / "data", tmp_path / "work"
    shutil.copytree(source, data)
    intake = data / "intake.json"
    intake.write_text(json.dumps(doc))
    return dict(
        schema_version="two-subject-pilot-workflow-v1",
        evidence_class=EVIDENCE,
        media_kind="SYNTHETIC_TEST_MEDIA",
        data_root=str(data),
        work_dir=str(work),
        intake=str(intake),
        registry=str(data / "registry.json"),
        subjects=doc["subjects"],
        asset_ids=[r["capture"]["asset_id"] for r in doc["rows"]],
        models=["e0", "gaitpt", "gpgait", "gpgaitpp", "gpgaitpp_casia"],
        variants=["vidstab-s10", "vidstab-s15"],
        augmentations=[{"kind": "brightness", "value": 0.04}],
        preview_assets=[doc["rows"][0]["capture"]["asset_id"]],
        preview_flagged_frames=1,
    )


def injected(cfg):
    policy = WindowQualityPolicy()
    models = []
    for name in cfg["models"]:
        if name == "e0":
            continue

        class SyntheticEncoder(StatEncoder):
            def __init__(self, name):
                self.metadata = replace(
                    StatEncoder.metadata, model_id=f"synthetic-{name}"
                )

            def encode(self, sequence):
                return replace(
                    super().encode(sequence), model_id=self.metadata.model_id
                )

        models.append(
            ModelSpec(
                name,
                f"synthetic-{name}",
                lambda n=name: WindowedEncoder(SyntheticEncoder(n), policy),
            )
        )
    processing = ProcessingConfig(
        FakeDetector.backend_id, policy, tuple(models), ("pose", "quality", "embed")
    )

    class SyntheticE0:
        model_id = "SYNTHETIC-E0-TEST-ONLY"

        def score(self, query, refs):
            assert query.shape == (60, 78) and all(r.shape == (60, 78) for r in refs)
            return float(query.mean() - np.mean(refs)), {}

    return PilotWorkflow(
        cfg,
        processing=processing,
        detector_factory=FakeDetector,
        cuda_probe=lambda: True,
        e0_extractor=lambda p: np.ones((60, 78), np.float32),
        e0_backend_factory=SyntheticE0,
        gpu_lock_dir=Path(cfg["work_dir"]).parent / "gpu-locks",  # never a user's slot
    )


def keys(obj):
    """Every dictionary key anywhere in a report (metric names, not FFmpeg text)."""
    if isinstance(obj, dict):
        return {k.lower() for k in obj} | set().union(*map(keys, obj.values()))
    if isinstance(obj, list):
        return set().union(*map(keys, obj))
    return set()


def test_complete_16_original_workflow_and_resume(tmp_path, sheet):
    cfg = configured(tmp_path, sheet)
    workflow = injected(cfg)
    originals = {p: p.read_bytes() for p in Path(cfg["data_root"]).rglob("*.mp4")}
    plan = workflow.plan()
    assert not Path(cfg["work_dir"]).exists()
    assert plan["independent_originals"] == 16
    assert plan["plan_id"] == workflow.plan()["plan_id"]
    result = workflow.all()
    assert result["status"] == "COMPLETE"
    report = result["report"]
    assert report["independent_originals"] == 16 and report["derivatives"] == 48
    assert len(report["quality"]["recordings"]) == 64
    assert all(len(s["outcomes"]) == 16 for s in report["verification"].values())
    assert all(
        s["counts"]["score_available"] == 16 for s in report["verification"].values()
    )
    assert all(s["counts"]["genuine"] == 8 for s in report["verification"].values())
    assert all(
        o["status"] == "POLICY_NOT_READY" and o["verdict"] is None
        for s in report["verification"].values()
        for o in s["outcomes"]
    )
    assert all(r["independent_reference_count"] == 4 for r in report["enrollment"])
    assert report["matched_variant_scoring"]["reason"] == "RECIPE_MISMATCH"
    # vid.stab's own `accuracy` recipe field lives in augmentation provenance only.
    reported = {k: v for k, v in report.items() if k != "augmentation"}
    assert not keys(reported) & {
        "accuracy",
        "roc_auc",
        "eer",
        "far",
        "frr",
        "tar",
        "metrics",
        "threshold",
        "bootstrap",
    }
    assert "TWO-PERSON EXPLORATORY PILOT" in report["notice"]
    snapshots = list(Path(cfg["work_dir"]).rglob("overlay-*.jpg"))
    assert len(snapshots) == 3
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in snapshots)
    assert all(p.read_bytes() == b for p, b in originals.items())
    before = {
        p: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in Path(cfg["work_dir"]).rglob("*.npz")
    }
    resumed = workflow.all()
    assert resumed["status"] == "COMPLETE"
    assert all(
        (p.read_bytes(), p.stat().st_mtime_ns) == value for p, value in before.items()
    )
    assert all(
        s["cache_reused"] == 16 for s in resumed["report"]["verification"].values()
    )
    swap = workflow.score("set2-to-set1")
    assert all(s["counts"]["submitted_trials"] == 16 for s in swap.values())


def test_missing_inputs_all_does_not_register_or_load_models(tmp_path, sheet):
    cfg = configured(tmp_path, sheet)
    first = Path(cfg["data_root"]) / "originals/SYN001/left45/set1.mp4"
    first.unlink()
    result = injected(cfg).all()
    assert result["status"] == "WAITING_FOR_RECORDINGS_OR_METADATA"
    assert not Path(cfg["registry"]).exists()


def test_source_alias_output_and_changed_original_fail_closed(tmp_path, sheet):
    cfg = configured(tmp_path, sheet)
    workflow = injected(cfg)
    workflow.register()
    target = Path(cfg["data_root"]) / "originals/SYN001/left45/set1.mp4"
    cfg["work_dir"] = str(target.parent)
    with pytest.raises(ValueError):
        injected(cfg).plan()
    cfg["work_dir"] = str(tmp_path / "work2")
    target.write_bytes(b"corrupt synthetic input")
    with pytest.raises(ValueError):
        injected(cfg).enroll()


def test_failed_recording_is_isolated_and_counts_reconcile(tmp_path, sheet):
    """A too-short take fails the provisional gate; nothing else is hidden or dropped."""
    from test_phase1c import make_video

    cfg = configured(tmp_path, sheet) | dict(
        variants=[], augmentations=[], models=["e0", "gaitpt"], preview_assets=[]
    )
    bad = Path(cfg["data_root"]) / "originals/SYN001/left45/set1.mp4"
    bad.unlink()
    make_video(bad, 1.0)  # shorter than one 2-second quality window
    result = injected(cfg).all()
    assert result["status"] == "COMPLETE_WITH_FAILURES"
    report = result["report"]
    row = next(
        r
        for r in report["quality"]["recordings"]
        if r["asset_id"] == "SYN001_left45_set1"
    )
    assert row["pose_engineering_gate"]["status"] == "NOT_USABLE_BY_PROVISIONAL_GATE"
    assert row["recommend_manual_review"]
    refs = {(r["claimed_subject_id"], r["model"]): r for r in report["enrollment"]}
    assert refs[("SYN001", "gaitpt")]["independent_reference_count"] == 3
    assert refs[("SYN001", "gaitpt")]["failures"][0]["reason"].startswith("QUALITY:")
    gaitpt = report["verification"]["gaitpt"]
    assert gaitpt["counts"]["score_available"] == 10
    assert gaitpt["status_reason_counts"][
        "CLAIM_NOT_ENROLLED/ENROLLMENT_INCOMPLETE"
    ] == {"genuine": 3, "wrong_claim": 3}
    for s in report["verification"].values():  # every trial accounted for exactly once
        c = s["counts"]
        assert (
            c["score_available"] + c["unscored_or_abstained"]
            == c["submitted_trials"]
            == 16
        )
        assert sum(sum(v.values()) for v in s["status_reason_counts"].values()) == 16
        assert sum(sum(v.values()) for v in s["by_probe_view"].values()) == 16
    summary = next(Path(cfg["work_dir"]).rglob("summary.md")).read_text()
    assert "SYN001_left45_set1" in summary and "NOT A VALIDATED" in summary


def test_cli_init_validate_and_read_only_plan(tmp_path, sheet, monkeypatch, capsys):
    from scripts.gait_verification import two_subject_pilot as cli

    def invoke(*args):
        monkeypatch.setattr(sys, "argv", ["two_subject_pilot.py", *map(str, args)])
        code = cli.main()
        return code, json.loads(capsys.readouterr().out)

    data = tmp_path / "data"
    code, out = invoke(
        "init",
        "--subjects",
        "SYN001",
        "SYN002",
        "--media-kind",
        "SYNTHETIC_TEST_MEDIA",
        "--data-root",
        data,
        "--work-dir",
        tmp_path / "work",
        "--registry",
        data / "registry.json",
        "--intake-out",
        tmp_path / "intake.json",
        "--config-out",
        tmp_path / "pilot.json",
    )
    assert code == 0 and out["rows"] == 16
    assert (
        invoke(
            "init",
            "--subjects",
            "SYN001",
            "SYN002",
            "--media-kind",
            "SYNTHETIC_TEST_MEDIA",
            "--data-root",
            data,
            "--work-dir",
            tmp_path / "work",
            "--registry",
            data / "registry.json",
            "--intake-out",
            tmp_path / "intake.json",
            "--config-out",
            tmp_path / "pilot.json",
        )[0]
        == 3
    )  # never overwrites
    data.mkdir()
    code, out = invoke("validate", "--config", tmp_path / "pilot.json")
    assert code == 2 and len(out["missing_originals"]) == 16
    assert all(
        any("MISSING_METADATA:date" in i for i in v) for v in out["row_issues"].values()
    )
    source, doc = sheet
    shutil.rmtree(data)
    shutil.copytree(source, data)
    (tmp_path / "intake2.json").write_text(json.dumps(doc))
    cfg = json.loads((tmp_path / "pilot.json").read_text()) | dict(
        intake=str(tmp_path / "intake2.json")
    )
    (tmp_path / "pilot2.json").write_text(json.dumps(cfg))
    code, out = invoke("plan", "--config", tmp_path / "pilot2.json")
    assert (
        code == 0
        and out["independent_originals"] == 16
        and out["planned_derivatives"] == 32
    )
    assert not (tmp_path / "work").exists() and not (data / "registry.json").exists()
    cfg["media_kind"] = (
        "REAL_PARTICIPANT_RECORDINGS"  # synthetic metadata cannot pass as real
    )
    (tmp_path / "pilot3.json").write_text(json.dumps(cfg))
    assert invoke("validate", "--config", tmp_path / "pilot3.json")[0] == 3


@pytest.mark.skipif(
    os.environ.get("GAIT_V2_CUDA_TESTS") != "1",
    reason="opt-in real frozen CUDA pilot workflow",
)
def test_cli_all_with_real_frozen_cuda_models(
    tmp_path, sheet, monkeypatch, capsys, extracted
):
    """Real GaitPT/GPGait/GPGait++ x2 and E0 head on CUDA. Poses and E0 features are
    named synthetic injections: generated patterns contain no human gait."""
    from gait_verification import e0_processing, registry as loaders
    from scripts.gait_verification import evaluate_claims, two_subject_pilot as cli

    checkpoint, _ = extracted
    cfg = configured(tmp_path, sheet) | dict(
        variants=[],
        augmentations=[],
        preview_assets=[],
        runtime=cli.RUNTIME | dict(e0_checkpoint=str(checkpoint)),
    )
    path = tmp_path / "pilot.json"
    path.write_text(json.dumps(cfg))
    monkeypatch.setattr(loaders, "load_rtmpose", lambda *a: FakeDetector())
    original = evaluate_claims.processing_config
    monkeypatch.setattr(
        evaluate_claims,
        "processing_config",
        lambda a: replace(original(a), pose_backend_id=FakeDetector.backend_id),
    )

    class SyntheticE0Features:
        def __init__(self, *args):
            pass

        def __call__(self, video):
            return np.linspace(-1, 1, 60 * 78, dtype=np.float32).reshape(60, 78)

    monkeypatch.setattr(e0_processing, "MediaPipeE0Extractor", SyntheticE0Features)
    monkeypatch.setattr(
        sys, "argv", ["two_subject_pilot.py", "all", "--config", str(path)]
    )
    assert cli.main() == 0, capsys.readouterr().out
    out = json.loads(capsys.readouterr().out)
    assert out["scored"] == {m: 16 for m in cfg["models"]}
    report = json.loads(next(Path(cfg["work_dir"]).rglob("report.json")).read_text())
    for name, s in report["verification"].items():
        assert {o["status"] for o in s["outcomes"]} == {"POLICY_NOT_READY"}
        if name != "e0":
            assert s["recipe_model_ids"][0].startswith(
                loaders.model_entry(name)["model_id"]
            )
    monkeypatch.setattr(
        sys, "argv", ["two_subject_pilot.py", "process", "--config", str(path)]
    )
    assert cli.main() == 0
    rerun = json.loads(capsys.readouterr().out)
    assert rerun["cache_reused"] == {m: 16 for m in cfg["models"]}  # nothing recomputed


def test_cuda_absence_never_falls_back_to_cpu_or_loads_models(tmp_path, sheet):
    cfg = configured(tmp_path, sheet) | dict(
        variants=[], augmentations=[], models=["gaitpt"], preview_assets=[]
    )
    workflow = injected(cfg)

    def forbidden():
        raise AssertionError("model loaded without CUDA")

    workflow.processing = replace(
        workflow.processing,
        models=(ModelSpec("gaitpt", "synthetic-gaitpt", forbidden),),
    )
    workflow.detector_factory = forbidden
    workflow.cuda_probe = lambda: False
    result = workflow.all()
    assert result["status"] == "COMPLETE_WITH_FAILURES"
    report = result["report"]
    # Pose fails retryably; quality is then blocked rather than computed on nothing.
    assert {f["final_reason"] for f in report["processing"]["failures"]} == {
        "CUDA_UNAVAILABLE_NO_CPU_FALLBACK",
        "POSE_ARTIFACT_NOT_READY",
    }
    counts = report["verification"]["gaitpt"]["counts"]
    assert counts["score_available"] == 0 and counts["unscored_or_abstained"] == 16
    assert all(
        o["verdict"] is None for o in report["verification"]["gaitpt"]["outcomes"]
    )
