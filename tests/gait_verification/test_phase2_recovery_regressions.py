"""Regression evidence for defects found during Phase 2 continuation."""

import json
import subprocess
import pytest
from test_phase2 import (
    CLI,
    Detectors,
    build_world,
    config,
    final,
    locked_split,
    needs_ffmpeg,
    run,
    spec,
)
from gait_verification import dataset_processing as dp


@needs_ffmpeg
@pytest.mark.parametrize("stage", ["pose", "embed"])
def test_model_load_failure_is_persisted_and_retryable(tmp_path, stage):
    w = build_world(tmp_path, subjects=("SYN001",), views=("left45",))

    def broken():
        raise RuntimeError("CUDA out of memory during model loading")

    cfg = config(w, spec(w.policy, factory=broken)) if stage == "embed" else config(w)
    report = run(w, cfg, detector=broken if stage == "pose" else Detectors())
    assert report["status"] == "COMPLETE_WITH_FAILURES"
    assert final(report, stage) == {"FAILED_RETRYABLE"}
    assert report["failures"][0]["final_reason"] == "CUDA_OR_HOST_OUT_OF_MEMORY"
    assert list((w.work / "runs").glob("*.json"))
    assert run(w, config(w, spec(w.policy)))["status"] == "COMPLETE"


@needs_ffmpeg
def test_blocked_selection_cannot_report_complete(tmp_path):
    w = build_world(tmp_path, subjects=("SYN001",), views=("left45",))
    report = run(w, config(w), ["ghost"])
    assert report["status"] == "COMPLETE_WITH_FAILURES"
    assert report["blocked_inputs"] == [
        {"asset_id": "ghost", "reason": "UNKNOWN_ASSET"}
    ]


@needs_ffmpeg
@pytest.mark.parametrize("change", ["edit", "delete"])
def test_postrun_source_integrity_failure_cannot_report_complete(tmp_path, change):
    w = build_world(tmp_path, subjects=("SYN001",), views=("left45",))

    class ChangedSource(Detectors):
        def __call__(self):
            detector = super().__call__()
            original = detector.__class__.__call__

            class Mutating(detector.__class__):
                def __call__(self, frame):
                    # Modify after decoding has started. A source is never repaired.
                    result = original(self, frame)
                    path = w.paths[w.ids[0]]
                    if path.exists():
                        path.unlink() if change == "delete" else path.write_bytes(
                            b"changed"
                        )
                    return result

            return Mutating()

    report = run(w, config(w), detector=ChangedSource())
    assert report["status"] == "COMPLETE_WITH_FAILURES"
    assert report["originals"]["changed_during_processing"] == w.ids


@needs_ffmpeg
def test_cli_plan_enforces_supplied_split_and_changes_plan_identity(tmp_path):
    w = build_world(tmp_path, subjects=("SYN001", "SYN002"), views=("left45",))
    split = locked_split(
        w.registry, {"development": 1, "calibration": 0, "locked_final": 1}
    )
    regfile, splitfile = tmp_path / "registry.json", tmp_path / "split.json"
    regfile.write_text(json.dumps(w.registry.document))
    splitfile.write_text(json.dumps(split))
    out = tmp_path / "plan.json"
    (tmp_path / "assets.txt").write_text("\n".join(w.ids))
    result = subprocess.run(
        CLI
        + [
            "plan",
            "--registry",
            str(regfile),
            "--data-root",
            str(w.data),
            "--work-dir",
            str(w.work),
            "--assets-file",
            str(tmp_path / "assets.txt"),
            "--split",
            str(splitfile),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    plan = json.loads(out.read_text())
    assert plan["blocked"][0]["reason"] == "LOCKED_FINAL_ACCESS_DENIED"
    unrestricted = dp.build_plan(w.registry, w.data, w.work, w.ids, config(w))
    assert plan["plan_id"] != unrestricted["plan_id"]
    assert not w.work.exists()


def test_ledger_append_recovers_after_torn_line(tmp_path):
    ledger = dp.Ledger(tmp_path / "ledger.jsonl")
    ledger.append(task_id="t", state="DONE")
    with ledger.path.open("a") as f:
        f.write('{"task_id":"t","state":"FAIL')
    ledger.append(task_id="t", state="FAILED_PERMANENT", reason="INVALID_MEDIA")
    assert ledger.latest()["t"]["state"] == "FAILED_PERMANENT"


@needs_ffmpeg
def test_derive_never_processes_locked_final_original(tmp_path):
    w = build_world(tmp_path, subjects=("SYN001",), views=("left45",))
    from test_phase2 import registry_of

    registry = registry_of([dict(w.rows[0], split_role="locked_final")])
    calls = []
    results, rows = dp.derive_variants(
        registry, w.data, w.ids, ["vidstab-s10"], stabilizer=lambda *a: calls.append(a)
    )
    assert results[0]["status"] == "FAILED" and "LOCKED_FINAL" in results[0]["reason"]
    assert not calls and not rows and not (w.data / "derived").exists()


@needs_ffmpeg
@pytest.mark.parametrize("damage", ["parent", "manifest"])
def test_derivative_processing_revalidates_ancestry_and_transform(tmp_path, damage):
    from test_phase2 import derived_row, make_video, registry_of, sha256_file

    w = build_world(tmp_path, subjects=("SYN001",), views=("left45",))
    child_path = make_video(w.data / "child.mp4", 3.5)
    manifest_path = w.data / "transform.json"
    manifest_path.write_text(
        json.dumps(
            {
                "status": "OK",
                "input": {"sha256": w.rows[0]["sha256"]},
                "output": {"sha256": sha256_file(child_path)},
            }
        )
    )
    child = derived_row(
        w.rows[0],
        "vidstab-s10",
        sha256=sha256_file(child_path),
        source_file_uri="child.mp4",
        transformation_manifest_ref="transform.json",
        transformation_manifest_sha256=sha256_file(manifest_path),
    )
    registry = registry_of(w.rows + [child])
    if damage == "parent":
        w.paths[w.ids[0]].write_bytes(b"changed parent")
    else:
        manifest_path.write_text("{}")
    plan = dp.build_plan(registry, w.data, w.work, [child["asset_id"]], config(w))
    assert plan["blocked"] and all(t["state"] == "BLOCKED" for t in plan["tasks"])


@needs_ffmpeg
@pytest.mark.parametrize("kind", ["quality", "embed"])
def test_resealed_incomplete_cache_is_invalid_and_coverage_remains_readable(
    tmp_path, kind
):
    from gait_verification.offline_eval import seal

    w = build_world(tmp_path, subjects=("SYN001",), views=("left45",))
    cfg = config(w, spec(w.policy))
    run(w, cfg)
    path = (
        next((w.work / "quality").glob("*.json"))
        if kind == "quality"
        else next((w.work / "embeddings").rglob("*.json"))
    )
    body = json.loads(path.read_text())
    del body["summary" if kind == "quality" else "status"]
    path.write_text(json.dumps(seal(body)))
    plan = dp.build_plan(w.registry, w.data, w.work, w.ids, cfg)
    assert next(t for t in plan["tasks"] if t["kind"] == kind)["state"] == "INVALID"
    assert len(dp.coverage_report(w.registry, w.work, w.ids, cfg)["recordings"]) == 1


@needs_ffmpeg
def test_work_directory_cannot_contain_registered_originals(tmp_path):
    w = build_world(tmp_path, subjects=("SYN001",), views=("left45",))
    from test_phase2 import snapshot

    before = snapshot(w.data)
    with pytest.raises(ValueError, match="contains registered source"):
        dp.run_processing(
            w.registry,
            w.data,
            w.data,
            w.ids,
            config(w),
            detector_factory=Detectors(),
            cuda_probe=lambda: True,
            gpu_lock_dir=w.gpu,
        )
    assert snapshot(w.data) == before


@needs_ffmpeg
def test_missing_checkpoint_during_model_load_is_retryable(tmp_path):
    w = build_world(tmp_path, subjects=("SYN001",), views=("left45",))

    def missing():
        raise FileNotFoundError("checkpoint absent")

    report = run(w, config(w), detector=missing)
    assert final(report, "pose") == {"FAILED_RETRYABLE"}
    assert run(w, config(w))["status"] == "COMPLETE"
