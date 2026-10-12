"""Phase 2: dataset ingestion, dry-run planning and resumable, integrity-checked processing.

Evidence types are kept apart (see PHASE_2_DATASET_INTEGRATION_REPORT.md):
  * unit tests            literal values, no media
  * synthetic dataset     generated `testsrc2` clips with SYNTHETIC pseudonymous subjects, a fake
                          pose detector and stand-in encoders. These test processing and integrity
                          behaviour ONLY; they are not gait data and say nothing about recognition.
  * real FFmpeg / media   real ffprobe, decoding and vid.stab on generated clips
  * real CUDA smoke       opt-in (GAIT_V2_CUDA_TESTS=1); real RTMPose / frozen encoders
  * real scientific study NOT PERFORMED (no cohort exists; see DEFERRED_EVIDENCE_AND_DECISIONS.md)
"""

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from gait_verification import dataset_processing as dp
from gait_verification.capture_registry import (
    SCHEMA,
    CaptureRegistry,
    ProvenanceError,
    load_capture_schema,
)
from gait_verification.contracts import ModelMetadata
from gait_verification.dataset_ingest import diagnose_rows, ingest_report
from gait_verification.dataset_processing import (
    GpuBusy,
    ModelSpec,
    ProcessingConfig,
    build_plan,
    coverage_report,
    derive_variants,
    gpu_slot,
    run_processing,
)
from gait_verification.decisions import OwnerDecisionRequired, load_decisions
from gait_verification.enrollments import DirectoryEnrollmentStore
from gait_verification.media import sha256_file
from gait_verification.pipeline import (
    CacheIntegrityError,
    CachedPoseExtractor,
    enroll_registered,
)
from gait_verification.pose_quality import WindowQualityPolicy, WindowedEncoder
from test_phase1c import POINTS, FakeDetector, StatEncoder, make_video
from test_phase1d import NOW, derived_row, locked_split, raw_row, sha

REPO = Path(__file__).resolve().parents[2]
CLI = [sys.executable, str(REPO / "scripts/gait_verification/evaluate_claims.py")]
DOC = REPO / "docs/gait_verification/DEFERRED_EVIDENCE_AND_DECISIONS.md"
CAPTURE = load_capture_schema()
needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg unavailable",
)
CUDA = os.environ.get("GAIT_V2_CUDA_TESTS") == "1"


# ----------------------------------------------------------------------- fixtures


class Stat2(StatEncoder):
    metadata = ModelMetadata(
        "stat2:recipe", "none", "none", "n/a", 0, "synthetic", "stat", "none"
    )

    def encode(self, s):
        return replace(super().encode(s), model_id="stat2:recipe")


class StatCkpt2(StatEncoder):  # a different pinned checkpoint of the same architecture
    metadata = ModelMetadata(
        "stat:recipe-ckpt2", "none", "none", "n/a", 0, "synthetic", "stat", "none"
    )

    def encode(self, s):
        return replace(super().encode(s), model_id="stat:recipe-ckpt2")


class Factory:
    """Counts how often an expensive model would have been loaded."""

    def __init__(self, make):
        self.make, self.loads = make, 0

    def __call__(self):
        self.loads += 1
        return self.make()


def spec(policy, name="stat", cls=StatEncoder, factory=None):
    key = cls.metadata.model_id
    return ModelSpec(
        name, key, factory or Factory(lambda: WindowedEncoder(cls(), policy))
    )


def build_world(
    tmp_path, subjects=("SYN001", "SYN002"), views=("left45", "right45"), extra=()
):
    """Synthetic pseudonymous subjects x generated clips, registered with ffprobe facts."""
    from gait_verification.media import probe_video

    data = tmp_path / "data"
    (data / "raw").mkdir(parents=True)
    rows, paths = [], {}
    for i, (subject, view) in enumerate((s, v) for s in subjects for v in views):
        path = make_video(
            data / "raw" / f"clip{i}.mp4", 3.0 + 0.1 * i
        )  # distinct bytes
        meta = probe_video(path)
        rows.append(
            raw_row(
                subject,
                view,
                sha256=meta.sha256,
                source_file_uri=str(path.relative_to(data)),
                duration_sec=meta.duration_sec,
            )
        )
        paths[rows[-1]["asset_id"]] = path
    rows += list(extra)
    return SimpleNamespace(
        data=data, work=tmp_path / "work", gpu=tmp_path / "gpu", rows=rows, paths=paths,
        registry=registry_of(rows), policy=WindowQualityPolicy(), ids=sorted(paths), tmp=tmp_path,
    )  # fmt: skip


def registry_of(rows):
    return CaptureRegistry({"schema_version": SCHEMA, "assets": rows}, CAPTURE, NOW)


def config(w, *models, stages=None, backend="fake-rtmpose:v1", policy=None):
    models = tuple(models)
    stages = stages or (("pose", "quality", "embed") if models else ("pose", "quality"))
    return ProcessingConfig(backend, policy or w.policy, models, stages)


class Detectors:
    def __init__(self, **kw):
        self.kw, self.loads = kw, 0

    def __call__(self):
        self.loads += 1
        return FakeDetector(**self.kw)


def run(w, cfg, ids=None, detector=None, **kw):
    kw.setdefault("cuda_probe", lambda: True)
    return run_processing(
        w.registry, w.data, w.work, ids or w.ids, cfg,
        detector_factory=detector if detector is not None else Detectors(),
        gpu_lock_dir=w.gpu, **kw,
    )  # fmt: skip


def states(report, kind=None):
    return sorted(
        (r["asset_id"], r["kind"], r["final_state"])
        for r in report["tasks"]
        if kind in (None, r["kind"])
    )


def final(report, kind=None):
    return {r["final_state"] for r in report["tasks"] if kind in (None, r["kind"])}


def snapshot(*roots):
    out = {}
    for root in roots:
        root = Path(root)
        if root.exists():
            for p in sorted(root.rglob("*")):
                st = p.stat()
                out[str(p)] = (
                    p.is_dir(),
                    st.st_size,
                    st.st_mtime_ns,
                    None if p.is_dir() else sha256_file(p),
                )
    return out


# ================================================================ unit: register


def test_deferred_register_is_complete_and_no_deferred_item_is_marked_done():
    text = DOC.read_text()
    ids = {
        "EXT": range(1, 9),
        "OWN": range(1, 13),
        "SCI": range(1, 11),
        "SEC": range(1, 6),
    }
    for prefix, numbers in ids.items():
        for n in numbers:
            row = [
                line
                for line in text.splitlines()
                if line.startswith(f"| {prefix}-{n:02d} ")
            ]
            assert len(row) == 1, f"{prefix}-{n:02d}"
            cells = [c.strip() for c in row[0].strip("|").split("|")]
            assert len(cells) == 7 and all(
                cells
            )  # id, description, status, why, evidence, prereq, blocks
            assert cells[2].split()[0] in ("OPEN", "ENGINEERING_READY"), cells[
                2
            ]  # never DONE
            assert re.search(r"ENG|SCI|DEP", cells[6])
    for heading in (
        "TODO_EXTERNAL",
        "TODO_OWNER_DECISION",
        "TODO_SCIENTIFIC",
        "TODO_SECURITY",
        "IMPLEMENTABLE_NOW",
    ):
        assert f"## {heading}" in text
    amendment = "Physical collection and empirical validation are deferred until completion of the software architecture."
    for name in (
        "GAIT_VERIFICATION_TECHNICAL_PRD.md",
        "GAIT_VERIFICATION_REARCHITECTURE_HANDOFF.md",
    ):
        body = (REPO / "docs/gait_verification" / name).read_text()
        assert amendment in body
        assert "Phase 1D" in body  # historical scope preserved


# ================================================================ ingestion


@needs_ffmpeg
def test_ingest_report_from_explicit_metadata_keeps_layers_apart(tmp_path):
    w = build_world(tmp_path)
    report = ingest_report(w.registry, w.data, measure=True)
    assert report["ok"] and report["summary"]["originals"] == 4
    row = report["recordings"][0]
    assert set(row) >= {
        "media_validation",
        "operator_metadata",
        "pose_engineering_gate",
        "scientific_capture_suitability",
    }
    assert row["media_validation"]["measured"]["display_width"] == 240
    assert row["scientific_capture_suitability"]["status"] == "NOT_ASSESSED"
    assert row["pose_engineering_gate"]["provisional"] is True
    assert row["operator_metadata"]["consent_verified_by_program"] is False
    assert (
        "viewing_angle_definition" in row["operator_metadata"]["missing_fields"]
    )  # listed, not guessed
    text = json.dumps(report)
    assert (
        "CONSENT-" not in text and str(tmp_path) not in text
    )  # no consent refs or paths


@needs_ffmpeg
def test_declared_vs_measured_differences_are_flagged_not_hidden(tmp_path):
    w = build_world(tmp_path, subjects=("SYN001",))
    rows = [
        dict(r, nominal_fps=60, resolution="1920x1080", duration_sec=100.0)
        for r in w.rows
    ]
    report = ingest_report(registry_of(rows), w.data, measure=True)
    diff = report["recordings"][0]["media_validation"]["declared_vs_measured"]
    assert (
        diff["duration_sec"]["differs"]
        and diff["fps"]["differs"]
        and diff["resolution"]["differs"]
    )
    assert report["summary"]["declared_vs_measured_differences"] == {
        "duration_sec": 2,
        "fps": 2,
        "resolution": 2,
    }
    assert report["provisional_tolerances"]  # tolerances are labelled provisional


@pytest.mark.parametrize(
    "field",
    ["consent_ref", "session_id", "date", "subject_id", "sha256", "original_take_id"],
)
def test_missing_required_metadata_is_refused(field):
    rows = [raw_row("SYN001", "left45")]
    rows[0][field] = None
    with pytest.raises(ProvenanceError):
        registry_of(rows)
    assert diagnose_rows(rows) is not None  # diagnostics never raise


@pytest.mark.parametrize(
    "consent", [None, "", "unknown", "TBD-1", 123, ["x"], "placeholder"]
)
def test_invalid_consent_reference_fields_are_refused(consent):
    with pytest.raises(ProvenanceError):
        registry_of([raw_row("SYN001", "left45", consent_ref=consent)])


def test_invalid_and_conflicting_checksums_are_refused(tmp_path):
    for bad in ("abc", "G" * 64, "A" * 64, None, 7):
        with pytest.raises(ProvenanceError):
            registry_of([raw_row("SYN001", "left45", sha256=bad)])
    # An original whose bytes equal a registered transformation output is exposed.
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw/a.mp4").write_bytes(b"original")
    original = raw_row(
        "SYN001", "left45", sha256=sha256_text(b"original"), source_file_uri="raw/a.mp4"
    )
    child = derived_row(
        original, "vidstab-s10", sha256=sha("child"), source_file_uri="raw/a.mp4"
    )
    (tmp_path / "raw/a.manifest.json").write_text(
        json.dumps(
            {
                "status": "OK",
                "input": {"sha256": original["sha256"]},
                "output": {"sha256": "0" * 64},
            }
        )
    )
    child["transformation_manifest_ref"] = "raw/a.manifest.json"
    report = registry_of([original, child]).verify_files(tmp_path)
    assert not report["ok"] and any("checksum mismatch" in e for e in report["errors"])


def sha256_text(data):
    return hashlib.sha256(data).hexdigest()


def test_duplicate_originals_and_conflicting_identities_are_detected_with_asset_ids():
    a = raw_row("SYN001", "left45")
    same_bytes = raw_row("SYN002", "left45", sha256=a["sha256"])
    dup_take = dict(
        raw_row("SYN001", "right45"), original_take_id=a["original_take_id"]
    )
    for rows in ([a, same_bytes], [a, dict(a)], [a, dup_take]):
        with pytest.raises(ProvenanceError):
            registry_of(rows)
    codes = {i["code"] for i in diagnose_rows([a, same_bytes, dup_take, dict(a)])}
    assert {
        "DUPLICATE_BYTES",
        "DUPLICATE_ASSET_ID",
        "DUPLICATE_ORIGINAL_TAKE",
        "CONFLICTING_SUBJECTS_FOR_SAME_BYTES",
    } <= codes
    issue = next(
        i for i in diagnose_rows([a, same_bytes]) if i["code"] == "DUPLICATE_BYTES"
    )
    assert issue["asset_ids"] == sorted([a["asset_id"], same_bytes["asset_id"]])


def test_invalid_derivative_parent_relationships_are_refused():
    parent = raw_row("SYN001", "left45")
    other = raw_row("SYN002", "left45")
    good = derived_row(parent, "vidstab-s10")
    cases = [
        dict(good, parent_asset_id="no-such-asset"),
        dict(good, original_take_id=other["original_take_id"]),
        dict(good, subject_id="SYN002"),
        dict(good, parent_asset_id=good["asset_id"]),  # self-parent / cycle
        dict(good, variant_id="raw"),
    ]
    for bad in cases:
        with pytest.raises(ProvenanceError):
            registry_of([parent, other, bad])


@needs_ffmpeg
def test_cli_validate_reports_diagnostics_for_a_bad_registry(tmp_path):
    a = raw_row("SYN001", "left45")
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA,
                "assets": [a, raw_row("SYN002", "left45", sha256=a["sha256"])],
            }
        )
    )
    p = subprocess.run(
        CLI + ["validate", "--registry", str(path), "--out", str(tmp_path / "v.json")],
        capture_output=True,
        text=True,
    )
    assert p.returncode == 2
    codes = {
        d["code"] for d in json.loads((tmp_path / "v.json").read_text())["diagnostics"]
    }
    assert {"DUPLICATE_BYTES", "CONFLICTING_SUBJECTS_FOR_SAME_BYTES"} <= codes


# ================================================================ planning / dry run


@needs_ffmpeg
def test_plans_are_deterministic_and_never_mutate_anything(tmp_path):
    w = build_world(tmp_path)
    cfg = config(w, spec(w.policy))
    before = snapshot(w.data, w.work, w.gpu)
    a = build_plan(w.registry, w.data, w.work, w.ids, cfg, measure=True)
    b = build_plan(
        w.registry, w.data, w.work, list(reversed(w.ids)) + w.ids[:1], cfg, measure=True
    )
    assert snapshot(w.data, w.work, w.gpu) == before
    assert (
        not w.work.exists() and not w.gpu.exists()
    )  # no directory, lock or ledger created
    for p in (a, b):
        p["storage"].pop("disk_free_bytes")  # volatile by nature
    assert a == b and a["plan_id"] == b["plan_id"]
    keys = [(t["kind"], t["asset_id"], t["model"] or "") for t in a["tasks"]]
    assert keys == sorted(
        keys, key=lambda k: (("pose", "quality", "embed").index(k[0]), k[1], k[2])
    )
    assert a["selection"] == dict(
        requested=4,
        assets=4,
        originals=4,
        derivatives=0,
        subjects=2,
        sessions=2,
        views=["left45", "right45"],
    )
    assert a["task_states"] == {"MISSING": 4, "PENDING_UPSTREAM": 8}
    assert a["cuda"]["jobs_requiring_cuda"] == 8  # 4 pose + 4 embed
    assert a["storage"]["pose_cache_uncompressed_estimate_bytes"] > 0
    assert a["owner_decisions_needed_for_this_operation"] == []
    assert set(a["owner_decisions_unset"]) >= {"OWN-01", "OWN-05"}


@needs_ffmpeg
def test_explicit_scope_is_required_and_selection_is_selective(tmp_path):
    w = build_world(tmp_path)
    cfg = config(w)
    with pytest.raises(ValueError, match="Explicit scope"):
        build_plan(w.registry, w.data, w.work, [], cfg)
    chosen = w.ids[:2]
    plan = build_plan(w.registry, w.data, w.work, chosen + ["ghost"], cfg)
    assert plan["selection"]["assets"] == 2 and plan["selection"]["requested"] == 3
    assert plan["blocked"] == [dict(asset_id="ghost", reason="UNKNOWN_ASSET")]
    report = run(w, cfg, chosen)
    assert {r["asset_id"] for r in report["tasks"]} == set(chosen)
    assert (
        len(list((w.work / "pose").glob("*.npz"))) == 2
    )  # the other two were never touched


@needs_ffmpeg
def test_cli_plan_is_a_dry_run_and_names_the_models(tmp_path):
    w = build_world(tmp_path)
    reg = tmp_path / "registry.json"
    reg.write_text(json.dumps({"schema_version": SCHEMA, "assets": w.rows}))
    before = snapshot(w.data, w.work)
    args = [
        "--registry",
        reg,
        "--data-root",
        w.data,
        "--work-dir",
        w.work,
        "--asset",
        w.ids[0],
        "--model",
        "gaitpt",
    ]
    p = subprocess.run(
        CLI + ["plan", *map(str, args), "--out", str(tmp_path / "plan.json")],
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stdout + p.stderr
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert plan["models"][0][0] == "gaitpt" and plan["stages"] == [
        "pose",
        "quality",
        "embed",
    ]
    assert snapshot(w.data, w.work) == before and not w.work.exists()
    bad = subprocess.run(
        CLI
        + [
            "plan",
            "--registry",
            str(reg),
            "--data-root",
            str(w.data),
            "--work-dir",
            str(w.work),
        ],
        capture_output=True,
        text=True,
    )
    assert bad.returncode == 3 and "Explicit scope" in bad.stdout  # no implicit 'all'


# ================================================================ processing, resume, recovery


@needs_ffmpeg
def test_end_to_end_processing_then_resume_reuses_everything_and_preserves_originals(
    tmp_path,
):
    w = build_world(tmp_path)
    models = Factory(lambda: WindowedEncoder(StatEncoder(), w.policy))
    cfg = config(w, spec(w.policy, factory=models))
    originals = snapshot(w.data)
    first = run(w, cfg)
    assert first["status"] == "COMPLETE" and final(first) == {"DONE"}
    assert (
        first["task_counts"] == {"DONE": 12} and models.loads == 1
    )  # one load serves 4 videos
    assert first["originals"]["changed_during_processing"] == []
    assert (
        snapshot(w.data) == originals
    )  # bytes, mtimes and the directory tree are untouched
    artifacts = snapshot(w.work / "pose", w.work / "quality", w.work / "embeddings")
    detector = Detectors()
    second = run(w, cfg, detector=detector)
    assert final(second) == {"SKIPPED_VALID"} and second["status"] == "COMPLETE"
    assert (
        detector.loads == 0 and models.loads == 1
    )  # nothing was loaded: no CUDA needed to resume
    assert (
        snapshot(w.work / "pose", w.work / "quality", w.work / "embeddings")
        == artifacts
    )
    assert build_plan(w.registry, w.data, w.work, w.ids, cfg)["task_states"] == {
        "VALID": 12
    }
    events = w.work.joinpath("ledger.jsonl").read_text().splitlines()
    assert len(events) >= 24 and all(json.loads(e)["run_id"] for e in events)


@needs_ffmpeg
def test_interrupted_run_recovers_and_a_half_published_artifact_is_not_trusted(
    tmp_path,
):
    w = build_world(tmp_path)
    cfg = config(w)
    ticks = iter(range(100))
    interrupted = run(
        w, cfg, should_stop=lambda: next(ticks) >= 2
    )  # stop before the 3rd task
    assert interrupted["status"] == "INTERRUPTED"
    assert interrupted["task_counts"]["DONE"] == 2
    assert "NOT_RUN" in interrupted["task_counts"]
    # Crash between publishing the NPZ and its sidecar: the entry must not be reused.
    sidecars = sorted((w.work / "pose").glob("*.json"))
    sidecars[0].unlink()
    plan = build_plan(w.registry, w.data, w.work, w.ids, cfg)
    assert "INVALID" in plan["task_states"]
    resumed = run(w, cfg)
    assert resumed["status"] == "COMPLETE" and not any(
        s.startswith("FAILED") for s in resumed["task_counts"]
    )
    assert list((w.work / "pose").glob("*.npz.invalid-1"))  # preserved, never deleted
    assert build_plan(w.registry, w.data, w.work, w.ids, cfg)["task_states"] == {
        "VALID": 8
    }


@needs_ffmpeg
def test_transient_failure_is_retried_and_leaves_no_partial_artifact(tmp_path):
    w = build_world(tmp_path, subjects=("SYN001",))

    class OutOfMemory(FakeDetector):
        def __call__(self, frame):
            raise RuntimeError("CUDA out of memory")

    cfg = config(w)
    bad = run(w, cfg, detector=lambda: OutOfMemory())
    assert bad["status"] == "COMPLETE_WITH_FAILURES"
    assert {r["final_reason"] for r in bad["failures"] if r["kind"] == "pose"} == {
        "CUDA_OR_HOST_OUT_OF_MEMORY"
    }
    assert final(bad, "quality") == {
        "BLOCKED"
    }  # unusable upstream is not a quality verdict
    assert list((w.work / "pose").iterdir()) == []  # nothing partial
    good = run(w, cfg)
    assert good["status"] == "COMPLETE" and final(good) == {"DONE"}
    attempts = [
        json.loads(line) for line in (w.work / "ledger.jsonl").read_text().splitlines()
    ]
    assert [a["attempt"] for a in attempts if a["state"] == "FAILED_RETRYABLE"] == [
        1,
        1,
    ]


@needs_ffmpeg
def test_corrupt_and_truncated_media_fail_permanently_without_stopping_the_batch(
    tmp_path,
):
    w = build_world(tmp_path, subjects=("SYN001",))
    good = Path(next(iter(w.paths.values())))
    junk, cut = w.data / "raw/junk.mp4", w.data / "raw/cut.mp4"
    junk.write_bytes(b"\x00" * 4096)
    cut.write_bytes(good.read_bytes()[: good.stat().st_size // 3])
    extra = [
        raw_row(
            "SYN001",
            "toward",
            sha256=sha256_file(p),
            source_file_uri=f"raw/{p.name}",
            take=f"SYN001_sess01_toward_{p.stem}",
        )
        for p in (junk, cut)
    ]
    extra[1] = dict(extra[1], view_label="away")
    w.rows += extra
    w.registry = registry_of(w.rows)
    ids = sorted(r["asset_id"] for r in w.rows)
    cfg = config(w)
    first = run(w, cfg, ids)
    failed = {
        r["asset_id"]: r["final_reason"]
        for r in first["failures"]
        if r["kind"] == "pose"
    }
    assert set(failed.values()) == {"INVALID_MEDIA"} and len(failed) == 2
    assert first["status"] == "COMPLETE_WITH_FAILURES"
    assert (
        sum(r["final_state"] == "DONE" and r["kind"] == "pose" for r in first["tasks"])
        == 2
    )  # isolation
    detector = Detectors()
    again = run(w, cfg, ids, detector=detector)
    assert {r["final_reason"] for r in again["failures"] if r["kind"] == "pose"} == {
        "INVALID_MEDIA"
    }
    plan = build_plan(w.registry, w.data, w.work, ids, cfg)
    assert plan["task_states"]["PRIOR_PERMANENT_FAILURE"] == 2  # not blindly retried
    assert detector.loads == 0  # all valid work reused and permanent failures skipped
    run(w, cfg, ids, retry_permanent=True)
    assert [json.loads(line)["attempt"] for line in (w.work / "ledger.jsonl").read_text().splitlines()
            if '"FAILED_PERMANENT"' in line][-2:] == [2, 2]  # fmt: skip


@needs_ffmpeg
def test_changed_or_missing_source_blocks_tasks_and_is_never_repaired(tmp_path):
    w = build_world(tmp_path, subjects=("SYN001",))
    victim, other = w.ids
    w.paths[victim].write_bytes(b"edited after registration")
    plan = build_plan(w.registry, w.data, w.work, w.ids, config(w))
    assert plan["blocked"] == [dict(asset_id=victim, reason="SOURCE_CHECKSUM_MISMATCH")]
    report = run(w, config(w))
    assert {r["final_state"] for r in report["tasks"] if r["asset_id"] == victim} == {
        "BLOCKED"
    }
    assert w.paths[victim].read_bytes() == b"edited after registration"
    w.paths[victim].unlink()
    assert (
        build_plan(w.registry, w.data, w.work, [victim], config(w))["blocked"][0][
            "reason"
        ]
        == "SOURCE_MISSING"
    )
    assert other in {
        r["asset_id"] for r in report["tasks"] if r["final_state"] == "DONE"
    }


@needs_ffmpeg
def test_missing_or_ambiguous_skeletons_are_kept_as_unusable_recordings_not_mismatches(
    tmp_path,
):
    w = build_world(tmp_path)
    cfg = config(w, spec(w.policy))
    empty, crowded, ok1, ok2 = w.ids
    run(w, cfg, [empty], detector=Detectors(people=0))
    run(w, cfg, [crowded], detector=Detectors(people=2))
    report = run(w, cfg, [ok1, ok2])
    assert report["status"] == "COMPLETE"
    cov = coverage_report(w.registry, w.work, w.ids, cfg)
    by = {r["asset_id"]: r for r in cov["recordings"]}
    assert by[ok1]["pose_engineering_gate"]["status"] == "USABLE_BY_PROVISIONAL_GATE"
    for bad in (empty, crowded):
        gate = by[bad]["pose_engineering_gate"]
        assert gate["status"] == "NOT_USABLE_BY_PROVISIONAL_GATE" and gate["reasons"]
        assert (
            by[bad]["embeddings"]["stat"]["status"] == "INSUFFICIENT_GAIT"
        )  # persisted abstention
    assert by[crowded]["pose_engineering_gate"]["ambiguous_detection_fraction"] == 1.0
    assert by[empty]["pose_engineering_gate"]["person_detection_fraction"] == 0.0
    assert cov["pose_gate_outcomes"] == {
        "NOT_USABLE_BY_PROVISIONAL_GATE": 2,
        "USABLE_BY_PROVISIONAL_GATE": 2,
    }
    assert cov["pose_gate_reason_counts"]  # failure counts and reasons are retained
    assert cov["processing_failures"] == []  # unusable != processing failure
    assert "mismatch" not in json.dumps(cov["recordings"]).lower()
    assert by[empty]["scientific_capture_suitability"]["status"] == "NOT_ASSESSED"


# ================================================================ cache integrity


@needs_ffmpeg
def test_tampered_artifacts_are_detected_quarantined_and_rebuilt(tmp_path):
    w = build_world(tmp_path, subjects=("SYN001",))
    cfg = config(w, spec(w.policy))
    run(w, cfg)
    npz = sorted((w.work / "pose").glob("*.npz"))[0]
    data = bytearray(npz.read_bytes())
    data[len(data) // 2] ^= 0xFF
    npz.write_bytes(bytes(data))
    plan = build_plan(w.registry, w.data, w.work, w.ids, cfg)
    reasons = {t["reason"] for t in plan["tasks"] if t["state"] == "INVALID"}
    assert reasons == {"ARTIFACT_CHECKSUM_MISMATCH"}
    assert (
        plan["task_states"]["PENDING_UPSTREAM"] == 2
    )  # dependants of the bad pose artifact
    # The legacy extractor refuses it too, so enrollment and queries cannot consume it.
    asset = w.registry.asset(
        next(t["asset_id"] for t in plan["tasks"] if t["state"] == "INVALID")
    )
    with pytest.raises(CacheIntegrityError, match="Pose cache"):
        CachedPoseExtractor(
            FakeDetector(),
            w.work / "pose",
            asset.original_take_id,
            require_integrity=True,
        ).extract(w.paths[asset.asset_id])
    report = run(w, cfg)
    assert report["status"] == "COMPLETE" and list(
        (w.work / "pose").glob("*.npz.invalid-1")
    )
    # Editing a quality record or an embedding breaks its seal.
    q = sorted((w.work / "quality").glob("*.json"))[0]
    q.write_text(q.read_text().replace('"frames": ', '"frames": 1', 1))
    e = sorted((w.work / "embeddings").rglob("*.json"))[0]
    record = json.loads(e.read_text())
    record["values"][0] += 0.5
    e.write_text(json.dumps(record))
    plan = build_plan(w.registry, w.data, w.work, w.ids, cfg)
    assert {t["reason"] for t in plan["tasks"] if t["state"] == "INVALID"} == {
        "SEAL_MISMATCH"
    }
    assert run(w, cfg)["status"] == "COMPLETE"
    assert list(w.work.rglob("*.invalid-1"))


@needs_ffmpeg
def test_changed_pose_recipe_invalidates_or_isolates_pose_artifacts(
    tmp_path, monkeypatch
):
    w = build_world(tmp_path, subjects=("SYN001",))
    run(w, config(w))
    old = snapshot(w.work / "pose")
    # (a) another detector/checkpoint identity -> different artifact names; old ones untouched.
    newer = config(w, backend="fake-rtmpose:v2")
    assert build_plan(w.registry, w.data, w.work, w.ids, newer)["task_states"] == {
        "MISSING": 2,
        "PENDING_UPSTREAM": 2,
    }

    class V2(FakeDetector):
        backend_id = "fake-rtmpose:v2"

    assert run(w, newer, detector=lambda: V2())["status"] == "COMPLETE"
    assert all(snapshot(w.work / "pose")[k] == v for k, v in old.items())
    assert len(list((w.work / "pose").glob("*.npz"))) == 4
    # (b) changed extraction settings -> the existing entry is stale, not silently reused.
    from gait_verification import pipeline

    original = pipeline.extractor_config
    patched = lambda: dict(original(), max_center_jump=0.5)  # noqa: E731
    monkeypatch.setattr(pipeline, "extractor_config", patched)
    monkeypatch.setattr(dp, "extractor_config", patched)
    plan = build_plan(w.registry, w.data, w.work, w.ids, config(w))
    assert {t["reason"] for t in plan["tasks"] if t["state"] == "INVALID"} == {
        "STALE_EXTRACTOR"
    }
    assert run(w, config(w))["status"] == "COMPLETE"


@needs_ffmpeg
def test_changed_window_policy_reuses_pose_but_never_old_quality_or_embeddings(
    tmp_path,
):
    w = build_world(tmp_path, subjects=("SYN001",))
    detector = Detectors()
    run(w, config(w, spec(w.policy)), detector=detector)
    old = snapshot(w.work / "quality", w.work / "embeddings")
    other = replace(w.policy, min_valid_fraction=0.75)
    cfg = config(w, spec(other), policy=other)
    plan = build_plan(w.registry, w.data, w.work, w.ids, cfg)
    assert plan["task_states"] == {
        "MISSING": 4,
        "VALID": 2,
    }  # pose reused; the rest is new
    second = run(w, cfg, detector=detector)
    assert detector.loads == 1  # no repeated pose extraction
    assert second["status"] == "COMPLETE"
    now = snapshot(w.work / "quality", w.work / "embeddings")
    assert all(now[k] == v for k, v in old.items()) and len(now) > len(
        old
    )  # old kept, new added
    assert (
        len(list((w.work / "embeddings").iterdir())) == 2
    )  # one store directory per recipe


@needs_ffmpeg
def test_changed_checkpoint_and_swapped_weights_cannot_mix_embeddings(tmp_path):
    w = build_world(tmp_path, subjects=("SYN001",))
    a = spec(w.policy)
    run(w, config(w, a))
    old = snapshot(w.work / "embeddings")
    b = spec(
        w.policy, name="stat", cls=StatCkpt2
    )  # same name, different pinned checkpoint
    assert a.encoder_key != b.encoder_key
    assert run(w, config(w, b))["status"] == "COMPLETE"
    now = snapshot(w.work / "embeddings")
    assert (
        all(now[k] == v for k, v in old.items())
        and len(list((w.work / "embeddings").iterdir())) == 2
    )
    # The registry pins model X but the loaded weights identify as Y: refused, not stored.
    wrong = ModelSpec(
        "stat",
        "stat:recipe-pinned-elsewhere",
        Factory(lambda: WindowedEncoder(StatEncoder(), w.policy)),
    )
    report = run(w, config(w, wrong))
    assert {r["final_reason"] for r in report["failures"]} == {
        "ENCODER_DOES_NOT_MATCH_PLAN"
    }
    # A detector that is not the planned one is refused for the same reason.
    mismatch = run(build_world_copy(w, "other"), config(w, backend="fake-rtmpose:v9"))
    assert {r["final_reason"] for r in mismatch["failures"] if r["kind"] == "pose"} == {
        "DETECTOR_DOES_NOT_MATCH_PLAN"
    }


def build_world_copy(w, name):
    return SimpleNamespace(**{**vars(w), "work": w.tmp / name})


@needs_ffmpeg
def test_models_are_isolated_in_separate_stores(tmp_path):
    w = build_world(tmp_path)
    one, two = spec(w.policy), spec(w.policy, name="stat2", cls=Stat2)
    cfg = config(w, one, two)
    report = run(w, cfg)
    assert report["status"] == "COMPLETE" and report["task_counts"] == {"DONE": 16}
    dirs = sorted(p.name for p in (w.work / "embeddings").iterdir())
    assert (
        len(dirs) == 2 and dirs[0].startswith("stat-") and dirs[1].startswith("stat2-")
    )
    ids = {}
    for d in dirs:
        ids[d] = {
            json.loads(f.read_text())["model_id"].split(":")[0]
            for f in (w.work / "embeddings" / d).glob("*.json")
        }
    assert ids[dirs[0]] == {"stat"} and ids[dirs[1]] == {"stat2"}
    victim = sorted((w.work / "embeddings" / dirs[0]).glob("*.json"))[0]
    victim.write_text(victim.read_text().replace("EMBEDDED", "EMBEDDEX"))
    plan = build_plan(w.registry, w.data, w.work, w.ids, cfg)
    by_model = {}
    for t in plan["tasks"]:
        if t["kind"] == "embed":
            by_model.setdefault(t["model"], set()).add(t["state"])
    assert by_model == {"stat": {"VALID", "INVALID"}, "stat2": {"VALID"}}


# ================================================================ resources / CUDA


def test_gpu_slots_are_bounded_across_holders(tmp_path):
    with gpu_slot(tmp_path, 1):
        with pytest.raises(GpuBusy):
            with gpu_slot(tmp_path, 1):
                pass
    with gpu_slot(tmp_path, 1):  # released on exit
        pass
    with gpu_slot(tmp_path, 2) as first, gpu_slot(tmp_path, 2) as second:
        assert {first, second} == {0, 1}
        with pytest.raises(GpuBusy), gpu_slot(tmp_path, 2):
            pass
    for bad in (0, -1):
        with pytest.raises(ValueError), gpu_slot(tmp_path, bad):
            pass
    assert stat.S_IMODE(tmp_path.stat().st_mode) == 0o700


@needs_ffmpeg
def test_busy_gpu_loads_nothing_and_cuda_absence_never_falls_back_to_cpu(tmp_path):
    w = build_world(tmp_path, subjects=("SYN001",))
    models, detector = (
        Factory(lambda: WindowedEncoder(StatEncoder(), w.policy)),
        Detectors(),
    )
    cfg = config(w, spec(w.policy, factory=models))
    with gpu_slot(w.gpu, 1):
        busy = run(w, cfg, detector=detector)
    assert {r["final_reason"] for r in busy["failures"] if r["kind"] == "pose"} == {
        "GPU_BUSY"
    }
    unavailable = run(w, cfg, detector=detector, cuda_probe=lambda: False)
    assert {
        r["final_reason"] for r in unavailable["failures"] if r["kind"] == "pose"
    } == {"CUDA_UNAVAILABLE_NO_CPU_FALLBACK"}
    assert detector.loads == 0 and models.loads == 0  # no model was even constructed
    assert list((w.work / "pose").iterdir()) == []


def test_real_encoder_and_detector_refuse_cpu_substitution(monkeypatch):
    torch = pytest.importorskip("torch")
    from gait_verification.backends.gaitpt import GaitPTEncoder

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CPU substitution"):
        GaitPTEncoder("/nonexistent", "/nonexistent", "0" * 64)
    source = (REPO / "gait_verification/pose.py").read_text()
    assert "disable_fallback()" in source and "did not initialize with CUDA" in source


@needs_ffmpeg
def test_models_load_lazily_once_per_stage_and_are_released(tmp_path, monkeypatch):
    w = build_world(tmp_path)
    released = []
    monkeypatch.setattr(dp, "release_gpu_memory", lambda: released.append(1))
    models, detector = (
        Factory(lambda: WindowedEncoder(StatEncoder(), w.policy)),
        Detectors(),
    )
    run(w, config(w, spec(w.policy, factory=models)), detector=detector)
    assert (detector.loads, models.loads) == (1, 1) and len(
        released
    ) == 2  # detector + encoder
    only_pose = Detectors()
    run(build_world_copy(w, "w2"), config(w), detector=only_pose)
    assert only_pose.loads == 1


# ================================================================ owner decisions / protections


def test_no_owner_decision_has_a_default_anywhere():
    assert load_decisions() == {
        k: None for k in ("OWN-01", "OWN-02", "OWN-03", "OWN-04", "OWN-05")
    }
    config_text = (
        REPO / "configs/gait_verification/phase1d_evaluation.json"
    ).read_text()
    assert '"target_fars": null' in config_text and "0.05" not in config_text
    source = " ".join(
        (REPO / "gait_verification" / f).read_text()
        for f in ("dataset_processing.py", "dataset_ingest.py")
    )
    assert not re.search(
        r"assign_split|lock_split|revise_split|LockedTestAuthorization\(", source
    )


def test_library_calls_that_need_a_decision_name_it_instead_of_defaulting():
    from gait_verification.offline_eval import compare_runs, run_evaluation

    with pytest.raises(OwnerDecisionRequired, match="OWN-05"):
        run_evaluation({}, None, {}, None, {}, "SYNTHETIC_FIXTURE")
    with pytest.raises(OwnerDecisionRequired, match="OWN-05"):
        compare_runs(({}, []), ({}, []))


@needs_ffmpeg
def test_cli_asks_for_unset_decisions_only_when_an_operation_needs_them(tmp_path):
    rows = [
        raw_row(f"SYN{i:03d}", v)
        for i in range(1, 7)
        for v in ("left45", "right45", "toward", "away")
    ]
    reg = tmp_path / "registry.json"
    reg.write_text(json.dumps({"schema_version": SCHEMA, "assets": rows}))

    def cli(*args):
        p = subprocess.run(CLI + [str(a) for a in args], capture_output=True, text=True)
        return p.returncode, p.stdout

    code, out = cli("validate", "--registry", reg, "--out", tmp_path / "v.json")
    assert code == 0  # validation needs no split and no decision
    code, out = cli(
        "split",
        "--registry",
        reg,
        "--policy-version",
        "v1",
        "--out",
        tmp_path / "s.json",
    )
    assert code == 3 and "OWN-01" in out and not (tmp_path / "s.json").exists()
    code, out = cli("split", "--registry", reg, "--policy-version", "v1", "--development", 4, "--calibration", 1,
                    "--locked-final", 1, "--out", tmp_path / "s.json")  # fmt: skip
    assert code == 3 and "OWN-02" in out
    split = locked_split(registry_of(rows))
    (tmp_path / "split.json").write_text(json.dumps(split))
    code, out = cli("trials", "--registry", reg, "--split", tmp_path / "split.json", "--role", "development",
                    "--policy-version", "t", "--out", tmp_path / "t.json")  # fmt: skip
    assert code == 3 and "OWN-04" in out
    code, out = cli("trials", "--registry", reg, "--split", tmp_path / "split.json", "--role", "development",
                    "--policy-version", "t", "--seed", 1, "--out", tmp_path / "t.json")  # fmt: skip
    assert code == 3 and "OWN-03" in out
    code, out = cli("run", "--registry", reg, "--split", tmp_path / "split.json", "--trials", tmp_path / "x",
                    "--enrollments", tmp_path / "e", "--query-store", tmp_path / "q", "--model", "gaitpt",
                    "--evidence-class", "SYNTHETIC_FIXTURE", "--out", tmp_path / "run")  # fmt: skip
    assert code == 3  # fails before any scoring; the FAR decision is never defaulted
    code, out = cli("plan", "--help")
    assert "authoriz" not in out.lower()  # no CLI path to the protected group


@needs_ffmpeg
def test_locked_final_recordings_are_denied_to_processing_without_authorization(
    tmp_path,
):
    w = build_world(
        tmp_path, subjects=("SYN001", "SYN002", "SYN003"), views=("left45",)
    )
    split = locked_split(
        w.registry, {"development": 2, "calibration": 0, "locked_final": 1}
    )
    final_subject = next(
        s for s, r in split["assignments"].items() if r == "locked_final"
    )
    final_asset = next(a.asset_id for a in w.registry.originals(final_subject))
    cfg = config(w, spec(w.policy))
    plan = build_plan(w.registry, w.data, w.work, w.ids, cfg, split=split)
    assert plan["blocked"] == [
        dict(asset_id=final_asset, reason="LOCKED_FINAL_ACCESS_DENIED")
    ]
    assert plan["selection"]["assets"] == 2
    detector = Detectors()
    report = run(w, cfg, detector=detector, split=split)
    assert final_asset not in {r["asset_id"] for r in report["tasks"]}
    assert all(not list(w.work.rglob(f"*{final_asset}*")) for _ in (0,))
    cov = coverage_report(w.registry, w.work, w.ids, cfg, split)
    assert cov["locked_final_access_denied"] == [final_asset]
    # A subject missing from the supplied split is never assumed safe.
    partial = dict(
        split,
        assignments={k: v for k, v in split["assignments"].items() if k != "SYN001"},
    )
    assert any(
        b["reason"] == "LOCKED_FINAL_ACCESS_DENIED"
        for b in build_plan(w.registry, w.data, w.work, w.ids, cfg, split=partial)[
            "blocked"
        ]
    )
    # With no split, a registry-declared locked_final role is still denied.
    rows = [
        dict(r, split_role="locked_final") if r["asset_id"] == final_asset else r
        for r in w.rows
    ]
    declared = registry_of(rows)
    assert (
        build_plan(declared, w.data, w.work, w.ids, cfg)["blocked"][0]["reason"]
        == "LOCKED_FINAL_ACCESS_DENIED"
    )


# ================================================================ enrollment reuse / derivatives


@needs_ffmpeg
def test_batch_enrollment_is_idempotent_and_reuses_the_sealed_pose_cache(tmp_path):
    w = build_world(tmp_path, subjects=("SYN001", "SYN002"), views=("left45",))
    run(w, config(w))
    split = locked_split(
        w.registry, {"development": 2, "calibration": 0, "locked_final": 0}
    )
    store = DirectoryEnrollmentStore(tmp_path / "enroll")
    encoder = WindowedEncoder(StatEncoder(), w.policy)

    class Forbidden(FakeDetector):
        def __call__(self, frame):
            raise AssertionError(
                "pose was re-extracted instead of reusing the sealed cache"
            )

    def factory(take, variant):
        return CachedPoseExtractor(
            Forbidden(), w.work / "pose", take, variant, require_integrity=True
        )

    first = enroll_registered(
        w.registry, split, w.ids, w.data, factory, [encoder], store, w.policy
    )
    assert {r["status"] for r in first} == {"ENROLLED"}
    again = enroll_registered(
        w.registry, split, w.ids, w.data, factory, [encoder], store, w.policy
    )
    assert {r["status"] for r in again} == {"ALREADY_ENROLLED"}
    assert all(len(store.get_claim(s)) == 1 for s in ("SYN001", "SYN002"))


@needs_ffmpeg
def test_stabilized_pose_variant_is_part_of_the_recipe_and_cannot_match_raw_references():
    from gait_verification.contracts import PoseSequence

    n = 90
    xy = np.stack([POINTS.copy() for _ in range(n)])
    xy[:, 15, 0] += 8 * np.sin(np.arange(n) / 5)

    def seq(variant):
        return PoseSequence(xy, np.ones((n, 17)), np.ones((n, 17), bool), np.arange(n) / 30, "coco17", 640, 640,
                            "t", "a" * 64, "b", np.arange(n), np.zeros(n, int), np.zeros((n, 4)), variant)  # fmt: skip

    encoder = WindowedEncoder(StatEncoder(), WindowQualityPolicy())
    raw, stabilized = encoder.encode(seq("raw")), encoder.encode(seq("vidstab-s10"))
    assert raw.model_id != stabilized.model_id and stabilized.model_id.endswith(
        ":variant=vidstab-s10"
    )
    assert (
        ":variant=" not in raw.model_id
    )  # RAW recipe ids (and stored enrollments) are unchanged


def vidstab_available():
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True
    ).stdout
    return "vidstabdetect" in out


@needs_ffmpeg
@pytest.mark.skipif(
    not vidstab_available() if shutil.which("ffmpeg") else True,
    reason="libvidstab unavailable",
)
def test_registered_derivatives_have_provenance_resume_and_leave_originals_untouched(
    tmp_path,
):
    w = build_world(tmp_path, subjects=("SYN001",), views=("left45",))
    (asset,) = w.ids
    before = snapshot(w.data / "raw")
    results, rows = derive_variants(
        w.registry, w.data, [asset], ["vidstab-s10", "vidstab-s15"]
    )
    assert [r["status"] for r in results] == ["CREATED", "CREATED"] and len(rows) == 2
    assert (
        snapshot(w.data / "raw") == before
    )  # original untouched, derivatives elsewhere
    extended = registry_of(w.rows + rows)
    report = extended.verify_files(w.data, measure=True)
    assert report["ok"], report["errors"]
    assert {a.variant_id for a in extended.assets()} == {
        "raw",
        "vidstab-s10",
        "vidstab-s15",
    }
    assert all(extended.original(r["asset_id"]).asset_id == asset for r in rows)
    manifest = json.loads((w.data / rows[0]["transformation_manifest_ref"]).read_text())
    assert (
        manifest["status"] == "OK"
        and manifest["commands"]
        and manifest["ffmpeg_version"]
    )
    assert manifest["input"]["sha256"] == w.registry.asset(asset).sha256
    assert manifest["output"]["sha256"] == rows[0]["sha256"]
    again, rows2 = derive_variants(w.registry, w.data, [asset], ["vidstab-s10"])
    assert again[0]["status"] == "REUSED" and rows2[0]["sha256"] == rows[0]["sha256"]
    registered, none = derive_variants(extended, w.data, [asset], ["vidstab-s10"])
    assert registered[0]["status"] == "ALREADY_REGISTERED" and none == []
    # Process a derivative explicitly: its pose is a distinct artifact and its recipe differs.
    cfg = config(w, spec(w.policy))
    out = run(
        build_world_copy(SimpleNamespace(**{**vars(w), "registry": extended}), "wd"),
        cfg,
        [rows[0]["asset_id"]],
    )
    assert out["status"] == "COMPLETE" and out["task_counts"] == {"DONE": 3}
    warn = build_plan(extended, w.data, w.work, [rows[0]["asset_id"]], cfg)["warnings"]
    assert warn and "OWN-08" in warn[0]


@needs_ffmpeg
def test_derive_refuses_derivatives_raw_unknown_assets_and_unselected_work(tmp_path):
    w = build_world(tmp_path, subjects=("SYN001",), views=("left45",))
    (asset,) = w.ids
    for bad in ("raw", "vidstab-s99", "rotate"):
        with pytest.raises(ValueError, match="Unsupported variant"):
            derive_variants(w.registry, w.data, [asset], [bad])
    with pytest.raises(ValueError, match="Explicit"):
        derive_variants(w.registry, w.data, [], ["vidstab-s10"])
    results, rows = derive_variants(w.registry, w.data, ["ghost"], ["vidstab-s10"])
    assert (
        results[0]["status"] == "FAILED"
        and rows == []
        and not (w.data / "derived").exists()
    )
    failing = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ffmpeg failed"))  # noqa: E731
    results, rows = derive_variants(
        w.registry, w.data, [asset], ["vidstab-s10"], stabilizer=failing
    )
    assert results[0]["status"] == "FAILED" and rows == []
    assert w.paths[asset].read_bytes()  # original untouched after the failed attempt


@needs_ffmpeg
@pytest.mark.skipif(
    not vidstab_available() if shutil.which("ffmpeg") else True,
    reason="libvidstab unavailable",
)
def test_cli_derive_writes_a_new_validated_registry_and_never_edits_the_old_one(
    tmp_path,
):
    w = build_world(tmp_path, subjects=("SYN001",), views=("left45",))
    reg = tmp_path / "registry.json"
    reg.write_text(json.dumps({"schema_version": SCHEMA, "assets": w.rows}))
    old = reg.read_bytes()
    args = ["derive", "--registry", reg, "--data-root", w.data, "--asset", w.ids[0], "--variant", "vidstab-s10",
            "--out-registry", tmp_path / "registry2.json", "--out", tmp_path / "derive.json"]  # fmt: skip
    p = subprocess.run(CLI + [str(a) for a in args], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert reg.read_bytes() == old
    v = subprocess.run(CLI + ["validate", "--registry", str(tmp_path / "registry2.json"), "--data-root", str(w.data),
                              "--measure", "--out", str(tmp_path / "v.json")], capture_output=True, text=True)  # fmt: skip
    assert v.returncode == 0, v.stdout
    assert json.loads((tmp_path / "v.json").read_text())["summary"][
        "assets_by_variant"
    ] == {"raw": 1, "vidstab-s10": 1}


# ================================================================ privacy / repository safety


@needs_ffmpeg
def test_artifacts_are_private_and_reports_leak_no_biometrics_consent_or_paths(
    tmp_path,
):
    w = build_world(tmp_path)
    cfg = config(w, spec(w.policy))
    plan = build_plan(w.registry, w.data, w.work, w.ids, cfg)
    report = run(w, cfg)
    cov = coverage_report(w.registry, w.work, w.ids, cfg)
    ingest = ingest_report(w.registry, w.data, measure=False)
    vector = json.loads(next((w.work / "embeddings").rglob("*.json")).read_text())[
        "values"
    ]
    for doc in (plan, report, cov, ingest):
        text = json.dumps(doc)
        assert (
            "CONSENT-" not in text
            and str(tmp_path) not in text
            and '"values"' not in text
        )
        assert str(vector[0]) not in text
    for p in w.work.rglob("*"):
        mode = stat.S_IMODE(p.stat().st_mode)
        assert mode & 0o077 == 0, (p, oct(mode))  # no group/world access
    for p in (w.work, w.work / "pose", w.work / "runs"):
        assert stat.S_IMODE(p.stat().st_mode) == 0o700
    assert (
        report["restricted"]
        and cov["restricted"]
        and "not biometric performance evidence" in report["evidence_class"]
    )


def test_biometric_and_runtime_paths_are_git_ignored_and_untracked():
    for path in (
        "outputs/gait_verification/x/pose.npz",
        "data/gait_verification/raw/a.mp4",
        ".cache/gait_verification/q.json",
    ):
        assert (
            subprocess.run(["git", "check-ignore", "-q", path], cwd=REPO).returncode
            == 0
        ), path
    tracked = subprocess.run(
        [
            "git",
            "ls-files",
            "outputs/gait_verification",
            "data/gait_verification",
            ".cache",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    ).stdout
    assert tracked.strip() == ""


def test_path_components_and_registry_paths_cannot_escape(tmp_path):
    for bad in ("", ".hidden", "a/b", ".."):
        with pytest.raises(ProvenanceError):
            dp._safe_name(bad)
    w_row = raw_row("SYN001", "left45", source_file_uri="../outside.mp4")
    registry = registry_of([w_row])
    with pytest.raises(ProvenanceError):
        registry.path_of(registry.assets()[0], tmp_path)
    plan = build_plan(
        registry,
        tmp_path,
        tmp_path / "w",
        [registry.assets()[0].asset_id],
        config(SimpleNamespace(policy=WindowQualityPolicy())),
    )
    assert plan["blocked"][0]["reason"] == "SOURCE_PATH_REJECTED"


def test_ledger_ignores_a_torn_final_line_and_quarantine_never_overwrites(tmp_path):
    ledger = dp.Ledger(tmp_path / "ledger.jsonl")
    ledger.append(task_id="t", kind="pose", asset_id="a", state="DONE")
    with open(tmp_path / "ledger.jsonl", "a") as f:
        f.write('{"task_id": "t", "state": "FAIL')  # crash mid-write
    assert (
        ledger.latest()["t"]["state"] == "DONE"
        and stat.S_IMODE((tmp_path / "ledger.jsonl").stat().st_mode) == 0o600
    )
    target = tmp_path / "x.npz"
    for i in range(3):
        target.write_text(str(i))
        dp.quarantine(target)
    assert sorted(p.name for p in tmp_path.glob("x.npz.invalid-*")) == [
        "x.npz.invalid-1",
        "x.npz.invalid-2",
        "x.npz.invalid-3",
    ]


def test_query_store_rejects_an_edited_sealed_record_but_still_reads_legacy_records(
    tmp_path,
):
    from gait_verification.offline_eval import (
        QueryEmbeddingStore,
        RecordIntegrityError,
        seal,
    )

    store = QueryEmbeddingStore(tmp_path)
    base = dict(schema_version="query-embedding-v1", encoder_key="k", values=[1.0, 0.0])
    store.put(seal(dict(base, asset_id="sealed")))
    store.put(dict(base, asset_id="legacy"))  # Phase 1D record: no seal, still accepted
    assert store.get("legacy", "k")["values"] == [1.0, 0.0]
    path = store._path("sealed", "k")
    path.write_text(path.read_text().replace("1.0", "0.9", 1))
    with pytest.raises(RecordIntegrityError):
        store.get("sealed", "k")


def test_interrupt_signals_only_raise_a_flag_and_handlers_are_restored():
    import signal

    before = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
    with dp.interrupt_flag() as stop:
        assert not stop()
        signal.raise_signal(
            signal.SIGTERM
        )  # no exception: the run finishes its current task
        assert stop()
    assert (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)) == before


def test_stale_plan_id_refuses_to_execute_and_creates_nothing(tmp_path):
    w = SimpleNamespace(policy=WindowQualityPolicy())
    registry = registry_of([raw_row("SYN001", "left45")])
    with pytest.raises(ValueError, match="Plan ID differs"):
        run_processing(
            registry,
            tmp_path,
            tmp_path / "w",
            [registry.assets()[0].asset_id],
            config(w),
            expect_plan_id="0" * 32,
        )
    assert not (tmp_path / "w").exists()


def test_historical_e0_and_v2_import_isolation():
    diff = subprocess.run(
        ["git", "diff", "--stat", "HEAD", "--", "models", "utils", "scripts/training", "scripts/inference",
         "scripts/evaluation", "tests/test_future_work.py", "outputs/checkpoints"],
        cwd=REPO, capture_output=True, text=True,
    )  # fmt: skip
    assert diff.stdout.strip() == ""
    for name in ("dataset_processing.py", "dataset_ingest.py", "decisions.py"):
        text = (REPO / "gait_verification" / name).read_text()
        assert not re.search(r"^\s*(from|import)\s+(models|utils)\b", text, re.M)


# ================================================================ real CUDA engineering smoke (opt-in)


@pytest.mark.skipif(
    not CUDA, reason="set GAIT_V2_CUDA_TESTS=1 (needs GPU, ONNX and model caches)"
)
@needs_ffmpeg
def test_cuda_smoke_real_rtmpose_on_a_generated_clip_without_a_person(tmp_path):
    """Real RTMPose on a generated clip: engineering plumbing only (no person is present)."""
    from gait_verification.registry import load_json, load_quality_policy, load_rtmpose

    rt = load_json("configs/gait_verification/phase1a.json")["rtmpose"]
    backend = f"rtmpose-coco17:{rt['detector']['sha256']}:{rt['pose']['sha256']}"
    w = build_world(tmp_path, subjects=("SYN001",), views=("left45",))
    cfg = ProcessingConfig(backend, load_quality_policy(), (), ("pose", "quality"))
    report = run_processing(
        w.registry, w.data, w.work, w.ids, cfg,
        detector_factory=lambda: load_rtmpose("~/.cache/rtmlib/hub/checkpoints"), gpu_lock_dir=w.gpu,
    )  # fmt: skip
    assert report["status"] == "COMPLETE" and report["task_counts"] == {"DONE": 2}
    assert (
        report["runtime"]["gpu_slot"] == 0
        and report["runtime"]["pose_backend_id"] == backend
    )
    cov = coverage_report(w.registry, w.work, w.ids, cfg)
    assert cov["pose_gate_outcomes"] == {
        "NOT_USABLE_BY_PROVISIONAL_GATE": 1
    }  # a test pattern is not a walker
    again = run_processing(
        w.registry,
        w.data,
        w.work,
        w.ids,
        cfg,
        detector_factory=lambda: 1 / 0,
        gpu_lock_dir=w.gpu,
    )
    assert final(again) == {"SKIPPED_VALID"}  # resume needs no GPU at all


@pytest.mark.skipif(
    not CUDA, reason="set GAIT_V2_CUDA_TESTS=1 (needs GPU, ONNX and model caches)"
)
@pytest.mark.parametrize("model", ["gaitpt", "gpgait", "gpgaitpp", "gpgaitpp_casia"])
@needs_ffmpeg
def test_cuda_smoke_all_four_frozen_encoders_embed_through_the_runner(tmp_path, model):
    """Synthetic walker skeleton + real frozen CUDA encoder: no accuracy or ranking is implied."""
    from gait_verification.registry import load_encoder, model_entry

    w = build_world(tmp_path, subjects=("SYN001",), views=("left45",))
    policy = w.policy
    entry = model_entry(model)
    encoder = ModelSpec(
        model, entry["model_id"],
        lambda: load_encoder(model, REPO / ".cache/gait_verification/sources", REPO / ".cache/gait_verification/checkpoints", policy),
    )  # fmt: skip
    cfg = config(w, encoder)
    report = run(w, cfg, detector=Detectors(), cuda_probe=dp.cuda_available)
    assert report["status"] == "COMPLETE", report["failures"]
    assert report["runtime"]["models"][model]["model_id"] == entry["model_id"]
    record = json.loads(next((w.work / "embeddings").rglob("*.json")).read_text())
    assert (
        record["status"] == "EMBEDDED"
        and record["model_id"].startswith(entry["model_id"])
        and ":windows=" in record["model_id"]
    )
    assert len(record["values"]) in (256, 4864)
