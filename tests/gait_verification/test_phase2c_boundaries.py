"""Adversarial engineering checks; no biometric or scientific evidence."""

import json
import os
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
import test_phase1d as fixtures

from gait_verification.contracts import GaitEmbedding, Status, VerificationResult
from gait_verification.dataset_processing import Ledger, source_issue
from gait_verification.pipeline import (
    EnrollmentRefused,
    build_template,
    enroll_original,
    verify_registered,
)
from gait_verification.enrollments import DirectoryEnrollmentStore
from gait_verification.media import sha256_file
from gait_verification.pose_quality import WindowQualityPolicy, WindowedEncoder
from test_phase1c import StatEncoder
from test_pilot import moving_pose
from gait_verification.pose import save_pose
from test_foundation import pose


def registered(tmp_path, role=None):
    row = fixtures.raw_row("SYN001", "left45", split_role=role)
    path = tmp_path / row["source_file_uri"]
    path.parent.mkdir(parents=True)
    path.write_bytes(row["original_take_id"].encode())
    return fixtures.make_registry([row]), path


def scored_factory(calls):
    def factory(take, variant):
        calls.append((take, variant))
        return SimpleNamespace(
            verify_detailed=lambda *args: (
                VerificationResult("SYN001", Status.POLICY_NOT_READY),
                {"reason": "NO_POLICY", "score": 0.5},
            )
        )

    return factory


def test_managed_query_refuses_declared_locked_final_before_factory(tmp_path):
    registry, path = registered(tmp_path, "locked_final")
    calls = []
    split = fixtures.locked_split(
        registry, {"development": 0, "calibration": 0, "locked_final": 1}
    )
    result, diag = verify_registered(
        registry, path, "SYN001", scored_factory(calls), data_root=tmp_path, split=split
    )
    assert result.status == Status.INVALID_INPUT and result.verdict is None
    assert diag["reason"] == "LOCKED_FINAL_ACCESS_DENIED"
    assert calls == []


def test_managed_invalid_claim_never_loads_a_model(tmp_path):
    registry, path = registered(tmp_path)
    calls = []
    result, _ = verify_registered(registry, path, "../bad", scored_factory(calls))
    assert result.status == Status.INVALID_INPUT
    assert calls == []


def test_managed_query_reports_model_setup_failure(tmp_path):
    registry, path = registered(tmp_path)

    def unavailable(*args):
        raise RuntimeError("private checkpoint path")

    result, diag = verify_registered(
        registry,
        path,
        "SYN001",
        unavailable,
        data_root=tmp_path,
        split=fixtures.locked_split(
            registry, {"development": 1, "calibration": 0, "locked_final": 0}
        ),
    )
    assert result.status == Status.PROCESSING_ERROR and result.verdict is None
    assert "private" not in json.dumps(diag)


@pytest.mark.parametrize("damage", ["parent", "manifest"])
def test_managed_query_validates_derivative_ancestry(tmp_path, damage):
    registry, path = registered(tmp_path)
    parent = registry.document["assets"][0]
    child = fixtures.derived_row(parent, "reencoded")
    output = tmp_path / child["source_file_uri"]
    output.parent.mkdir(parents=True)
    output.write_bytes(child["asset_id"].encode())
    manifest = tmp_path / child["transformation_manifest_ref"]
    manifest.write_text(
        json.dumps(
            {
                "status": "OK",
                "input": {"sha256": parent["sha256"]},
                "output": {"sha256": child["sha256"]},
            }
        )
    )
    registry = fixtures.make_registry([parent, child])
    if damage == "parent":
        path.write_bytes(b"substituted source")
    else:
        manifest.write_text("[]")
    calls = []
    result, diag = verify_registered(
        registry,
        output,
        "SYN001",
        scored_factory(calls),
        data_root=tmp_path,
        split=fixtures.locked_split(
            registry, {"development": 1, "calibration": 0, "locked_final": 0}
        ),
    )
    assert result.status == Status.INVALID_INPUT and result.verdict is None
    assert calls == [] and diag["reason"] in {
        "ANCESTOR_CHECKSUM_MISMATCH",
        "SOURCE_PROVENANCE_INVALID",
    }


@pytest.mark.parametrize("link", ["symbolic", "hard"])
def test_ledger_never_writes_through_a_link_to_an_original(tmp_path, link):
    original = tmp_path / "original.mp4"
    original.write_bytes(b"immutable original bytes")
    ledger = tmp_path / "ledger.jsonl"
    if link == "symbolic":
        ledger.symlink_to(original)
    else:
        os.link(original, ledger)
    before = original.read_bytes()
    with pytest.raises((OSError, ValueError)):
        Ledger(ledger).append(task_id="task", state="DONE")
    assert original.read_bytes() == before


def test_malformed_ledger_events_do_not_break_recovery(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text(
        'null\n[]\n{"state":"FAILED_RETRYABLE"}\n{"task_id":"task","state":"DONE"}\n'
    )
    assert Ledger(path).latest() == {"task": {"task_id": "task", "state": "DONE"}}


def test_pose_cache_is_private_without_relying_on_umask(tmp_path):
    path = tmp_path / "pose.npz"
    old = os.umask(0o022)
    try:
        save_pose(pose(), path)
    finally:
        os.umask(old)
    assert path.stat().st_mode & 0o777 == 0o600


def test_malformed_transform_is_reported_instead_of_crashing(tmp_path):
    registry, _ = registered(tmp_path)
    parent = registry.document["assets"][0]
    child = fixtures.derived_row(parent, "reencoded")
    output = tmp_path / child["source_file_uri"]
    output.parent.mkdir(parents=True)
    output.write_bytes(child["asset_id"].encode())
    (tmp_path / child["transformation_manifest_ref"]).write_text("[]")
    registry = fixtures.make_registry([parent, child])
    assert not registry.verify_files(tmp_path)["ok"]
    assert source_issue(registry, registry.asset(child["asset_id"]), tmp_path)


def test_enrollment_never_relabels_a_derivative_as_raw():
    sequence = replace(pose(), variant_id="vidstab-s10")
    embedding = GaitEmbedding(
        [1.0], "synthetic", sequence.original_take_id, sequence.source_sha256
    )
    with pytest.raises(EnrollmentRefused, match="original"):
        build_template(
            sequence,
            None,
            embedding,
            None,
            None,
            subject_id="SYN001",
            take_id=sequence.original_take_id,
            session_id="synthetic-session",
            view="left45",
            recorded_at="2026-10-01T10:00:00+05:30",
            consent_ref="synthetic-fixture",
        )


def test_enrollment_refuses_source_substitution_before_publication(tmp_path):
    path = tmp_path / "synthetic-source.mp4"
    path.write_bytes(b"synthetic-original")
    sequence = replace(moving_pose(), source_sha256=sha256_file(path))

    def extract(_):
        path.write_bytes(b"substituted-synthetic-source")
        return sequence

    store = DirectoryEnrollmentStore(tmp_path / "enrollments")
    policy = WindowQualityPolicy()
    with pytest.raises(EnrollmentRefused, match="source"):
        enroll_original(
            path,
            SimpleNamespace(extract=extract),
            [WindowedEncoder(StatEncoder(), policy)],
            store,
            policy,
            subject_id="SYN001",
            take_id=sequence.original_take_id,
            session_id="synthetic-session",
            view="left45",
            recorded_at="2026-10-01T10:00:00+05:30",
            consent_ref="synthetic-fixture",
        )
    assert not store.get_claim("SYN001")


def test_unenrolled_managed_cli_never_loads_models(tmp_path, monkeypatch, capsys):
    from scripts.gait_verification import verify_claim

    registry, path = registered(tmp_path)
    registry_path, split_path = tmp_path / "registry.json", tmp_path / "split.json"
    registry_path.write_text(json.dumps(registry.document))
    split_path.write_text(
        json.dumps(
            fixtures.locked_split(
                registry, {"development": 1, "calibration": 0, "locked_final": 0}
            )
        )
    )

    def forbidden(*args):
        pytest.fail("Unenrolled claim loaded a CUDA model")

    monkeypatch.setattr(verify_claim, "load_encoder", forbidden)
    monkeypatch.setattr(verify_claim, "load_rtmpose", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_claim.py",
            "--video",
            str(path),
            "--claimed-id",
            "SYN001",
            "--model",
            "gaitpt",
            "--enrollments",
            str(tmp_path / "enroll"),
            "--capture-registry",
            str(registry_path),
            "--split",
            str(split_path),
            "--data-root",
            str(tmp_path),
        ],
    )
    assert verify_claim.main() == 2
    assert json.loads(capsys.readouterr().out)["status"] == "CLAIM_NOT_ENROLLED"


def test_stabilization_manifests_are_private_under_a_shared_parent(tmp_path):
    from gait_verification.contracts import Recording
    from gait_verification.stabilization import stabilize
    from test_phase1c import make_video

    source = make_video(tmp_path / "synthetic.mp4", 1.0)
    recording = Recording(
        "synthetic",
        "synthetic",
        "SYN001",
        "synthetic-session",
        "left45",
        "2026-10-01T10:00:00+05:30",
        sha256_file(source),
        "development",
    )
    work = tmp_path / "derivative"
    old = os.umask(0o022)
    try:
        stabilize(source, work, recording)
    finally:
        os.umask(old)
    assert work.stat().st_mode & 0o777 == 0o700
    assert (work / "manifest.json").stat().st_mode & 0o777 == 0o600
