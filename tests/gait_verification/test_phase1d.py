"""Phase 1D: capture registry, splits, trials, metrics and offline evaluation.

Evidence types are kept apart:
  * pure unit tests        - registry/split/metric logic on literal values
  * synthetic cohort tests - invented pseudonymous subjects and vectors; they test
                             contracts only and are NOT biometric performance evidence
  * media integration      - real ffmpeg / vid.stab files with an injected fake
                             detector and stand-in encoder (no real gait model)
Real new-cohort scientific evaluation is unavailable (no cohort exists).
"""

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pytest
from gait_verification.capture_registry import (
    SCHEMA,
    CaptureRegistry,
    ProvenanceError,
    load_capture_schema,
)
from gait_verification.contracts import GaitEmbedding, Recording, Status
from gait_verification.enrollments import (
    DirectoryEnrollmentStore,
    EnrollmentTemplate,
    MemoryEnrollmentStore,
)
from gait_verification.evaluation import (
    TrialPolicy,
    generate_verification_trials,
    load_trials,
    manifest_hash,
    paired_comparison,
    subject_bootstrap,
    validate_trial_manifest,
    verification_metrics,
)
from gait_verification.offline_eval import (
    ClaimRequest,
    EmbeddingTrialScorer,
    QueryEmbeddingStore,
    compare_runs,
    read_run,
    run_evaluation,
    write_run,
)
from gait_verification.splits import (
    LockedTestAccessDenied,
    LockedTestAuthorization,
    SplitPolicy,
    assign_split,
    check_transition,
    lock_split,
    require_roles,
    revise_split,
    subjects_for,
    validate_split,
)
from gait_verification.verifier import VerificationPolicy

REPO = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 10, 12, 12, tzinfo=timezone.utc)
CAPTURE = load_capture_schema()
VIEWS = ("left45", "right45", "toward", "away")
RECIPE = "stat:recipe:pipeline=p:windows=w"
ENCODER_KEY = "stat:recipe"


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def raw_row(subject, view, session="sess01", day="2026-10-01", **kw):
    take = kw.pop("take", f"{subject}_{session}_{view}_take01")
    row = {k: None for k in CAPTURE["required_fields"]}
    row.update(
        asset_id=f"asset-{take}",  # deliberately differs from the take ID
        variant_id="raw",
        parent_asset_id=None,
        split_role=None,
        original_take_id=take,
        subject_id=subject,
        session_id=f"{subject}_{session}",
        date=f"{day}T10:00:00+05:30",
        view_label=view,
        source_file_uri=f"raw/{take}.mp4",
        sha256=sha(take),
        consent_ref=f"CONSENT-{subject}",
        actual_frame_timestamps_ref=f"ffprobe/{take}.json",
        duration_sec=30.0,
    )
    row.update(kw)
    return row


def derived_row(parent, variant, root=None, **kw):
    root = root or parent
    asset = f"{parent['asset_id']}.{variant}"
    row = dict(
        asset_id=asset,
        variant_id=variant,
        parent_asset_id=parent["asset_id"],
        original_take_id=root["original_take_id"],
        subject_id=root["subject_id"],
        sha256=sha(asset),
        source_file_uri=f"derived/{asset}.mp4",
        transformation_manifest_ref=f"derived/{asset}.manifest.json",
    )
    row.update(kw)
    return row


def cohort_rows(n=6, views=VIEWS):
    return [raw_row(f"S{i:03d}", v) for i in range(1, n + 1) for v in views]


def make_registry(rows):
    return CaptureRegistry({"schema_version": SCHEMA, "assets": rows}, CAPTURE, NOW)


def locked_split(registry, counts=None, seed=7, pinned=()):
    counts = counts or {"development": 4, "calibration": 1, "locked_final": 1}
    policy = SplitPolicy("split-test-v1", seed, counts, tuple(pinned))
    return lock_split(assign_split(registry.subjects(), policy), "tester", "2026-10-12")


def trial_policy(k=2, seed=3, **kw):
    return TrialPolicy("view-heldout-test-v1", seed, k, **kw)


# ============================================================ pure unit: registry


def test_valid_registry_parsing_summary_and_order_independent_fingerprint():
    rows = cohort_rows(3)
    rows.append(derived_row(rows[0], "vidstab-s10"))
    reg = make_registry(rows)
    again = make_registry(list(reversed(rows)))
    assert reg.fingerprint == again.fingerprint
    summary = reg.summary()
    assert summary["subjects"] == 3 and summary["originals"] == 12
    assert summary["assets_by_variant"] == {"raw": 12, "vidstab-s10": 1}
    # A consent reference is a pointer, never program-verified consent.
    assert summary["consent_verified_by_program"] is False
    child = reg.asset(rows[-1]["asset_id"])
    assert (child.subject_id, child.session_id, child.view) == (
        "S001",
        "S001_sess01",
        "left45",
    )


def _bad(kind):
    rows = cohort_rows(2)
    first = rows[0]
    child = derived_row(first, "vidstab-s10")
    if kind == "duplicate_asset":
        rows.append(dict(rows[1], asset_id=first["asset_id"]))
    elif kind == "conflicting_take":
        rows.append(raw_row("S001", "left45", asset_id="other", sha256=sha("x")))
    elif kind == "same_bytes_twice":
        rows.append(raw_row("S002", "toward", session="sess02", sha256=first["sha256"]))
    elif kind == "derivative_subject":
        rows.append(dict(child, subject_id="S002"))
    elif kind == "derivative_split":
        rows[0] = dict(first, split_role="development")
        rows.append(derived_row(rows[0], "vidstab-s10", split_role="locked_final"))
    elif kind == "derivative_session":
        rows.append(dict(child, session_id="S001_sess09"))
    elif kind == "derivative_take":
        rows.append(dict(child, original_take_id=rows[1]["original_take_id"]))
    elif kind == "unresolved_parent":
        rows.append(dict(child, parent_asset_id="missing"))
    elif kind == "cycle":
        a = dict(child, asset_id="a", parent_asset_id="b", sha256=sha("a"))
        b = dict(child, asset_id="b", parent_asset_id="a", sha256=sha("b"))
        rows += [a, b]
    elif kind == "derivative_as_original":
        rows.append(
            raw_row("S001", "left45", take="copy", transformation_manifest_ref="m")
        )
    elif kind == "derivative_without_manifest":
        rows.append(dict(child, transformation_manifest_ref=None))
    elif kind == "derivative_same_bytes":
        rows.append(dict(child, sha256=first["sha256"]))
    elif kind == "missing_consent":
        rows[0] = dict(first, consent_ref="")
    elif kind == "placeholder_session":
        rows[0] = dict(first, session_id="S001_unknown")
    elif kind == "naive_time":
        rows[0] = dict(first, date="2026-10-01T10:00:00")
    elif kind == "future_time":
        rows[0] = dict(first, date="2027-01-01T10:00:00+05:30")
    elif kind == "session_spans_days":
        rows[1] = dict(rows[1], date="2026-10-02T10:00:00+05:30")
    elif kind == "unknown_field":
        rows[0] = dict(first, subjectid="S001")
    elif kind == "path_like_subject":
        rows[0] = dict(first, subject_id="../S001")
    elif kind == "bad_split_role":
        rows[0] = dict(first, split_role="test")
    return rows


@pytest.mark.parametrize(
    "kind",
    [
        "duplicate_asset",
        "conflicting_take",
        "same_bytes_twice",
        "derivative_subject",
        "derivative_split",
        "derivative_session",
        "derivative_take",
        "unresolved_parent",
        "cycle",
        "derivative_as_original",
        "derivative_without_manifest",
        "derivative_same_bytes",
        "missing_consent",
        "placeholder_session",
        "naive_time",
        "future_time",
        "session_spans_days",
        "unknown_field",
        "path_like_subject",
        "bad_split_role",
    ],
)
def test_registry_rejects_conflicts_and_missing_provenance(kind):
    with pytest.raises(ProvenanceError):
        make_registry(_bad(kind))


def test_registry_rejects_unsupported_schema_and_empty_documents():
    for doc in (
        {"schema_version": "capture-registry-v0", "assets": cohort_rows(1)},
        {"schema_version": SCHEMA, "assets": []},
        [],
    ):
        with pytest.raises(ProvenanceError):
            CaptureRegistry(doc, CAPTURE, NOW)


def test_hash_resolution_ignores_filenames(tmp_path):
    rows = cohort_rows(2)
    data = {r["asset_id"]: f"bytes of {r['asset_id']}".encode() for r in rows}
    rows = [
        dict(r, sha256=hashlib.sha256(data[r["asset_id"]]).hexdigest()) for r in rows
    ]
    reg = make_registry(rows)
    target = rows[3]
    misleading = tmp_path / "S002_sess01_away_take01.mp4"  # names another walker
    misleading.write_bytes(data[target["asset_id"]])
    assert reg.resolve_file(misleading).asset_id == target["asset_id"]
    looks_registered = tmp_path / f"{target['asset_id']}.mp4"
    looks_registered.write_bytes(b"edited or re-encoded bytes")
    assert reg.resolve_file(looks_registered) is None  # never guessed from the name
    with pytest.raises(ProvenanceError):
        reg.path_of(
            replace(reg.asset(target["asset_id"]), source_uri="../x.mp4"), tmp_path
        )


def test_multi_level_derivative_ancestry_and_lineage():
    rows = cohort_rows(2)
    s10 = derived_row(rows[0], "vidstab-s10")
    reencode = derived_row(s10, "reencode-crf30", root=rows[0])
    reg = make_registry(rows + [s10, reencode])
    chain = reg.ancestry(reencode["asset_id"])
    assert [a.asset_id for a in chain] == [
        reencode["asset_id"],
        s10["asset_id"],
        rows[0]["asset_id"],
    ]
    assert reg.original(reencode["asset_id"]).asset_id == rows[0]["asset_id"]
    lineage = reg.lineage_ids(reencode["asset_id"])
    assert {rows[0]["original_take_id"], rows[0]["sha256"], s10["sha256"]} <= lineage
    assert (
        reg.variant_of(rows[0]["original_take_id"], "vidstab-s10").asset_id
        == s10["asset_id"]
    )


def _write_manifest(root, row, parent_sha, output_sha=None, status="OK"):
    path = root / row["transformation_manifest_ref"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": status,
                "input": {"sha256": parent_sha},
                "output": {"sha256": output_sha or row["sha256"]},
            }
        )
    )


def test_verify_files_checks_bytes_transformations_and_disguised_originals(tmp_path):
    rows = cohort_rows(1, views=("left45", "right45"))
    child = derived_row(rows[0], "vidstab-s10")
    content = {}
    for r in rows + [child]:
        content[r["asset_id"]] = f"video {r['asset_id']}".encode()
        r["sha256"] = hashlib.sha256(content[r["asset_id"]]).hexdigest()
        path = tmp_path / r["source_file_uri"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content[r["asset_id"]])
    _write_manifest(tmp_path, child, rows[0]["sha256"])
    report = make_registry(rows + [child]).verify_files(tmp_path)
    assert report["ok"], report["errors"]
    assert "subject_id" in report["operator_supplied_not_verified"]
    # Transformation input that is not the declared parent.
    _write_manifest(tmp_path, child, rows[1]["sha256"])
    assert not make_registry(rows + [child]).verify_files(tmp_path)["ok"]
    _write_manifest(tmp_path, child, rows[0]["sha256"])
    # The derivative's bytes registered as an independent original are exposed.
    disguised = raw_row(
        "S001",
        "toward",
        sha256=child["sha256"],
        source_file_uri=child["source_file_uri"],
        take="S001_sess01_toward_take01",
    )
    other = dict(child, sha256=sha("other"))
    _write_manifest(tmp_path, other, rows[0]["sha256"], output_sha=child["sha256"])
    report = make_registry(rows + [other, disguised]).verify_files(tmp_path)
    assert any("transformation output" in e for e in report["errors"])
    # Altered source bytes.
    (tmp_path / rows[1]["source_file_uri"]).write_bytes(b"changed")
    assert any(
        "checksum" in e for e in make_registry(rows).verify_files(tmp_path)["errors"]
    )


# ============================================================ pure unit: splits


def test_split_is_deterministic_disjoint_and_never_defaults_counts():
    reg = make_registry(cohort_rows(6))
    policy = SplitPolicy(
        "v1", 11, {"development": 3, "calibration": 2, "locked_final": 1}
    )
    a = assign_split(reg.subjects(), policy)
    b = assign_split(list(reversed(reg.subjects())), policy)
    assert a == b and a["status"] == "DRAFT"
    assert sorted(a["assignments"]) == list(reg.subjects())  # exactly one role each
    assert sorted(a["assignments"].values()).count("development") == 3
    other = assign_split(reg.subjects(), replace(policy, seed=12))
    assert other["fingerprint"] != a["fingerprint"]
    for counts in (
        {"development": 4, "calibration": 1, "locked_final": 0},  # does not sum
        {"development": 6},
        {"development": 5, "calibration": 1, "locked_final": -0.0},
    ):
        with pytest.raises(ValueError):
            assign_split(reg.subjects(), replace(policy, counts=counts))
    config = json.loads(
        (REPO / "configs/gait_verification/phase1d_evaluation.json").read_text()
    )
    assert all(v is None for v in config["split_policy"]["counts"].values())
    assert config["split_policy"]["seed"] is None


def test_pinned_pilot_subjects_stay_in_development():
    subjects = [f"S{i:03d}" for i in range(1, 7)]
    counts = {"development": 2, "calibration": 2, "locked_final": 2}
    for seed in range(10):
        s = assign_split(subjects, SplitPolicy("v1", seed, counts, ("S004", "S005")))
        assert s["assignments"]["S004"] == s["assignments"]["S005"] == "development"


def test_tampering_lock_and_protocol_revision():
    reg = make_registry(cohort_rows(6))
    split = locked_split(reg)
    validate_split(split, reg.subjects())
    tampered = json.loads(json.dumps(split))
    s = next(k for k, v in tampered["assignments"].items() if v == "development")
    tampered["assignments"][s] = "calibration"
    with pytest.raises(ValueError, match="modified"):
        validate_split(tampered)
    with pytest.raises(ValueError):
        check_transition(split, tampered)
    with pytest.raises(ValueError):
        lock_split(split, "someone")  # already locked
    revised = revise_split(
        split, {s: "calibration"}, "rev-1", "pilot reassigned", "owner"
    )
    check_transition(split, revised)
    assert revised["revisions"][0]["previous_fingerprint"] == split["fingerprint"]
    locked_subject = next(
        k for k, v in split["assignments"].items() if v == "locked_final"
    )
    for moves in ({locked_subject: "development"}, {s: "locked_final"}):
        with pytest.raises(ValueError, match="Locked-final"):
            revise_split(split, moves, "rev-2", "x", "owner")
    with pytest.raises(ValueError):
        validate_split(split, reg.subjects()[:-1])


def test_locked_test_access_requires_explicit_authorization():
    reg = make_registry(cohort_rows(6))
    split = locked_split(reg)
    assert set(subjects_for(split, ("development",))) <= set(reg.subjects())
    good = LockedTestAuthorization(split["fingerprint"], "a" * 64, "owner", "final")
    for auth in (
        None,
        replace(good, split_fingerprint="0" * 64),
        replace(good, frozen_bundle_sha256="not-a-hash"),
        replace(good, authorized_by=""),
    ):
        with pytest.raises(LockedTestAccessDenied):
            subjects_for(split, ("development", "locked_final"), auth)
    draft = assign_split(
        reg.subjects(), SplitPolicy("v1", 1, split["policy"]["counts"])
    )
    with pytest.raises(LockedTestAccessDenied):
        require_roles(
            draft,
            ("locked_final",),
            replace(good, split_fingerprint=draft["fingerprint"]),
        )
    # Synthetic-only check that a well-formed authorization object is recognized.
    assert require_roles(split, ("locked_final",), good) == ("locked_final",)
    with pytest.raises(LockedTestAccessDenied):
        generate_verification_trials(reg, split, trial_policy(), "locked_final")


# ============================================================ synthetic cohort: trials


def test_trial_generation_is_deterministic_and_leakage_free():
    rows = cohort_rows(6)
    reg = make_registry(rows)
    split = locked_split(reg)
    doc = generate_verification_trials(reg, split, trial_policy(k=2), "development")
    again = generate_verification_trials(
        make_registry(list(reversed(rows))), split, trial_policy(k=2), "development"
    )
    assert doc == again and doc["manifest_sha256"] == manifest_hash(load_trials(doc))
    trials = load_trials(doc)
    dev = set(subjects_for(split, ("development",)))
    assert doc["counts"]["genuine"] == len(dev) * 4  # one per query original
    assert doc["counts"]["impostor"] == len(dev) * 4 * 2
    assert doc["counts"]["impostor_shortfall"] == 0
    for t in trials:
        assert t.query_subject_id in dev and t.claimed_subject_id in dev
        assert t.label_same_identity == (t.claimed_subject_id == t.query_subject_id)
        assert t.query_original_take_id not in t.reference_original_take_ids
        assert t.probe_view not in t.enroll_views and len(t.enroll_views) == 3
        assert all(
            r.startswith(t.claimed_subject_id) for r in t.reference_original_take_ids
        )
        assert t.dataset_fingerprint == reg.fingerprint
    # Sampling is seeded: another seed changes claims, not class proportions.
    other = generate_verification_trials(
        reg, split, trial_policy(k=2, seed=4), "development"
    )
    assert (
        other["counts"] == doc["counts"]
        and other["manifest_sha256"] != doc["manifest_sha256"]
    )
    # Asking for more impostors than exist records the shortfall explicitly.
    big = generate_verification_trials(reg, split, trial_policy(k=10), "development")
    assert big["counts"]["impostor"] == len(dev) * 4 * (len(dev) - 1)
    assert big["counts"]["impostor_shortfall"] == len(dev) * 4 * (10 - len(dev) + 1)


def test_derivative_query_uses_registered_lineage():
    rows = cohort_rows(6)
    children = [derived_row(r, "vidstab-s10") for r in rows]
    reg = make_registry(rows + children)
    split = locked_split(reg)
    doc = generate_verification_trials(
        reg, split, trial_policy(query_variant_id="vidstab-s10"), "development"
    )
    for t in load_trials(doc):
        assert t.query_asset_id.endswith(".vidstab-s10")
        assert t.query_variant_id == "vidstab-s10"
        assert t.query_original_take_id not in t.reference_original_take_ids
    partial = make_registry(rows + children[:4])
    doc = generate_verification_trials(
        partial,
        locked_split(partial),
        trial_policy(query_variant_id="vidstab-s10"),
        "development",
    )
    assert doc["counts"]["skipped_queries"]["NO_QUERY_VARIANT"] > 0


def _genuine(trials):
    return next(t for t in trials if t["label_same_identity"])


def _retamper(doc, mutate):
    doc = json.loads(json.dumps(doc))
    mutate(doc["trials"])
    doc["manifest_sha256"] = manifest_hash(load_trials(doc))
    return doc


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda ts: ts[0].update(
                label_same_identity=not ts[0]["label_same_identity"]
            ),
            "label",
        ),
        (lambda ts: ts.append(dict(ts[0])), "Duplicate trial"),
        (lambda ts: ts.append(dict(ts[0], trial_id="f" * 24)), "Duplicate query"),
        (
            lambda ts: _genuine(ts).update(
                reference_original_take_ids=_genuine(ts)["reference_original_take_ids"]
                + [_genuine(ts)["query_original_take_id"]],
                enroll_views=_genuine(ts)["enroll_views"]
                + [_genuine(ts)["probe_view"]],
            ),
            "share an original",
        ),  # fmt: skip
        (lambda ts: ts[0].update(reference_original_take_ids=[]), "reference"),
        (lambda ts: ts[0].update(trial_id="0" * 24), "Trial ID"),
        (lambda ts: ts[0].update(query_subject_id="S999"), "provenance"),
        (lambda ts: ts[0].update(split_role="calibration"), "split"),
        (
            lambda ts: ts[0].update(
                query_original_take_id=ts[0]["query_original_take_id"] + "-x"
            ),
            "provenance",
        ),
    ],
)
def test_trial_manifest_invariants_reject_tampering(mutate, match):
    reg = make_registry(cohort_rows(6))
    split = locked_split(reg)
    doc = generate_verification_trials(reg, split, trial_policy(), "development")
    with pytest.raises((ValueError, ProvenanceError), match=match):
        validate_trial_manifest(_retamper(doc, mutate), reg, split)


def test_trial_manifest_bound_to_registry_split_and_hash():
    reg = make_registry(cohort_rows(6))
    split = locked_split(reg)
    doc = generate_verification_trials(reg, split, trial_policy(), "development")
    with pytest.raises(ValueError, match="hash"):
        validate_trial_manifest(dict(doc, trials=doc["trials"][1:]), reg, split)
    grown = make_registry(
        cohort_rows(6) + [derived_row(cohort_rows(1)[0], "vidstab-s10")]
    )
    with pytest.raises(ValueError, match="registry"):
        validate_trial_manifest(doc, grown, split)
    with pytest.raises(ValueError, match="split"):
        validate_trial_manifest(doc, reg, locked_split(reg, seed=99))
    draft = assign_split(reg.subjects(), SplitPolicy("v", 1, split["policy"]["counts"]))
    with pytest.raises(ValueError, match="LOCKED"):
        generate_verification_trials(reg, draft, trial_policy(), "development")


# ============================================================ pure unit: metrics


def test_metric_known_values_and_score_orientation():
    labels = [1, 1, 1, 0, 0, 0]
    scores = [0.9, 0.8, 0.4, 0.7, 0.3, 0.2]
    m = verification_metrics(labels, scores, target_fars=(1 / 3,))
    assert m["roc_auc"] == pytest.approx(8 / 9) and m["eer"] == pytest.approx(1 / 3)
    assert m["tar_at_far"] == pytest.approx(2 / 3)
    assert m["tar_at_fars"][str(1 / 3)]["tar"] == 1.0
    flipped = verification_metrics(labels, [-s for s in scores])
    assert flipped["roc_auc"] == pytest.approx(
        1 / 9
    )  # lower similarity never supports a claim
    perfect = verification_metrics([1, 0], [0.9, 0.1])
    assert (perfect["roc_auc"], perfect["eer"], perfect["tar_at_far"]) == (
        1.0,
        0.0,
        1.0,
    )


def test_ties_interpolated_eer_and_threshold_boundary():
    m = verification_metrics([1, 1, 0, 0], [0.9, 0.5, 0.5, 0.1], threshold=0.5)
    assert m["roc_auc"] == pytest.approx(0.875) and m["eer"] == pytest.approx(0.25)
    # Equal score accepts: both 0.5 trials accepted.
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (2, 1, 1, 0)
    assert (m["far"], m["frr"]) == (0.5, 0.0)
    above = verification_metrics([1, 1, 0, 0], [0.9, 0.5, 0.5, 0.1], threshold=0.5001)
    assert (above["far"], above["frr"]) == (0.0, 0.5)
    m = verification_metrics([1, 1, 1, 0, 0], [0.9, 0.8, 0.3, 0.85, 0.2])
    assert m["eer"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "labels,scores,statuses",
    [
        ([1, 1], [0.5, 0.6], None),
        ([0, 0], [0.5, 0.6], None),
        ([1, 0], [None, None], ["INSUFFICIENT_GAIT", "CLAIM_NOT_ENROLLED"]),
        ([], [], []),
    ],
)
def test_undefined_metrics_carry_reasons_not_zero_or_nan(labels, scores, statuses):
    m = verification_metrics(labels, scores, statuses, threshold=0.5)
    assert m["roc_auc"] is None and m["eer"] is None and m["tar_at_far"] is None
    assert m["undefined"]["roc_auc"].startswith("UNDEFINED_METRIC")
    if not labels:
        assert m["coverage"] is None and m["score_coverage"] is None
    if 0 not in labels:
        assert m["far"] is None and "far" in m["undefined"]


@pytest.mark.parametrize(
    "scores,statuses",
    [
        ([float("nan"), 0.1], ["OK", "OK"]),
        ([float("inf"), 0.1], ["POLICY_NOT_READY", "POLICY_NOT_READY"]),
        ([0.9, 0.1], ["INSUFFICIENT_GAIT", "OK"]),
        ([None, 0.1], ["OK", "OK"]),
    ],
)
def test_invalid_scores_fail_loudly(scores, statuses):
    with pytest.raises(ValueError):
        verification_metrics([1, 0], scores, statuses)
    with pytest.raises(ValueError):
        verification_metrics([1, 0], [0.9, 0.1], target_far=1.5)
    with pytest.raises(ValueError):
        verification_metrics([1, 0], [0.9, 0.1], threshold=float("nan"))


def test_policy_not_ready_scores_count_as_scored_not_decided():
    m = verification_metrics(
        [1, 0, 1, 0],
        [0.9, 0.1, None, None],
        [
            "POLICY_NOT_READY",
            "POLICY_NOT_READY",
            "INSUFFICIENT_GAIT",
            "CLAIM_NOT_ENROLLED",
        ],
    )
    assert m["evaluable"] == 2 and m["score_coverage"] == 0.5
    assert m["coverage"] == 0.0 and m["operational_decisions"] == 0
    assert m["roc_auc"] == 1.0
    assert m["score_coverage_by_class"] == {"0": 0.5, "1": 0.5}
    small = verification_metrics([1, 0, 0], [0.9, 0.2, 0.1])
    assert any("BELOW_RESOLUTION" in w for w in small["warnings"])


# ============================================================ uncertainty


def test_subject_bootstrap_determinism_and_limitations():
    labels, scores, subjects = [], [], []
    for i, s in enumerate(["S1", "S2", "S3", "S4"]):
        labels += [1, 0]
        scores += [0.6 + i / 10, 0.4 - i / 20]
        subjects += [s, s]
    statuses = ["POLICY_NOT_READY"] * len(labels)
    a = subject_bootstrap(labels, scores, statuses, subjects, 200, seed=5)
    assert a == subject_bootstrap(labels, scores, statuses, subjects, 200, seed=5)
    assert a["status"] == "OK" and a["ci95"]["roc_auc"] == [1.0, 1.0]
    assert any("FEW_CLUSTERS" in w for w in a["warnings"])
    one = subject_bootstrap([1, 0], [0.9, 0.1], statuses[:2], ["S1", "S1"])
    assert one["status"] == "UNDEFINED_METRIC"
    # Only S1 has an impostor trial: most replicates are undefined -> no interval.
    lab, sco, sub = [1, 0], [0.9, 0.1], ["S1", "S1"]
    for s in ("S2", "S3", "S4", "S5"):
        lab, sco, sub = lab + [1], sco + [0.8], sub + [s]
    unstable = subject_bootstrap(lab, sco, ["OK"] * len(lab), sub, 300, seed=1)
    assert unstable["status"] == "UNSTABLE" and unstable["ci95"]["roc_auc"] is None
    assert "UNSTABLE" in unstable["limitations"]["roc_auc"]


def _rows(scores, statuses=None):
    rows = []
    for i, score in enumerate(scores):
        rows.append(
            dict(
                trial_id=f"t{i}",
                label=i % 2,
                query_subject_id=f"S{i // 2}",
                claimed_subject_id=f"S{i // 2}" if i % 2 else "SX",
                split_role="development",
                status=(statuses or ["POLICY_NOT_READY"] * len(scores))[i],
                score=score,
            )
        )
    return rows


def test_paired_comparison_on_identical_trials():
    a = _rows([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.6, 0.4])
    statuses = ["POLICY_NOT_READY"] * 8
    statuses[1] = "INSUFFICIENT_GAIT"
    b_scores = [0.5, None, 0.2, 0.8, 0.3, 0.7, 0.6, 0.4]
    b = _rows(b_scores, statuses)
    c = paired_comparison(a, b, repetitions=100, seed=1)
    assert c["trials"] == 8 and c["common_scoreable_trials"] == 7
    assert c["system_a"]["full_manifest"]["score_coverage"] == 1.0
    assert c["system_b"]["full_manifest"]["score_coverage"] == 7 / 8
    common = [r for r in a if r["trial_id"] != "t1"]
    expected = verification_metrics(
        [r["label"] for r in common], [r["score"] for r in common]
    )["roc_auc"]
    assert c["system_a"]["common_scoreable"]["roc_auc"] == pytest.approx(expected)
    assert c["difference_a_minus_b"]["roc_auc"] == pytest.approx(
        expected - c["system_b"]["common_scoreable"]["roc_auc"]
    )
    assert (
        "winner" not in json.dumps(c).lower() and "better" not in json.dumps(c).lower()
    )
    assert c == paired_comparison(a, b, repetitions=100, seed=1)
    with pytest.raises(ValueError, match="same trial IDs"):
        paired_comparison(a, b[:-1])
    with pytest.raises(ValueError, match="labels"):
        paired_comparison(a, [dict(r, label=1 - r["label"]) for r in b])
    with pytest.raises(LockedTestAccessDenied):
        locked = [dict(r, split_role="locked_final") for r in a]
        paired_comparison(locked, locked)


# ============================================================ synthetic offline evaluation


def _template(asset, model=RECIPE, role="development", vector=None):
    rec = Recording(
        asset.asset_id,
        asset.original_take_id,
        asset.subject_id,
        asset.session_id,
        asset.view,
        asset.recorded_at,
        asset.sha256,
        role,
    )
    emb = GaitEmbedding(vector, model, asset.original_take_id, asset.sha256)
    return EnrollmentTemplate(
        asset.subject_id, emb, rec, 1.0, f"enr-{asset.asset_id}-{model[:4]}", {}
    )


def _subject_vector(subject, rng, dim=8):
    base = np.random.default_rng(int(subject[1:])).normal(size=dim)
    return base + 0.3 * rng.normal(size=dim)


class SpyStore:
    def __init__(self, inner):
        self.inner, self.calls = inner, []

    def get_claim(self, claim):
        self.calls.append(claim)
        return self.inner.get_claim(claim)


@pytest.fixture
def synthetic_eval(tmp_path):
    reg = make_registry(cohort_rows(6))
    split = locked_split(reg)
    doc = generate_verification_trials(reg, split, trial_policy(k=2), "development")
    rng = np.random.default_rng(0)
    store = MemoryEnrollmentStore()
    for subject in subjects_for(split, ("development",)):
        store.enroll(
            subject,
            [
                _template(a, vector=_subject_vector(subject, rng))
                for a in reg.originals(subject)
            ],
        )
    queries = QueryEmbeddingStore(tmp_path / "queries")
    trials = load_trials(doc)
    abstain = trials[0].query_asset_id
    for asset_id in sorted({t.query_asset_id for t in trials}):
        a = reg.asset(asset_id)
        ok = asset_id != abstain
        values = _subject_vector(a.subject_id, rng)
        queries.put(
            {
                "schema_version": "query-embedding-v1",
                "asset_id": asset_id,
                "original_take_id": a.original_take_id,
                "variant_id": "raw",
                "source_sha256": a.sha256,
                "encoder_key": ENCODER_KEY,
                "model_id": RECIPE if ok else None,
                "status": "EMBEDDED" if ok else "INSUFFICIENT_GAIT",
                "reason": "OK" if ok else "QUALITY_GATE",
                "quality_reasons": [] if ok else ["ankles_missing"],
                "values": (values / np.linalg.norm(values)).tolist() if ok else None,
                "runtime_sec": 0.5,
                "peak_cuda_mib": None,
            }
        )
    system = {"label": "stat-a", "encoder_key": ENCODER_KEY, "training_dataset": "GREW"}
    return reg, split, doc, store, queries, system, abstain


def test_offline_run_preserves_every_trial_and_separates_coverage(
    synthetic_eval, tmp_path
):
    reg, split, doc, store, queries, system, abstain = synthetic_eval
    spy = SpyStore(store)
    seen = []
    scorer = EmbeddingTrialScorer(reg, split, spy, queries, ENCODER_KEY)

    def watching(request):
        seen.append(request)
        return scorer(request)

    watching.recipes, watching.runtime = scorer.recipes, scorer.runtime
    report, outcomes = run_evaluation(
        doc, reg, split, watching, system, "SYNTHETIC_FIXTURE", bootstrap_repetitions=50
    )
    assert len(outcomes) == doc["counts"]["trials"]
    assert [o["trial_id"] for o in outcomes] == [t["trial_id"] for t in doc["trials"]]
    # Scorer input carries no walker identity or label; lookups are claims only.
    assert set(ClaimRequest.__dataclass_fields__) == {
        "trial_id",
        "query_asset_id",
        "query_original_take_id",
        "query_variant_id",
        "claimed_subject_id",
        "reference_original_take_ids",
    }
    assert sorted(spy.calls) == sorted({r.claimed_subject_id for r in seen})
    by_status = {}
    for o in outcomes:
        by_status.setdefault(o["status"], []).append(o)
    abstained = [o for o in outcomes if o["query_asset_id"] == abstain]
    assert all(o["status"] == "INSUFFICIENT_GAIT" for o in abstained)
    assert all(
        not o["score_available"] and not o["quality_evaluable"] for o in abstained
    )
    scored = by_status["POLICY_NOT_READY"]
    assert len(scored) == len(outcomes) - len(abstained)
    assert all(o["score_available"] and o["quality_evaluable"] for o in scored)
    assert not any(
        o["operational_decision_available"] or o["verdict"] for o in outcomes
    )
    assert report["coverage"]["operational"] == 0.0
    assert report["coverage"]["score"] == pytest.approx(len(scored) / len(outcomes))
    assert report["metrics"]["overall"]["roc_auc"] is not None
    assert set(report["metrics"]["by_probe_view"]) == set(VIEWS)
    assert (
        report["status_reason_counts"]["INSUFFICIENT_GAIT/QUALITY_GATE"]["genuine"] == 1
    )
    assert report["system"]["recipe_model_ids"] == [RECIPE]
    assert any("NOT biometric performance" in w for w in report["warnings"])
    assert any("cross-session" in w for w in report["warnings"])
    # Restricted artifacts: private files, no embeddings, consent refs or paths.
    written = write_run(tmp_path / "run", report, outcomes)
    text = (tmp_path / "run/report.json").read_text()
    for forbidden in ('"values"', "CONSENT-", "raw/", "source_file_uri"):
        assert forbidden not in text
    assert (tmp_path / "run/outcomes.jsonl").stat().st_mode & 0o777 == 0o600
    assert read_run(tmp_path / "run")[0] == written
    with pytest.raises(FileExistsError):
        write_run(tmp_path / "run", report, outcomes)


def test_missing_incomplete_and_mismatched_enrollment_statuses(synthetic_eval):
    reg, split, doc, store, queries, system, _ = synthetic_eval
    trials = load_trials(doc)
    t = next(t for t in trials if t.label_same_identity)
    req = ClaimRequest(
        t.trial_id,
        t.query_asset_id,
        t.query_original_take_id,
        t.query_variant_id,
        t.claimed_subject_id,
        t.reference_original_take_ids,
    )
    scorer = EmbeddingTrialScorer(
        reg, split, MemoryEnrollmentStore(), queries, ENCODER_KEY
    )
    assert scorer(req)[:2] == (Status.CLAIM_NOT_ENROLLED, "NO_REFERENCES")
    partial = MemoryEnrollmentStore()
    full = store.get_claim(t.claimed_subject_id)
    partial.enroll(t.claimed_subject_id, full[:1])
    s = EmbeddingTrialScorer(reg, split, partial, queries, ENCODER_KEY)
    assert s(req)[:2] == (Status.CLAIM_NOT_ENROLLED, "ENROLLMENT_INCOMPLETE")
    other = MemoryEnrollmentStore()
    other.enroll(
        t.claimed_subject_id,
        [
            replace(x, embedding=replace(x.embedding, model_id="other:recipe"))
            for x in full
        ],
    )
    s = EmbeddingTrialScorer(reg, split, other, queries, ENCODER_KEY)
    assert s(req)[:2] == (Status.CLAIM_NOT_ENROLLED, "RECIPE_MISMATCH")
    wrong_role = MemoryEnrollmentStore()
    wrong_role.enroll(
        t.claimed_subject_id,
        [
            replace(x, recording=replace(x.recording, split_role="calibration"))
            for x in full
        ],
    )
    s = EmbeddingTrialScorer(reg, split, wrong_role, queries, ENCODER_KEY)
    assert s(req)[1] == "ENROLLMENT_PROVENANCE_MISMATCH"
    empty = QueryEmbeddingStore(queries.root.parent / "none")
    s = EmbeddingTrialScorer(reg, split, store, empty, ENCODER_KEY)
    assert s(req)[:2] == (Status.PROCESSING_ERROR, "QUERY_EMBEDDING_MISSING")


def test_no_operational_policy_unless_isolated_test_only(synthetic_eval):
    reg, split, doc, store, queries, system, _ = synthetic_eval
    calibrated_looking = VerificationPolicy(
        RECIPE, 0.5, "v", calibration_manifest_sha256="x"
    )
    for policy in (VerificationPolicy(RECIPE), calibrated_looking):
        with pytest.raises(ValueError, match="authorized"):
            EmbeddingTrialScorer(reg, split, store, queries, ENCODER_KEY, policy)
    # Isolated synthetic demonstration threshold: never a deployable artifact.
    demo = VerificationPolicy(RECIPE, 0.5, "TEST_ONLY", test_only=True)
    scorer = EmbeddingTrialScorer(
        reg, split, store, queries, ENCODER_KEY, demo, allow_test_policy=True
    )
    report, outcomes = run_evaluation(
        doc, reg, split, scorer, system, "SYNTHETIC_FIXTURE", bootstrap_repetitions=20
    )
    decided = [o for o in outcomes if o["operational_decision_available"]]
    assert decided and all(isinstance(o["verdict"], bool) for o in decided)
    assert report["coverage"]["operational"] == report["coverage"]["score"]
    config = json.loads(
        (REPO / "configs/gait_verification/phase1d_evaluation.json").read_text()
    )
    assert config["threshold"] is None and config["calibration"] is None
    for script in ("evaluate_claims.py", "verify_claim.py", "enroll_subject.py"):
        text = (REPO / "scripts/gait_verification" / script).read_text()
        for forbidden in (
            "allow_test_policy",
            "VerificationPolicy",
            "test_only=",
            "LockedTestAuthorization",
        ):
            assert forbidden not in text


def test_runs_compare_only_on_identical_manifests_and_matched_pretraining(
    synthetic_eval,
):
    reg, split, doc, store, queries, system, _ = synthetic_eval
    scorer = EmbeddingTrialScorer(reg, split, store, queries, ENCODER_KEY)
    run_a = run_evaluation(
        doc, reg, split, scorer, system, "SYNTHETIC_FIXTURE", bootstrap_repetitions=20
    )

    def e0_like(request):
        # Stand-in for the future E0 pairwise adapter: same requests, own scores.
        h = int(hashlib.sha256(request.trial_id.encode()).hexdigest(), 16)
        return (
            Status.POLICY_NOT_READY,
            "NO_AUTHORIZED_POLICY",
            (h % 1000) / 1000,
            None,
            True,
        )

    e0 = dict(system, label="E0-standin", encoder_key="e0")
    run_b = run_evaluation(
        doc, reg, split, e0_like, e0, "SYNTHETIC_FIXTURE", bootstrap_repetitions=20
    )
    assert [o["trial_id"] for o in run_a[1]] == [o["trial_id"] for o in run_b[1]]
    comparison = compare_runs(run_a, run_b, repetitions=20)
    assert comparison["trials"] == doc["counts"]["trials"]
    with pytest.raises(ValueError, match="same system|different systems"):
        compare_runs(run_a, run_a)
    casia = (run_b[0] | {"system": dict(e0, training_dataset="CASIA-B")}, run_b[1])
    with pytest.raises(ValueError, match="CASIA-B"):
        compare_runs(run_a, casia)
    other = generate_verification_trials(
        reg, split, trial_policy(seed=9), "development"
    )
    run_c = run_evaluation(
        other, reg, split, e0_like, e0, "SYNTHETIC_FIXTURE", bootstrap_repetitions=20
    )
    with pytest.raises(ValueError, match="different"):
        compare_runs(run_a, run_c)


def test_v2_package_never_imports_historical_e0_code():
    for path in (REPO / "gait_verification").rglob("*.py"):
        text = path.read_text()
        for forbidden in ("from models", "import models", "from utils", "import utils"):
            assert forbidden not in text, path


def test_public_response_unchanged():
    from gait_verification.contracts import VerificationResult

    r = VerificationResult("S001", Status.POLICY_NOT_READY)
    assert set(r.public_dict()) == {
        "schema_version",
        "claimed_subject_id",
        "status",
        "verdict",
    }


# ============================================================ enrollment store atomicity


def test_multi_recipe_add_is_atomic_and_retry_safe(tmp_path, monkeypatch):
    reg = make_registry(cohort_rows(1))
    asset = reg.originals("S001")[0]
    store = DirectoryEnrollmentStore(tmp_path)
    vec = np.ones(4)
    group = [_template(asset, m, vector=vec) for m in ("a:recipe", "b:recipe")]
    real = store._atomic_write

    def crash(path, payload):
        if path.name != "source-index.json":
            raise OSError("power loss")
        real(path, payload)

    monkeypatch.setattr(store, "_atomic_write", crash)
    with pytest.raises(OSError):
        store.add_references("S001", group)
    monkeypatch.undo()
    assert store.get_claim("S001") == ()  # no partial recipe group
    store.add_references("S001", group)  # orphan index entry is harmless on retry
    assert {t.embedding.model_id for t in store.get_claim("S001")} == {
        "a:recipe",
        "b:recipe",
    }


# ============================================================ CLI workflow (synthetic)


def test_cli_registry_split_trials_run_and_compare(tmp_path):
    rows = cohort_rows(6)
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({"schema_version": SCHEMA, "assets": rows}))
    cli = [sys.executable, str(REPO / "scripts/gait_verification/evaluate_claims.py")]

    def run(*args, ok=True):
        p = subprocess.run(cli + [str(a) for a in args], capture_output=True, text=True)
        assert (p.returncode == 0) == ok, p.stdout + p.stderr
        return p.stdout

    run("validate", "--registry", registry_path, "--out", tmp_path / "validation.json")
    split_args = ["--registry", registry_path, "--policy-version", "v1", "--seed", 7]
    run("split", *split_args, "--development", 4, "--calibration", 1, "--locked-final", 1,
        "--out", tmp_path / "split-draft.json")  # fmt: skip
    run(
        "split", *split_args, "--development", 4, "--out", tmp_path / "x.json", ok=False
    )
    run("lock", "--split", tmp_path / "split-draft.json", "--locked-by", "tester",
        "--out", tmp_path / "split.json")  # fmt: skip
    trial_args = ["--registry", registry_path, "--split", tmp_path / "split.json"]
    run("trials", *trial_args, "--role", "development", "--policy-version", "t1",
        "--seed", 3, "--impostors-per-query", 2, "--out", tmp_path / "trials.json")  # fmt: skip
    run("trials", *trial_args, "--role", "locked_final", "--policy-version", "t1",
        "--seed", 3, "--impostors-per-query", 2, "--out", tmp_path / "t2.json", ok=False)  # fmt: skip
    doc = json.loads((tmp_path / "trials.json").read_text())
    reg = make_registry(rows)
    models = json.loads((REPO / "configs/gait_verification/models.json").read_text())
    key = models["gaitpt_grew"]["model_id"]
    recipe = key + ":pipeline=p:windows=w"
    enroll = DirectoryEnrollmentStore(tmp_path / "enroll")
    queries = QueryEmbeddingStore(tmp_path / "queries")
    rng = np.random.default_rng(1)
    split = json.loads((tmp_path / "split.json").read_text())
    for subject in subjects_for(split, ("development",)):
        enroll.enroll(
            subject,
            [
                _template(a, recipe, vector=_subject_vector(subject, rng))
                for a in reg.originals(subject)
            ],
        )
    for asset_id in sorted({t["query_asset_id"] for t in doc["trials"]}):
        a = reg.asset(asset_id)
        v = _subject_vector(a.subject_id, rng)
        queries.put(
            dict(
                schema_version="query-embedding-v1",
                asset_id=asset_id,
                original_take_id=a.original_take_id,
                variant_id="raw",
                source_sha256=a.sha256,
                encoder_key=key,
                model_id=recipe,
                status="EMBEDDED",
                reason="OK",
                quality_reasons=[],
                values=(v / np.linalg.norm(v)).tolist(),
            )
        )
    common = [*trial_args, "--trials", tmp_path / "trials.json",
              "--enrollments", tmp_path / "enroll", "--query-store", tmp_path / "queries",
              "--evidence-class", "SYNTHETIC_FIXTURE", "--bootstrap", 20]  # fmt: skip
    out = run("run", *common, "--model", "gaitpt", "--out", tmp_path / "run-a")
    assert "POLICY_NOT_READY" in out
    report = json.loads((tmp_path / "run-a/report.json").read_text())
    assert report["counts"]["score_available"] == doc["counts"]["trials"]
    assert report["counts"]["operational_decisions"] == 0
    # GPGait has no stored queries here: every trial preserved as a missing input.
    run("run", *common, "--model", "gpgait", "--out", tmp_path / "run-b")
    b = json.loads((tmp_path / "run-b/report.json").read_text())
    assert b["coverage"]["score"] == 0.0
    run("compare", "--run", tmp_path / "run-a", "--run", tmp_path / "run-b",
        "--bootstrap", 20, "--out", tmp_path / "cmp.json")  # fmt: skip
    run("run", *common, "--model", "gpgaitpp_casia", "--out", tmp_path / "run-c")
    run("compare", "--run", tmp_path / "run-a", "--run", tmp_path / "run-c",
        "--out", tmp_path / "cmp2.json", ok=False)  # fmt: skip


# ============================================================ media integration (ffmpeg)

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg unavailable"
)


def _media_registry(tmp_path, videos, extra=()):
    from gait_verification.media import probe_video

    rows = []
    for (subject, view), path in videos.items():
        meta = probe_video(path)
        rows.append(
            raw_row(
                subject,
                view,
                sha256=meta.sha256,
                source_file_uri=str(path.relative_to(tmp_path)),
                duration_sec=meta.duration_sec,
            )
        )
    return rows + list(extra)


@needs_ffmpeg
def test_managed_query_resolves_undeclared_derivative_and_fails_closed(tmp_path):
    from test_phase1c import FakeDetector, StatEncoder, make_video
    from gait_verification.pipeline import (
        CachedPoseExtractor,
        enroll_registered,
        query_verifier,
        verify_registered,
    )
    from gait_verification.pose_quality import WindowQualityPolicy, WindowedEncoder
    from gait_verification.stabilization import StabilizationConfig, stabilize

    filters = subprocess.run(
        ["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True
    ).stdout
    if "vidstabdetect" not in filters:
        pytest.skip("libvidstab unavailable")
    (tmp_path / "raw").mkdir()
    videos = {
        ("S001", "left45"): make_video(tmp_path / "raw/a.mp4", 3.0),
        ("S001", "right45"): make_video(tmp_path / "raw/b.mp4", 3.2),
    }
    rows = _media_registry(tmp_path, videos)
    reg0 = make_registry(rows)
    original = reg0.originals("S001")[0]
    work = tmp_path / "derived/a-s10"
    manifest = stabilize(
        tmp_path / original.source_uri,
        work,
        reg0.recording(original.asset_id, "development"),
        StabilizationConfig(enabled=True),
    )
    child = derived_row(
        next(r for r in rows if r["asset_id"] == original.asset_id),
        "vidstab-s10",
        sha256=manifest["output"]["sha256"],
        source_file_uri="derived/a-s10/stabilized.mp4",
        transformation_manifest_ref="derived/a-s10/manifest.json",
    )
    reg = make_registry(rows + [child])
    assert reg.verify_files(tmp_path, measure=True)["ok"]
    split = locked_split(reg, {"development": 1, "calibration": 0, "locked_final": 0})
    policy = WindowQualityPolicy()
    encoder = WindowedEncoder(StatEncoder(), policy)
    store = DirectoryEnrollmentStore(tmp_path / "enroll")
    calls = []

    def extractor_factory(take, variant):
        calls.append((take, variant))
        return CachedPoseExtractor(FakeDetector(), None, take, variant)

    first = enroll_registered(
        reg, split, [original.asset_id, child["asset_id"], "nope"], tmp_path,
        extractor_factory, [encoder], store, policy,
    )  # fmt: skip
    status = {r["asset_id"]: r["status"] for r in first}
    assert status == {
        original.asset_id: "ENROLLED",
        child["asset_id"]: "REFUSED",
        "nope": "REFUSED",
    }
    again = enroll_registered(
        reg,
        split,
        [original.asset_id],
        tmp_path,
        extractor_factory,
        [encoder],
        store,
        policy,
    )
    assert again[0]["status"] == "ALREADY_ENROLLED"
    ref = store.get_claim("S001")[0]
    assert (ref.recording.session_id, ref.provenance["consent_ref"]) == (
        "S001_sess01",
        "CONSENT-S001",
    )
    assert all(isinstance(a, str) and "S001" not in str(b) for a, b in calls)

    def make_verifier(take, variant):
        calls.append((take, variant))
        return query_verifier(store, encoder, extractor_factory(take, variant), policy)

    # Stabilized derivative of the enrolled original, submitted WITHOUT any declared
    # lineage flags: the registry resolves it and the comparison is refused.
    derivative = tmp_path / child["source_file_uri"]
    result, diag = verify_registered(reg, derivative, "S001", make_verifier)
    assert result.status == Status.INVALID_INPUT and diag["reason"] == "SHARED_ORIGINAL"
    # Independent registered original: scored, no threshold -> POLICY_NOT_READY.
    other = tmp_path / reg.originals("S001")[1].source_uri
    result, diag = verify_registered(reg, other, "S001", make_verifier)
    assert result.status == Status.POLICY_NOT_READY and -1 <= diag["score"] <= 1
    # Unregistered re-encode of the enrolled original: cannot be linked by bytes,
    # so the managed workflow fails closed instead of treating it as a new original.
    copy = tmp_path / "copy.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(derivative), "-c:v", "libx264",
                    "-crf", "30", str(copy)], check=True)  # fmt: skip
    result, diag = verify_registered(reg, copy, "S001", make_verifier)
    assert (
        result.status == Status.INVALID_INPUT
        and diag["reason"] == "UNREGISTERED_PROVENANCE"
    )
    assert result.verdict is None and set(result.public_dict()) == {
        "schema_version", "claimed_subject_id", "status", "verdict"
    }  # fmt: skip
    missing = verify_registered(reg, tmp_path / "absent.mp4", "S001", make_verifier)[1]
    assert missing["reason"] == "VIDEO_NOT_FOUND"
    # The derivative bytes registered as an "original" are exposed by verification.
    disguised = raw_row("S001", "toward", sha256=child["sha256"],
                        source_file_uri=child["source_file_uri"])  # fmt: skip
    report = make_registry(rows + [dict(child, sha256=sha("x"),
                           transformation_manifest_ref=child["transformation_manifest_ref"]),
                           disguised]).verify_files(tmp_path)  # fmt: skip
    assert not report["ok"]


@needs_ffmpeg
def test_batch_enrollment_reports_partial_failures(tmp_path):
    from test_phase1c import FakeDetector, StatEncoder, make_video
    from gait_verification.contracts import ModelMetadata
    from gait_verification.pipeline import CachedPoseExtractor, enroll_registered
    from gait_verification.pose_quality import WindowQualityPolicy, WindowedEncoder
    from gait_verification.offline_eval import embed_registered_queries
    from gait_verification.pose_quality import assess_window_quality

    (tmp_path / "raw").mkdir()
    videos = {
        ("S001", "left45"): make_video(tmp_path / "raw/a.mp4", 3.0),
        ("S001", "right45"): make_video(tmp_path / "raw/b.mp4", 3.1),
        ("S002", "left45"): make_video(tmp_path / "raw/c.mp4", 3.2),
        ("S003", "left45"): make_video(tmp_path / "raw/d.mp4", 3.3),
    }
    rows = _media_registry(tmp_path, videos)
    reg = make_registry(rows)
    split = locked_split(
        reg,
        {"development": 2, "calibration": 0, "locked_final": 1},
        pinned=("S001", "S002"),
    )
    policy = WindowQualityPolicy()

    class Second(StatEncoder):
        metadata = ModelMetadata(
            "stat2:recipe", "none", "none", "n/a", 0, "synthetic", "stat", "none"
        )

        def encode(self, s):
            e = super().encode(s)
            return replace(e, model_id="stat2:recipe")

    encoders = [
        WindowedEncoder(StatEncoder(), policy),
        WindowedEncoder(Second(), policy),
    ]
    store = DirectoryEnrollmentStore(tmp_path / "enroll")

    def factory(take, variant):
        return CachedPoseExtractor(
            FakeDetector(people=2 if take.startswith("S002") else 1),
            None,
            take,
            variant,
        )

    (tmp_path / rows[1]["source_file_uri"]).write_bytes(b"tampered")  # S001 right45
    results = enroll_registered(
        reg,
        split,
        [r["asset_id"] for r in rows],
        tmp_path,
        factory,
        encoders,
        store,
        policy,
    )
    by = {r["asset_id"]: r for r in results}
    assert by[rows[0]["asset_id"]]["status"] == "ENROLLED"
    assert len(by[rows[0]["asset_id"]]["model_ids"]) == 2  # both recipes, one write
    assert by[rows[1]["asset_id"]]["reason"] == "source_missing_or_checksum_mismatch"
    assert by[rows[2]["asset_id"]]["status"] == "REFUSED"  # ambiguous walker
    assert by[rows[3]["asset_id"]]["reason"] == "subject_role_not_selected"  # locked
    assert len(results) == 4  # nothing silently dropped
    # Query embedding ingestion: explicit assets, abstentions persisted, idempotent.
    qstore = QueryEmbeddingStore(tmp_path / "queries")
    first = embed_registered_queries(
        reg, [rows[0]["asset_id"], rows[2]["asset_id"], rows[1]["asset_id"]], tmp_path,
        factory, encoders[0], lambda s: assess_window_quality(s, policy), qstore,
    )  # fmt: skip
    status = {r["asset_id"]: r["status"] for r in first}
    assert status[rows[0]["asset_id"]] == "EMBEDDED"
    assert status[rows[2]["asset_id"]] == "INSUFFICIENT_GAIT"
    assert status[rows[1]["asset_id"]] == "REFUSED"
    record = qstore.get(rows[0]["asset_id"], "stat:recipe")
    assert "subject_id" not in record and record["model_id"].endswith(
        record["model_id"].split(":")[-1]
    )
    again = embed_registered_queries(
        reg, [rows[0]["asset_id"]], tmp_path, factory, encoders[0],
        lambda s: assess_window_quality(s, policy), qstore,
    )  # fmt: skip
    assert again[0]["status"] == "EXISTS"
