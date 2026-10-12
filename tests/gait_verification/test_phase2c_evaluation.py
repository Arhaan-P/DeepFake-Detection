"""Independent synthetic regressions for protected evaluation boundaries.

Fixtures contain invented identities, bytes and vectors. No real locked-final
recording, CUDA model, trainer or calibrator is opened by these tests.
"""

import copy
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest
import test_phase1d as fixtures

from gait_verification.capture_registry import canonical_sha256
from gait_verification.contracts import Status
from gait_verification.evaluation import (
    _trial_id,
    generate_verification_trials,
    load_trials,
    manifest_hash,
    validate_trial_manifest,
)
from gait_verification.offline_eval import (
    ClaimRequest,
    EmbeddingTrialScorer,
    QueryEmbeddingStore,
    compare_runs,
    embed_registered_queries,
    run_evaluation,
    seal,
)
from gait_verification.splits import check_transition

synthetic_eval = fixtures.synthetic_eval


def _byte_registry(tmp_path, declared_role=None):
    row = fixtures.raw_row("S001", "left45", split_role=declared_role)
    payload = b"INVENTED MEDIA BYTES FOR ACCESS-GUARD TEST"
    row["sha256"] = hashlib.sha256(payload).hexdigest()
    path = tmp_path / row["source_file_uri"]
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    return fixtures.make_registry([row]), row, path


def _unused(*args, **kwargs):
    pytest.fail("Protected or cached input unexpectedly reached a model loader")


def test_embed_cli_refuses_forged_trial_header_before_any_model_load(
    tmp_path, monkeypatch
):
    from scripts.gait_verification import evaluate_claims as cli
    from gait_verification import registry as loaders

    registry, row, _ = _byte_registry(tmp_path, "locked_final")
    registry_path, trials_path = tmp_path / "registry.json", tmp_path / "trials.json"
    registry_path.write_text(
        json.dumps({"schema_version": "capture-registry-v1", "assets": [row]})
    )
    # This is not a valid hashed manifest: a development header masks a final asset.
    trials_path.write_text(
        json.dumps(
            {
                "dataset_fingerprint": registry.fingerprint,
                "split_role": "development",
                "trials": [{"query_asset_id": row["asset_id"]}],
            }
        )
    )
    monkeypatch.setattr(loaders, "load_quality_policy", _unused)
    monkeypatch.setattr(loaders, "load_rtmpose", _unused)
    monkeypatch.setattr(loaders, "load_encoder", _unused)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_claims.py",
            "embed",
            "--registry",
            str(registry_path),
            "--trials",
            str(trials_path),
            "--data-root",
            str(tmp_path),
            "--model",
            "gaitpt",
            "--query-store",
            str(tmp_path / "queries"),
            "--out",
            str(tmp_path / "report.json"),
        ],
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code != 0
    assert not (tmp_path / "queries").exists()
    assert not (tmp_path / "report.json").exists()


def test_embed_helper_refuses_declared_final_before_source_resolution(
    tmp_path, monkeypatch
):
    registry, row, _ = _byte_registry(tmp_path, "locked_final")
    monkeypatch.setattr(registry, "path_of", _unused)
    encoder = SimpleNamespace(metadata=SimpleNamespace(model_id="synthetic:recipe"))
    result = embed_registered_queries(
        registry,
        [row["asset_id"]],
        tmp_path,
        _unused,
        encoder,
        _unused,
        QueryEmbeddingStore(tmp_path / "queries"),
    )
    assert result[0]["status"] == "REFUSED"
    assert not (tmp_path / "queries").exists()


def test_embed_cache_hit_rechecks_source_bytes_without_loading_models(tmp_path):
    registry, row, path = _byte_registry(tmp_path)
    encoder = SimpleNamespace(metadata=SimpleNamespace(model_id="synthetic:recipe"))
    store = QueryEmbeddingStore(tmp_path / "queries")
    store.put(
        {
            "schema_version": "query-embedding-v1",
            "asset_id": row["asset_id"],
            "encoder_key": encoder.metadata.model_id,
            "original_take_id": row["original_take_id"],
            "source_sha256": row["sha256"],
            "variant_id": "raw",
            "status": "INSUFFICIENT_GAIT",
            "reason": "SYNTHETIC_ABSTENTION",
            "model_id": None,
            "values": None,
        }
    )
    path.write_bytes(b"CHANGED SYNTHETIC SOURCE")
    result = embed_registered_queries(
        registry, [row["asset_id"]], tmp_path, _unused, encoder, _unused, store
    )
    assert result[0]["status"] == "REFUSED"
    assert result[0]["reason"] == "SOURCE_MISSING_OR_CHANGED"


@pytest.mark.parametrize("damage", ["final_move", "unlock"])
def test_split_transition_cannot_revise_final_membership_or_unlock(damage):
    registry = fixtures.make_registry(fixtures.cohort_rows(6))
    old = fixtures.locked_split(registry)
    new = copy.deepcopy(old)
    if damage == "unlock":
        new["status"] = "DRAFT"
    else:
        subject = next(
            s for s, role in old["assignments"].items() if role == "locked_final"
        )
        new["assignments"][subject] = "development"
        new["revisions"].append(
            {
                "revision_id": "synthetic-invalid-move",
                "previous_fingerprint": old["fingerprint"],
                "moves": {subject: "development"},
                "rationale": "Synthetic adversarial revision",
                "authorized_by": "synthetic-tester",
            }
        )
        new["fingerprint"] = canonical_sha256(
            {
                key: new[key]
                for key in (
                    "schema_version",
                    "policy",
                    "subject_set_sha256",
                    "assignments",
                    "revisions",
                )
            }
        )
    with pytest.raises(ValueError):
        check_transition(old, new)


@pytest.mark.parametrize("field", ["seed", "min_reference_originals", "reference_rule"])
def test_trial_validator_binds_declared_generation_policy(field):
    registry = fixtures.make_registry(fixtures.cohort_rows(6))
    split = fixtures.locked_split(registry)
    doc = generate_verification_trials(
        registry, split, fixtures.trial_policy(), "development"
    )
    doc["trial_policy"][field] = (
        "different-undocumented-rule" if field == "reference_rule" else 10000
    )
    with pytest.raises(ValueError):
        validate_trial_manifest(doc, registry, split)


def test_trial_validator_rejects_same_view_reference_even_when_rows_rehashed():
    registry = fixtures.make_registry(fixtures.cohort_rows(6))
    split = fixtures.locked_split(registry)
    doc = generate_verification_trials(
        registry, split, fixtures.trial_policy(), "development"
    )
    row = next(r for r in doc["trials"] if not r["label_same_identity"])
    reference = next(
        a
        for a in registry.originals(row["claimed_subject_id"])
        if a.view == row["probe_view"]
    )
    row["reference_original_take_ids"] = [reference.original_take_id]
    row["enroll_views"] = [reference.view]
    row["trial_id"] = _trial_id(
        row["trial_policy_version"],
        row["query_asset_id"],
        row["claimed_subject_id"],
        row["reference_original_take_ids"],
    )
    doc["manifest_sha256"] = manifest_hash(load_trials(doc))
    with pytest.raises(ValueError):
        validate_trial_manifest(doc, registry, split)


def test_embedding_scorer_refuses_wrong_cached_variant(synthetic_eval):
    registry, split, doc, enrollments, queries, _, abstain = synthetic_eval
    trial = next(
        t
        for t in load_trials(doc)
        if t.query_asset_id != abstain and t.label_same_identity
    )
    request = ClaimRequest(
        trial.trial_id,
        trial.query_asset_id,
        trial.query_original_take_id,
        trial.query_variant_id,
        trial.claimed_subject_id,
        trial.reference_original_take_ids,
    )
    path = queries._path(request.query_asset_id, fixtures.ENCODER_KEY)
    record = json.loads(path.read_text())
    record["variant_id"] = "vidstab-s10"
    path.write_text(json.dumps(seal(record)))
    result = EmbeddingTrialScorer(
        registry, split, enrollments, queries, fixtures.ENCODER_KEY
    )(request)
    assert result[0] in (Status.INVALID_INPUT, Status.PROCESSING_ERROR)
    assert result[2:] == (None, None, False)


@pytest.mark.parametrize(
    "status,verdict,quality",
    [
        (Status.POLICY_NOT_READY, True, True),
        (Status.OK, None, True),
        (Status.POLICY_NOT_READY, None, False),
    ],
)
def test_runner_rejects_inconsistent_scored_outcomes(
    synthetic_eval, status, verdict, quality
):
    registry, split, doc, _, _, system, _ = synthetic_eval

    def inconsistent_scorer(request):
        return status, "SYNTHETIC", 0.5, verdict, quality

    with pytest.raises(ValueError):
        run_evaluation(
            doc,
            registry,
            split,
            inconsistent_scorer,
            system,
            "SYNTHETIC_FIXTURE",
            target_fars=(0.05,),
            bootstrap_repetitions=2,
        )


def test_runner_refuses_study_evidence_for_wrong_split_role(synthetic_eval):
    registry, split, _, _, _, system, _ = synthetic_eval
    calibration_doc = generate_verification_trials(
        registry, split, fixtures.trial_policy(), "calibration"
    )

    def synthetic_scorer(request):
        return Status.POLICY_NOT_READY, "SYNTHETIC", 0.5, None, True

    with pytest.raises(ValueError):
        run_evaluation(
            calibration_doc,
            registry,
            split,
            synthetic_scorer,
            system,
            "STUDY_DEVELOPMENT",
            target_fars=(0.05,),
            bootstrap_repetitions=2,
        )


def test_matched_comparison_refuses_mixed_evidence_classes(synthetic_eval):
    registry, split, doc, _, _, system, _ = synthetic_eval

    def synthetic_scorer(request):
        return Status.POLICY_NOT_READY, "SYNTHETIC", 0.5, None, True

    run_a = run_evaluation(
        doc,
        registry,
        split,
        synthetic_scorer,
        system,
        "SYNTHETIC_FIXTURE",
        target_fars=(0.05,),
        bootstrap_repetitions=2,
    )
    run_b = copy.deepcopy(run_a)
    run_b[0]["system"]["label"] = "synthetic-other"
    run_b[0]["evidence_class"] = "STUDY_DEVELOPMENT"
    with pytest.raises(ValueError):
        compare_runs(run_a, run_b, target_far=0.05, repetitions=2)


@pytest.mark.parametrize("field", ["runtime_sec", "peak_cuda_mib"])
def test_malformed_query_runtime_remains_explicit_trial_error(synthetic_eval, field):
    registry, split, doc, enrollments, queries, system, abstain = synthetic_eval
    asset_id = next(
        r["query_asset_id"] for r in doc["trials"] if r["query_asset_id"] != abstain
    )
    path = queries._path(asset_id, fixtures.ENCODER_KEY)
    record = json.loads(path.read_text())
    record[field] = "corrupt string"
    path.write_text(json.dumps(seal(record)))
    report, outcomes = run_evaluation(
        doc,
        registry,
        split,
        EmbeddingTrialScorer(
            registry, split, enrollments, queries, fixtures.ENCODER_KEY
        ),
        system,
        "SYNTHETIC_FIXTURE",
        target_fars=(0.05,),
        bootstrap_repetitions=2,
    )
    assert report["counts"]["submitted_trials"] == len(doc["trials"])
    damaged = [row for row in outcomes if row["query_asset_id"] == asset_id]
    assert damaged and all(row["status"] == "PROCESSING_ERROR" for row in damaged)
    assert all(row["score"] is None and row["verdict"] is None for row in damaged)


def test_embedding_scorer_refuses_direct_final_request_before_artifact_read():
    registry = fixtures.make_registry(fixtures.cohort_rows(6))
    split = fixtures.locked_split(registry)
    final_subject = next(
        s for s, role in split["assignments"].items() if role == "locked_final"
    )
    query, reference = registry.originals(final_subject)[:2]
    query_store = SimpleNamespace(get=_unused)
    enrollment_store = SimpleNamespace(get_claim=_unused)
    request = ClaimRequest(
        "synthetic-final-request",
        query.asset_id,
        query.original_take_id,
        query.variant_id,
        final_subject,
        (reference.original_take_id,),
    )
    result = EmbeddingTrialScorer(
        registry, split, enrollment_store, query_store, fixtures.ENCODER_KEY
    )(request)
    assert result[0] in (Status.INVALID_INPUT, Status.PROCESSING_ERROR)
    assert result[2:] == (None, None, False)


def test_embedding_scorer_refuses_other_subject_reference_before_artifact_read():
    registry = fixtures.make_registry(fixtures.cohort_rows(6))
    split = fixtures.locked_split(registry)
    claim, other = [
        s for s, role in split["assignments"].items() if role == "development"
    ][:2]
    query = registry.originals(claim)[0]
    reference = registry.originals(other)[0]
    request = ClaimRequest(
        "synthetic-wrong-owner-request",
        query.asset_id,
        query.original_take_id,
        query.variant_id,
        claim,
        (reference.original_take_id,),
    )
    result = EmbeddingTrialScorer(
        registry,
        split,
        SimpleNamespace(get_claim=_unused),
        SimpleNamespace(get=_unused),
        fixtures.ENCODER_KEY,
    )(request)
    assert result[0] in (Status.INVALID_INPUT, Status.PROCESSING_ERROR)
    assert result[2:] == (None, None, False)
