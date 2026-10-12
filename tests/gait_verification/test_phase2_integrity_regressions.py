"""Synthetic regression evidence for Phase 2 cache trust boundaries; no gait accuracy."""

import json

import pytest
import test_phase1d

from gait_verification import dataset_processing as dp
from gait_verification.contracts import Status
from gait_verification.evaluation import load_trials
from gait_verification.offline_eval import (
    ClaimRequest,
    EmbeddingTrialScorer,
    QueryEmbeddingStore,
    RecordIntegrityError,
    seal,
)
from test_phase1d import ENCODER_KEY
from test_phase2 import build_world, config, needs_ffmpeg, run, spec

synthetic_eval = test_phase1d.synthetic_eval


@pytest.fixture
def processed_asset(tmp_path):
    w = build_world(tmp_path, subjects=("SYN001",), views=("left45",))
    cfg = config(w, spec(w.policy))
    assert run(w, cfg)["status"] == "COMPLETE"
    work = dp.WorkDir(w.work)
    asset = w.registry.asset(w.ids[0])
    return w, cfg, work, asset


def _query_request(doc, abstain):
    trial = next(
        t
        for t in load_trials(doc)
        if t.query_asset_id != abstain and t.label_same_identity
    )
    return ClaimRequest(
        trial.trial_id,
        trial.query_asset_id,
        trial.query_original_take_id,
        trial.query_variant_id,
        trial.claimed_subject_id,
        trial.reference_original_take_ids,
    )


@pytest.mark.parametrize("fingerprint", ["recipe", "pose_npz_sha256"])
def test_store_rejects_phase2_record_after_integrity_seal_is_removed(
    tmp_path, fingerprint
):
    store = QueryEmbeddingStore(tmp_path)
    record = seal(
        dict(
            schema_version="query-embedding-v1",
            asset_id="synthetic-asset",
            encoder_key="synthetic-encoder",
            values=[1.0, 0.0],
            **{fingerprint: "a" * 64},
        )
    )
    store.put(record)
    path = store._path(record["asset_id"], record["encoder_key"])
    del record["integrity_sha256"]
    path.write_text(json.dumps(record))
    with pytest.raises(RecordIntegrityError):
        store.get(record["asset_id"], record["encoder_key"])


def test_legacy_phase1d_unsealed_query_remains_scoreable(synthetic_eval):
    reg, split, doc, enrollments, queries, _, abstain = synthetic_eval
    request = _query_request(doc, abstain)
    record = queries.get(request.query_asset_id, ENCODER_KEY)
    assert "integrity_sha256" not in record and "recipe" not in record
    result = EmbeddingTrialScorer(reg, split, enrollments, queries, ENCODER_KEY)(
        request
    )
    assert result[0] == Status.POLICY_NOT_READY and result[2] is not None


def test_offline_scorer_rejects_phase2_query_with_removed_seal(synthetic_eval):
    reg, split, doc, enrollments, queries, _, abstain = synthetic_eval
    request = _query_request(doc, abstain)
    path = queries._path(request.query_asset_id, ENCODER_KEY)
    record = seal(
        dict(
            json.loads(path.read_text()),
            recipe="synthetic-phase2-recipe",
            pose_npz_sha256="a" * 64,
        )
    )
    del record["integrity_sha256"]
    path.write_text(json.dumps(record))
    result = EmbeddingTrialScorer(reg, split, enrollments, queries, ENCODER_KEY)(
        request
    )
    assert result[:2] == (Status.PROCESSING_ERROR, "QUERY_EMBEDDING_INTEGRITY")
    assert result[2:] == (None, None, False)


@needs_ffmpeg
@pytest.mark.parametrize("corruption", ["nonfinite", "missing_policy"])
def test_planner_classifies_malformed_quality_record_as_invalid(
    processed_asset, corruption
):
    w, cfg, work, asset = processed_asset
    path = dp.quality_path(work, asset, cfg)
    record = json.loads(path.read_text())
    if corruption == "nonfinite":
        record["summary"]["frames"] = float("nan")
    else:
        del record["policy"]
        record = seal(record)
    path.write_text(json.dumps(record))
    plan = dp.build_plan(w.registry, w.data, w.work, w.ids, cfg)
    task = next(t for t in plan["tasks"] if t["kind"] == "quality")
    assert task["state"] == "INVALID"


@needs_ffmpeg
@pytest.mark.parametrize("corruption", ["nonfinite", "missing_source"])
def test_planner_classifies_malformed_embedding_record_as_invalid(
    processed_asset, corruption
):
    w, cfg, work, asset = processed_asset
    model = cfg.models[0]
    path = dp.embed_store(work, cfg, model)._path(asset.asset_id, model.encoder_key)
    record = json.loads(path.read_text())
    if corruption == "nonfinite":
        record["values"][0] = float("nan")
    else:
        del record["source_sha256"]
        record = seal(record)
    path.write_text(json.dumps(record))
    plan = dp.build_plan(w.registry, w.data, w.work, w.ids, cfg)
    assert next(t for t in plan["tasks"] if t["kind"] == "embed")["state"] == "INVALID"


@needs_ffmpeg
@pytest.mark.parametrize("corruption", ["list", "missing_checksum"])
def test_planner_classifies_malformed_pose_sidecar_as_invalid(
    processed_asset, corruption
):
    w, cfg, work, asset = processed_asset
    path = work.pose / f"{dp.pose_key(asset, cfg)}.json"
    record = json.loads(path.read_text())
    if corruption == "list":
        record = []
    else:
        del record["npz_sha256"]
    path.write_text(json.dumps(record))
    plan = dp.build_plan(w.registry, w.data, w.work, w.ids, cfg)
    assert next(t for t in plan["tasks"] if t["kind"] == "pose")["state"] == "INVALID"
    assert all(
        t["state"] == "PENDING_UPSTREAM" for t in plan["tasks"] if t["kind"] != "pose"
    )


@needs_ffmpeg
@pytest.mark.parametrize(
    "field", ["pose_key", "asset_id", "original_take_id", "variant_id"]
)
def test_quality_cache_revalidates_sealed_dependency_fingerprints(
    processed_asset, field
):
    w, cfg, work, asset = processed_asset
    path = dp.quality_path(work, asset, cfg)
    record = json.loads(path.read_text())
    record[field] = "other-synthetic-source"
    path.write_text(json.dumps(seal(record)))
    plan = dp.build_plan(w.registry, w.data, w.work, w.ids, cfg)
    assert (
        next(t for t in plan["tasks"] if t["kind"] == "quality")["state"] == "INVALID"
    )


@needs_ffmpeg
@pytest.mark.parametrize("field", ["original_take_id", "variant_id", "pose_npz_sha256"])
def test_embedding_cache_revalidates_sealed_dependency_fingerprints(
    processed_asset, field
):
    w, cfg, work, asset = processed_asset
    model = cfg.models[0]
    path = dp.embed_store(work, cfg, model)._path(asset.asset_id, model.encoder_key)
    record = json.loads(path.read_text())
    record[field] = "other-synthetic-source"
    path.write_text(json.dumps(seal(record)))
    plan = dp.build_plan(w.registry, w.data, w.work, w.ids, cfg)
    assert next(t for t in plan["tasks"] if t["kind"] == "embed")["state"] == "INVALID"


@pytest.mark.parametrize(
    "damage",
    [
        "missing_source",
        "missing_status",
        "invalid_status",
        "invalid_model",
        "invalid_values",
    ],
)
def test_offline_scorer_preserves_error_status_for_malformed_query_record(
    synthetic_eval, damage
):
    reg, split, doc, enrollments, queries, _, abstain = synthetic_eval
    request = _query_request(doc, abstain)
    path = queries._path(request.query_asset_id, ENCODER_KEY)
    record = json.loads(path.read_text())
    if damage.startswith("missing_"):
        del record["source_sha256" if damage == "missing_source" else "status"]
    elif damage == "invalid_status":
        record["status"] = "EDITED"
    elif damage == "invalid_model":
        record["model_id"] = []
    else:
        record["values"] = [0.0, 0.0]
    path.write_text(json.dumps(seal(record)))
    result = EmbeddingTrialScorer(reg, split, enrollments, queries, ENCODER_KEY)(
        request
    )
    assert result == (
        Status.PROCESSING_ERROR,
        "QUERY_EMBEDDING_INTEGRITY",
        None,
        None,
        False,
    )
