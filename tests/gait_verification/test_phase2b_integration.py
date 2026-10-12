"""Synthetic Phase 2B cache/scorer/common-manifest security regressions."""

from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import test_phase1d as fixtures
import test_phase2b_e0 as e0_fixtures

from gait_verification.contracts import Status
from gait_verification.e0_processing import E0FeatureStore, feature_recipe, process_e0
from gait_verification.e0_scorer import E0TrialScorer
from gait_verification.offline_eval import ClaimRequest, run_evaluation, seal
from gait_verification.pipeline import CacheIntegrityError

extracted = e0_fixtures.extracted


@pytest.fixture
def world(tmp_path):
    rows = fixtures.cohort_rows(3)
    root = tmp_path / "data"
    for row in rows:
        path = root / row["source_file_uri"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(row["original_take_id"].encode())
    registry = fixtures.make_registry(rows)
    split = fixtures.locked_split(
        registry, {"development": 2, "calibration": 0, "locked_final": 1}
    )
    doc = fixtures.generate_verification_trials(
        registry, split, fixtures.trial_policy(k=1), "development"
    )
    recipe = feature_recipe("a" * 64)
    store = E0FeatureStore(tmp_path / "work" / "e0-features", recipe)
    calls = []

    def extractor(path):
        calls.append(path)
        return np.full((60, 78), len(path.stem), np.float32)

    assets = [
        a.asset_id
        for a in registry.assets()
        if split["assignments"][a.subject_id] == "development"
    ]
    report = process_e0(registry, split, root, store, assets, extractor)
    assert report["status"] == "COMPLETE"
    request = ClaimRequest(
        **{k: doc["trials"][0][k] for k in ClaimRequest.__dataclass_fields__}
    )
    backend_calls = []

    def pair(q, refs):
        backend_calls.append((q, refs))
        return float(q.mean() - np.mean(refs)), {"frozen": True}

    backend = SimpleNamespace(score=pair, model_id="e0:fixture:mean-signature-v1")
    return SimpleNamespace(
        registry=registry,
        split=split,
        doc=doc,
        recipe=recipe,
        store=store,
        calls=calls,
        root=root,
        request=request,
        extractor=extractor,
        assets=assets,
        backend=backend,
        backend_calls=backend_calls,
    )


def scorer(w):
    return E0TrialScorer(w.registry, w.split, w.root, w.store, lambda: w.backend)


def test_resume_idempotence_no_extractor_or_bytes_changed(world):
    w = world
    before = {p: p.read_bytes() for p in w.store.root.rglob("*.json")}
    calls = len(w.calls)
    out = process_e0(w.registry, w.split, w.root, w.store, w.assets[::-1], w.extractor)
    assert out["status"] == "COMPLETE" and set(r["status"] for r in out["results"]) == {
        "SKIPPED_VALID"
    }
    assert len(w.calls) == calls
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize(
    "mutation",
    [
        "source",
        "recipe",
        "dimension",
        "nan",
        "seal_removed",
        "take",
        "variant",
        "lineage",
        "missing_values",
    ],
)
def test_cache_integrity_and_contract(world, mutation):
    w = world
    asset = w.registry.asset(w.request.query_asset_id)
    path = w.store.path(asset)
    record = json.loads(path.read_text())
    if mutation == "source":
        record["source_sha256"] = "0" * 64
    elif mutation == "recipe":
        record["recipe"]["sequence_length"] = 59
    elif mutation == "dimension":
        record["values"] = [[1.0]]
    elif mutation == "nan":
        record["values"][0][0] = float("nan")
    elif mutation == "seal_removed":
        del record["integrity_sha256"]
    elif mutation == "take":
        record["original_take_id"] = "wrong"
    elif mutation == "variant":
        record["variant_id"] = "derived"
    elif mutation == "lineage":
        record["parent_asset_id"] = "other"
    else:
        del record["values"]
    if mutation not in ("nan", "seal_removed"):
        record = seal(record)  # even a re-sealed stale record must be refused
    path.write_text(json.dumps(record))
    with pytest.raises(CacheIntegrityError):
        w.store.get(asset)
    result = scorer(w)(w.request)
    assert result[0] == Status.PROCESSING_ERROR and result[2:4] == (None, None)


def test_cache_unsealed_legacy_not_trusted(world):
    w = world
    path = w.store.path(w.registry.asset(w.request.query_asset_id))
    path.write_text(json.dumps({"values": np.zeros((60, 78)).tolist()}))
    out = process_e0(
        w.registry, w.split, w.root, w.store, [w.request.query_asset_id], w.extractor
    )
    assert out["status"] == "COMPLETE"
    assert list(path.parent.glob(path.name + ".invalid-*"))


def test_only_claimed_references_accessed_no_actual_identity(world, monkeypatch):
    w = world
    s = scorer(w)
    loaded = []
    get = w.store.get
    monkeypatch.setattr(
        w.store, "get", lambda asset: (loaded.append(asset.asset_id), get(asset))[1]
    )
    status, _, score, verdict, quality = s(w.request)
    assert (
        status == Status.POLICY_NOT_READY
        and np.isfinite(score)
        and verdict is None
        and quality
    )
    expected = {w.request.query_asset_id} | {
        a.asset_id
        for a in w.registry.originals(w.request.claimed_subject_id)
        if a.original_take_id in w.request.reference_original_take_ids
    }
    assert set(loaded) == expected
    assert "query_subject_id" not in vars(w.request) and "label" not in vars(w.request)
    assert s.lookups == [w.request.claimed_subject_id]


@pytest.mark.parametrize(
    "failure",
    [
        "source_changed",
        "missing_query",
        "missing_reference",
        "missing_checkpoint",
        "nonfinite_score",
        "unenrolled",
        "wrong_reference",
        "shared_original",
        "wrong_query_take",
    ],
)
def test_explicit_failure_status(world, failure):
    w, r = world, world.request
    if failure == "source_changed":
        w.registry.path_of(w.registry.asset(r.query_asset_id), w.root).write_bytes(
            b"altered"
        )
    elif failure == "missing_query":
        w.store.path(w.registry.asset(r.query_asset_id)).unlink()
    elif failure == "missing_reference":
        ref = next(
            a
            for a in w.registry.originals(r.claimed_subject_id)
            if a.original_take_id in r.reference_original_take_ids
        )
        w.store.path(ref).unlink()
    elif failure == "unenrolled":
        r = replace(r, claimed_subject_id="NO_SUCH_CLAIM")
    elif failure == "wrong_reference":
        ref = next(
            a for a in w.registry.originals() if a.subject_id != r.claimed_subject_id
        )
        r = replace(r, reference_original_take_ids=(ref.original_take_id,))
    elif failure == "shared_original":
        r = replace(r, reference_original_take_ids=(r.query_original_take_id,))
    elif failure == "wrong_query_take":
        r = replace(r, query_original_take_id="wrong")
    if failure == "missing_checkpoint":

        def missing():
            raise FileNotFoundError("private path")

        s = E0TrialScorer(w.registry, w.split, w.root, w.store, missing)
    elif failure == "nonfinite_score":
        w.backend.score = lambda *a: (float("nan"), {})
        s = scorer(w)
    else:
        s = scorer(w)
    result = s(r)
    assert result[0] not in (Status.OK, Status.POLICY_NOT_READY)
    assert result[2:4] == (None, None)
    assert "private" not in str(result)


def test_locked_final_processing_and_scoring_refused_before_bytes(world, monkeypatch):
    w = world
    locked = next(
        a
        for a in w.registry.assets()
        if w.split["assignments"][a.subject_id] == "locked_final"
    )
    monkeypatch.setattr(
        w.registry, "path_of", lambda *args: pytest.fail("protected bytes accessed")
    )
    out = process_e0(
        w.registry, w.split, w.root, w.store, [locked.asset_id], w.extractor
    )
    assert (
        out["status"] == "PARTIAL"
        and out["results"][0]["reason"] == "LOCKED_FINAL_DENIED"
    )
    r = replace(
        w.request,
        query_asset_id=locked.asset_id,
        query_original_take_id=locked.original_take_id,
    )
    assert scorer(w)(r)[0] == Status.INVALID_INPUT


def test_same_manifest_preserved_scored_undecided_coverage(world):
    w = world
    report, rows = run_evaluation(
        w.doc,
        w.registry,
        w.split,
        scorer(w),
        {
            "label": "e0",
            "encoder_key": w.backend.model_id,
            "training_dataset": "historical-local-13-subject-shipped",
        },
        "SYNTHETIC_FIXTURE",
        target_fars=(0.05,),
        bootstrap_repetitions=20,
    )
    assert [r["trial_id"] for r in rows] == [r["trial_id"] for r in w.doc["trials"]]
    assert report["coverage"] == {"score": 1.0, "quality": 1.0, "operational": 0.0}
    assert all(r["verdict"] is None for r in rows)
    assert "far" not in report["metrics"]["overall"]
    assert "frr" not in report["metrics"]["overall"]


def test_abstention_and_all_undefined(world):
    w = world
    for a in w.registry.assets():
        if w.store.path(a).exists():
            w.store.path(a).unlink()
    process_e0(w.registry, w.split, w.root, w.store, w.assets, lambda path: None)
    report, rows = run_evaluation(
        w.doc,
        w.registry,
        w.split,
        scorer(w),
        {
            "label": "e0",
            "encoder_key": "e0-fixture",
            "training_dataset": "historical-local-13-subject-shipped",
        },
        "SYNTHETIC_FIXTURE",
        target_fars=(0.05,),
        bootstrap_repetitions=20,
    )
    assert all(r["status"] == "INSUFFICIENT_GAIT" for r in rows)
    assert report["metrics"]["overall"]["roc_auc"] is None
    assert report["coverage"]["score"] == 0


def test_registered_derivative_of_reference_is_rejected(world):
    w = world
    parent = w.registry.asset(w.request.query_asset_id)
    row = next(
        r for r in w.registry.document["assets"] if r["asset_id"] == parent.asset_id
    )
    child = fixtures.derived_row(row, "vidstab-s10")
    registry = fixtures.make_registry([*w.registry.document["assets"], child])
    s = E0TrialScorer(registry, w.split, w.root, w.store, lambda: w.backend)
    r = replace(
        w.request,
        query_asset_id=child["asset_id"],
        query_variant_id="vidstab-s10",
        reference_original_take_ids=(parent.original_take_id,),
    )
    assert s(r)[0:2] == (Status.INVALID_INPUT, "SHARED_ORIGINAL")
    assert not w.backend_calls


def test_unknown_selection_and_output_over_source_cannot_complete(world):
    w = world
    report = process_e0(
        w.registry, w.split, w.root, w.store, ["no-such-asset"], w.extractor
    )
    assert report["status"] == "PARTIAL"
    unsafe = E0FeatureStore(w.root, w.recipe)
    report = process_e0(
        w.registry, w.split, w.root, unsafe, [w.request.query_asset_id], w.extractor
    )
    assert report["status"] == "PARTIAL"
    assert not (w.root / "ledger.jsonl").exists()


def test_source_mutation_during_extraction_not_published(world):
    w = world
    asset = w.registry.asset(w.request.query_asset_id)
    w.store.path(asset).unlink()

    def mutate(path):
        path.write_bytes(b"changed")
        return np.zeros((60, 78), np.float32)

    report = process_e0(w.registry, w.split, w.root, w.store, [asset.asset_id], mutate)
    assert report["status"] == "PARTIAL" and not w.store.path(asset).exists()


def test_recipe_change_has_new_cache_key(world):
    w = world
    asset = w.registry.asset(w.request.query_asset_id)
    recipe = dict(w.recipe, min_detection_confidence=0.6)
    other = E0FeatureStore(w.store.root, recipe)
    assert other.path(asset) != w.store.path(asset)
    assert other.get(asset) is None and w.store.get(asset) is not None


def test_interrupt_then_resume(world):
    w = world
    for a in w.registry.assets():
        w.store.path(a).unlink(missing_ok=True)
    count = 0

    def extract(path):
        nonlocal count
        count += 1
        return w.extractor(path)

    partial = process_e0(
        w.registry, w.split, w.root, w.store, w.assets, extract, lambda: count == 1
    )
    assert partial["status"] == "INTERRUPTED"
    final = process_e0(w.registry, w.split, w.root, w.store, w.assets, extract)
    assert final["status"] == "COMPLETE"
    assert count == len(w.assets)


def test_scorer_public_result_has_no_other_people_or_scores(world):
    from gait_verification.contracts import VerificationResult

    w = world
    status, _, _, verdict, _ = scorer(w)(w.request)
    public = VerificationResult(
        w.request.claimed_subject_id, status, verdict
    ).public_dict()
    assert set(public) == {"schema_version", "claimed_subject_id", "status", "verdict"}
    assert public["status"] == "POLICY_NOT_READY" and public["verdict"] is None


def test_real_frozen_head_through_managed_synthetic_scorer(world, extracted):
    from gait_verification.e0 import FrozenE0

    w = world
    s = E0TrialScorer(
        w.registry,
        w.split,
        w.root,
        w.store,
        lambda: FrozenE0(extracted[0], extracted[1]["model_only_sha256"], device="cpu"),
    )
    status, _, value, verdict, quality = s(w.request)
    assert status == Status.POLICY_NOT_READY and np.isfinite(value)
    assert verdict is None and quality
    assert len(s.recipes) == 1
