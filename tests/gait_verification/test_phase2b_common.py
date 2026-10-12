"""Declared architecture comparison uses each score scale separately."""

import copy

import pytest
import test_phase1d as fixtures

from gait_verification.e0 import SCORE_SEMANTICS
from gait_verification.e0_checkpoint import SHIPPED_SHA256, TRAINING_ORIGIN
from gait_verification.offline_eval import (
    EmbeddingTrialScorer,
    compare_runs,
    run_evaluation,
)
from gait_verification.common_eval import (
    ARCHITECTURE_POLICY,
    compare_common_runs,
    run_common_evaluation,
)

synthetic_eval = fixtures.synthetic_eval


@pytest.fixture
def runs(synthetic_eval):
    reg, split, doc, store, queries, system, _ = synthetic_eval
    a = run_evaluation(
        doc,
        reg,
        split,
        EmbeddingTrialScorer(reg, split, store, queries, fixtures.ENCODER_KEY),
        system,
        "SYNTHETIC_FIXTURE",
        target_fars=(0.05,),
        bootstrap_repetitions=10,
    )
    b = copy.deepcopy(a)
    b[0]["system"] = {
        "label": "e0",
        "encoder_key": "e0:fixture",
        "model_family": "e0_tcn",
        "training_dataset": TRAINING_ORIGIN,
        "training_origin": TRAINING_ORIGIN,
        "source_checkpoint_sha256": SHIPPED_SHA256,
        "score_semantics": SCORE_SEMANTICS,
    }
    # A monotonic transform keeps ranking mathematics but never combines scales.
    for row in b[1]:
        if row["score"] is not None:
            row["score"] = 100 * row["score"] + 27
    return a, b


def test_default_guard_unchanged_and_explicit_architecture_policy(runs):
    a, b = runs
    with pytest.raises(ValueError, match="pretraining"):
        compare_runs(a, b, target_far=0.05)
    out = compare_runs(
        a, b, target_far=0.05, comparison_policy=ARCHITECTURE_POLICY, repetitions=10
    )
    assert out["comparison_policy"] == ARCHITECTURE_POLICY
    assert "training populations differ" in out["interpretation"]
    assert (
        out["system_a"]["common_scoreable"]["roc_auc"]
        == out["system_b"]["common_scoreable"]["roc_auc"]
    )
    assert out["full_manifest_coverage"][0]["operational"] == 0


@pytest.mark.parametrize(
    "bad",
    [
        "CASIA-B",
        "unknown_policy",
        "unknown_origin",
        "fake_source",
        "wrong_score",
        "different_manifest",
        "locked_final",
    ],
)
def test_architecture_exception_is_narrow(runs, bad):
    a, b = copy.deepcopy(runs)
    policy = ARCHITECTURE_POLICY
    if bad == "CASIA-B":
        a[0]["system"]["training_dataset"] = "CASIA-B"
    elif bad == "unknown_policy":
        policy = "allow_anything"
    elif bad == "unknown_origin":
        b[0]["system"]["training_origin"] = "unknown"
    elif bad == "fake_source":
        b[0]["system"]["source_checkpoint_sha256"] = "0" * 64
    elif bad == "wrong_score":
        b[0]["system"]["score_semantics"] = "cosine"
    elif bad == "different_manifest":
        b[0]["inputs"]["trial_manifest_sha256"] = "0" * 64
    else:
        for rows in (a[1], b[1]):
            rows[0]["split_role"] = "locked_final"
    with pytest.raises((ValueError, PermissionError)):
        compare_runs(a, b, target_far=0.05, comparison_policy=policy)


def test_all_system_common_subset_and_separate_casia(runs):
    a, b = copy.deepcopy(runs)
    third = copy.deepcopy(a)
    third[0]["system"]["label"] = "gpgait-fixture"
    third[1][0].update(score=None, status="INSUFFICIENT_GAIT", quality_evaluable=False)
    casia = copy.deepcopy(a)
    casia[0]["system"].update(label="casia-fixture", training_dataset="CASIA-B")
    out = compare_common_runs(
        [a, b, third, casia],
        target_far=0.05,
        comparison_policy=ARCHITECTURE_POLICY,
        repetitions=10,
    )
    assert out["separate_ablations"][0]["system"]["training_dataset"] == "CASIA-B"
    assert len(out["primary_systems"]) == 3
    expected = sum(
        all(run[1][i]["score"] is not None for run in [a, b, third])
        for i in range(len(a[1]))
    )
    assert out["common_scoreable_trials"] == expected
    assert all("winner" not in key for key in out)


def test_common_runner_uses_same_claim_requests_and_no_actual_identity(synthetic_eval):
    reg, split, doc, _, _, system, _ = synthetic_eval
    seen = [[], []]
    from gait_verification.contracts import Status

    def make(i):
        def score(request):
            seen[i].append(request)
            return Status.POLICY_NOT_READY, "NO_POLICY", float(i), None, True

        return score

    results = run_common_evaluation(
        doc,
        reg,
        split,
        [(system, make(0)), (dict(system, label="other"), make(1))],
        "SYNTHETIC_FIXTURE",
        target_fars=(0.05,),
        bootstrap_repetitions=10,
    )
    assert seen[0] == seen[1]
    assert [o["trial_id"] for o in results[0][1]] == [
        o["trial_id"] for o in results[1][1]
    ]
    assert all(
        not hasattr(r, "label_same_identity") and not hasattr(r, "query_subject_id")
        for r in seen[0]
    )
