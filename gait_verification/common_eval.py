"""Architecture-only comparison on common trials; distinct training/score regimes."""

import hashlib
from itertools import combinations

from .e0 import SCORE_SEMANTICS
from .e0_checkpoint import SHIPPED_SHA256, TRAINING_ORIGIN
from .evaluation import verification_metrics

MATCHED_POLICY = "matched-pretraining-v1"
ARCHITECTURE_POLICY = "historical-e0-versus-grew-architecture-v1"


def validate_comparison_policy(system_a, system_b, policy):
    if policy == MATCHED_POLICY:
        if system_a["training_dataset"] != system_b["training_dataset"]:
            raise ValueError(
                "Different pretraining datasets are not a matched comparison (CASIA-B is separate)"
            )
        return
    if policy != ARCHITECTURE_POLICY:
        raise ValueError("Unknown comparison policy")
    systems = (system_a, system_b)
    datasets = {s["training_dataset"] for s in systems}
    if datasets == {"GREW"}:
        return  # unchanged GREW-matched comparison inside the architecture family
    if datasets != {"GREW", TRAINING_ORIGIN}:
        raise ValueError(
            "Architecture policy admits only audited historical E0 versus GREW; CASIA-B is separate"
        )
    e0 = next(s for s in systems if s["training_dataset"] == TRAINING_ORIGIN)
    if (
        e0.get("model_family"),
        e0.get("training_origin"),
        e0.get("source_checkpoint_sha256"),
        e0.get("score_semantics"),
    ) != ("e0_tcn", TRAINING_ORIGIN, SHIPPED_SHA256, SCORE_SEMANTICS):
        raise ValueError(
            "Architecture comparison requires audited E0 origin and native score semantics"
        )


def run_common_evaluation(
    trial_doc, registry, split, systems_and_scorers, evidence_class, **kwargs
):
    """Every scorer gets the same ClaimRequest manifest; scores are never fused."""
    from .offline_eval import run_evaluation

    items = list(systems_and_scorers)
    labels = [system["label"] for system, _ in items]
    if len(items) < 2 or len(set(labels)) != len(labels):
        raise ValueError("Common run requires at least two uniquely labeled systems")
    return [
        run_evaluation(
            trial_doc, registry, split, scorer, system, evidence_class, **kwargs
        )
        for system, scorer in items
    ]


def compare_common_runs(
    runs, target_far, comparison_policy=MATCHED_POLICY, repetitions=1000, seed=0
):
    """All-primary intersection + pairwise subsets; CASIA-B standalone ablations."""
    from .offline_eval import compare_runs
    from .decisions import require_decision

    require_decision(target_far, "OWN-05", "compare_common_runs")

    runs = list(runs)
    if len(runs) < 2:
        raise ValueError("At least two runs required")
    labels = [r[0]["system"]["label"] for r in runs]
    if len(set(labels)) != len(labels):
        raise ValueError("Duplicate system in common comparison")
    if any(r[0]["inputs"] != runs[0][0]["inputs"] for r in runs):
        raise ValueError("Common comparison requires identical trial manifests")
    primary = [r for r in runs if r[0]["system"]["training_dataset"] != "CASIA-B"]
    ablations = [r for r in runs if r[0]["system"]["training_dataset"] == "CASIA-B"]
    if len(primary) < 2:
        raise ValueError("At least two primary runs required; CASIA-B stays separate")
    # Validate trial identity for ALL reports, including standalone ablations,
    # without extending the architecture policy to cross-CASIA comparisons.
    from .evaluation import paired_comparison

    for other in runs[1:]:
        paired_comparison(runs[0][1], other[1], repetitions=0, target_far=target_far)
    pairs = [
        compare_runs(
            a, b, repetitions, seed, target_far, comparison_policy=comparison_policy
        )
        for a, b in combinations(primary, 2)
    ]
    mappings = [{r["trial_id"]: r for r in rows} for _, rows in primary]
    ids = sorted(mappings[0])
    common = [t for t in ids if all(m[t]["score"] is not None for m in mappings)]
    metrics = []
    for (report, _), mapping in zip(primary, mappings):
        rows = [mapping[t] for t in common]
        metrics.append(
            {
                "system": report["system"],
                "full_manifest_coverage": report["coverage"],
                "common_scoreable_metrics": verification_metrics(
                    [r["label"] for r in rows],
                    [r["score"] for r in rows],
                    [r["status"] for r in rows],
                    target_far=target_far,
                ),
            }
        )
    return {
        "schema_version": "common-model-comparison-v1",
        "comparison_policy": comparison_policy,
        "inputs": runs[0][0]["inputs"],
        "submitted_trials": len(ids),
        "common_scoreable_trials": len(common),
        "common_scoreable_trial_ids_sha256": hashlib.sha256(
            "\n".join(common).encode()
        ).hexdigest(),
        "primary_systems": metrics,
        "pairwise": pairs,
        "separate_ablations": [
            {
                "system": r[0]["system"],
                "coverage": r[0]["coverage"],
                "metrics": r[0]["metrics"]["overall"],
            }
            for r in ablations
        ],
        "interpretation": "Descriptive comparisons only; training populations differ for E0 versus GREW. "
        "No causal architecture-superiority claim, score fusion or common numeric threshold is supported.",
    }
