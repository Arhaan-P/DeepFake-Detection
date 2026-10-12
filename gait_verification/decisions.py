"""Owner decisions that are deliberately UNSET in code and config.

An operation that needs one raises `OwnerDecisionRequired` naming its register ID in
docs/gait_verification/DEFERRED_EVIDENCE_AND_DECISIONS.md. Validation, planning and
processing need none of these and never call this.
"""

DECISIONS = {
    "OWN-01": "subject split counts",
    "OWN-02": "split random seed",
    "OWN-03": "impostor claims per query",
    "OWN-04": "trial-generation random seed",
    "OWN-05": "target FAR operating point(s)",
    "OWN-08": "RAW versus stabilized recipe policy",
}


class OwnerDecisionRequired(ValueError):
    """The operation needs a project-owner decision that has not been recorded."""


def require_decision(value, decision_id, operation):
    """Return `value` unless it is unset (None or empty); never substitute a default."""
    if value is None or value == () or value == []:
        raise OwnerDecisionRequired(
            f"{operation} needs owner decision {decision_id} "
            f"({DECISIONS[decision_id]}), which is unset; see "
            "docs/gait_verification/DEFERRED_EVIDENCE_AND_DECISIONS.md"
        )
    return value


def load_decisions(path="configs/gait_verification/phase1d_evaluation.json"):
    """Owner decisions as recorded in the evaluation config (all null until decided)."""
    from .registry import load_json

    cfg = load_json(path)
    split, trial, metrics = cfg["split_policy"], cfg["trial_policy"], cfg["metrics"]
    return {
        "OWN-01": split["counts"]
        if all(v is not None for v in split["counts"].values())
        else None,
        "OWN-02": split["seed"],
        "OWN-03": trial["impostor_claims_per_query"],
        "OWN-04": trial["seed"],
        "OWN-05": metrics["target_fars"],
    }
