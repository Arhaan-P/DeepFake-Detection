"""Offline claimed-identity trials. Subject identity is used only to label trials."""

from dataclasses import dataclass, asdict
import hashlib
import json
import numpy as np
from .contracts import Status


def validate_recordings(records):
    assets, subjects, takes, hashes = {}, {}, {}, {}
    for r in records:
        if r.asset_id in assets:
            raise ValueError("Duplicate asset ID")
        assets[r.asset_id] = r
        if not all(
            (
                r.original_take_id,
                r.subject_id,
                r.session_id,
                r.source_sha256,
                r.recorded_at,
            )
        ):
            raise ValueError("Missing subject/session/take/time/hash provenance")
        if r.split_role not in ("development", "calibration", "locked_final"):
            raise ValueError("Invalid split role")
        if subjects.setdefault(r.subject_id, r.split_role) != r.split_role:
            raise ValueError("Subject crosses split roles")
        group = (
            r.subject_id,
            r.session_id,
            r.split_role,
            r.view,
            r.recorded_at,
            r.source_sha256,
        )
        if takes.setdefault(r.original_take_id, group) != group:
            raise ValueError("Source lineage crosses metadata or split roles")
        if hashes.setdefault(r.source_sha256, r.original_take_id) != r.original_take_id:
            raise ValueError("Same source bytes have multiple take IDs")
    for r in records:
        if r.variant_id == "raw":
            if r.parent_asset_id is not None:
                raise ValueError("Original must not have a parent")
        else:
            parent = assets.get(r.parent_asset_id)
            if (
                parent is None
                or parent.variant_id != "raw"
                or parent.original_take_id != r.original_take_id
            ):
                raise ValueError("Derivative must point to its original asset")
    return assets


@dataclass(frozen=True)
class VerificationTrial:
    trial_id: str
    query_original_take_id: str
    query_variant_id: str
    query_subject_id: str  # offline clustering only; NEVER sent to verifier
    claimed_subject_id: str
    reference_original_take_ids: tuple[str, ...]
    label_same_identity: bool
    probe_view: str
    enroll_views: tuple[str, ...]
    query_session_id: str
    fold_id: str
    split_role: str
    trial_policy_version: str = "all-wrong-claims-v1"


def generate_trials(records, variant_id="raw"):
    """Deterministic rotate-one-take probes, all other-subject wrong claims.

    Every source original contributes one probe per candidate claim. Derivative
    selection is explicit; sources and descendants never become extra samples.
    """
    validate_recordings(records)
    selected = sorted(
        (r for r in records if r.variant_id == variant_id), key=lambda r: r.asset_id
    )
    if len({r.original_take_id for r in selected}) != len(selected):
        raise ValueError("Multiple variants/windows selected for a source take")
    trials = []
    for q in selected:
        claims = sorted(
            {r.subject_id for r in selected if r.split_role == q.split_role}
        )
        for claim in claims:
            refs = [
                r
                for r in selected
                if r.subject_id == claim and r.original_take_id != q.original_take_id
            ]
            if not refs:
                continue
            key = f"{q.original_take_id}:{variant_id}:{claim}:all-wrong-claims-v1"
            trials.append(
                VerificationTrial(
                    hashlib.sha256(key.encode()).hexdigest()[:24],
                    q.original_take_id,
                    variant_id,
                    q.subject_id,
                    claim,
                    tuple(r.original_take_id for r in refs),
                    q.subject_id == claim,
                    q.view,
                    tuple(r.view for r in refs),
                    q.session_id,
                    "rotate-take",
                    q.split_role,
                )
            )
    return trials


def manifest_hash(trials):
    return hashlib.sha256(
        json.dumps([asdict(t) for t in trials], sort_keys=True).encode()
    ).hexdigest()


def verification_metrics(
    labels, scores, statuses=None, threshold=None, target_far=0.05
):
    """Tie-aware empirical ROC. TAR uses achievable FAR<=target (no interpolation).

    EER interpolates the FAR/FRR crossing; it does not install an operating policy.
    None scores are required for abstentions. Undefined metrics carry reasons.
    """
    y = np.asarray(labels)
    if y.ndim != 1 or not np.isin(y, [0, 1]).all() or len(scores) != len(y):
        raise ValueError("Invalid labels/scores")
    if not 0 <= target_far <= 1:
        raise ValueError("Invalid target FAR")
    if threshold is not None and not np.isfinite(threshold):
        raise ValueError("Invalid threshold")
    states = [Status.OK] * len(y) if statuses is None else [Status(s) for s in statuses]
    if len(states) != len(y):
        raise ValueError("Status count mismatch")
    ok = np.array([s == Status.OK for s in states], dtype=bool)
    for score, accepted in zip(scores, ok):
        if accepted and (score is None or not np.isfinite(score)):
            raise ValueError("OK requires a finite score")
        if not accepted and score is not None:
            raise ValueError("Abstentions must not carry scores")
    result = {
        "total": len(y),
        "evaluable": int(ok.sum()),
        "coverage": float(ok.mean()) if len(y) else None,
        "coverage_by_class": {
            str(c): float(ok[y == c].mean()) if np.any(y == c) else None for c in (0, 1)
        },
        "status_counts": {s.value: states.count(s) for s in Status},
        "roc_auc": None,
        "eer": None,
        "tar_at_far": None,
        "target_far": target_far,
    }
    y = y[ok].astype(int)
    scores = np.asarray([s for s, keep in zip(scores, ok) if keep], dtype=float)
    p, n = int(y.sum()), int((1 - y).sum())
    if threshold is not None:
        accepted = scores >= threshold
        tp, fp = int(np.sum(accepted & (y == 1))), int(np.sum(accepted & (y == 0)))
        result.update(
            tp=tp,
            fp=fp,
            fn=p - tp,
            tn=n - fp,
            far=fp / n if n else None,
            frr=(p - tp) / p if p else None,
        )
    if p == 0 or n == 0:
        result["undefined_reason"] = (
            "UNDEFINED_METRIC: both evaluable claim classes required"
        )
        return result
    order = np.argsort(-scores, kind="stable")
    sy, ss = y[order], scores[order]
    ends = np.r_[np.flatnonzero(np.diff(ss) != 0), len(ss) - 1]
    tpr = np.r_[0, np.cumsum(sy)[ends] / p]
    fpr = np.r_[0, np.cumsum(1 - sy)[ends] / n]
    result["roc_auc"] = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
    diff = fpr + tpr - 1
    i = int(np.flatnonzero(diff >= 0)[0])
    a = 0.0 if diff[i] == 0 else -diff[i - 1] / (diff[i] - diff[i - 1])
    result["eer"] = float(
        fpr[i] if diff[i] == 0 else fpr[i - 1] + a * (fpr[i] - fpr[i - 1])
    )
    result["tar_at_far"] = float(tpr[fpr <= target_far].max())
    result["impostor_score_resolution"] = (
        1 / n
    )  # scores are correlated, not independent subjects
    return result


def evaluate_trials(trials, score_claim, threshold=None):
    """score_claim receives one trial and returns (Status, score-or-None).

    Preserve abstentions and provide view/subject/session grouping. Callback must
    use the trial's exact reference originals, never the entire subject store.
    """
    rows = []
    for trial in trials:
        status, score = score_claim(trial)
        rows.append((trial, status, score))

    def metrics(items):
        return verification_metrics(
            [r[0].label_same_identity for r in items],
            [r[2] for r in items],
            [r[1] for r in items],
            threshold,
        )

    return {
        "manifest_sha256": manifest_hash(trials),
        "overall": metrics(rows),
        "by_view": {
            v: metrics([r for r in rows if r[0].probe_view == v])
            for v in sorted({r[0].probe_view for r in rows})
        },
        "by_subject_session": {
            f"{s}/{session}": metrics(
                [
                    r
                    for r in rows
                    if (r[0].query_subject_id, r[0].query_session_id) == (s, session)
                ]
            )
            for s, session in sorted(
                {(r[0].query_subject_id, r[0].query_session_id) for r in rows}
            )
        },
    }


def subject_bootstrap(labels, scores, statuses, subject_ids, repetitions=1000, seed=0):
    """Resample probe subjects with all their trials, not individual pair scores.

    This is a probe-subject cluster CI. Claimed subjects are crossed/reused, so it
    is not a complete two-way cluster model; report that limitation in studies.
    """
    if repetitions < 1 or len(subject_ids) != len(labels):
        raise ValueError("Invalid bootstrap parameters")
    verification_metrics(labels, scores, statuses)
    subjects = sorted(set(subject_ids))
    if len(subjects) < 2:
        return {
            "status": "UNDEFINED_METRIC",
            "reason": "At least two probe subjects required",
        }
    ids = np.asarray(subject_ids)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repetitions):
        selection = rng.choice(subjects, len(subjects), replace=True)
        idx = np.concatenate([np.flatnonzero(ids == subject) for subject in selection])
        m = verification_metrics(
            [labels[i] for i in idx],
            [scores[i] for i in idx],
            [statuses[i] for i in idx],
        )
        if m["roc_auc"] is not None:
            draws.append(m["roc_auc"])
    return {
        "status": "OK" if draws else "UNDEFINED_METRIC",
        "seed": seed,
        "cluster": "probe_subject",
        "valid_replicates": len(draws),
        "requested_replicates": repetitions,
        "roc_auc_ci95": np.quantile(draws, [0.025, 0.975]).tolist() if draws else None,
    }
