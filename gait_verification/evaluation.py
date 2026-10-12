"""Offline claimed-identity trials and metrics. Subject identity only labels trials."""

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
    # verification-trials-v2 additions (empty for the legacy recording-list generator)
    query_asset_id: str = ""
    dataset_fingerprint: str = ""
    split_fingerprint: str = ""


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


TRIAL_SCHEMA = "verification-trials-v2"
REFERENCE_RULE = (
    "exclude_probe_view: references are the claimed subject's raw originals whose "
    "view differs from the probe view (never the probe's own original or its "
    "derivatives), so genuine and impostor claims use the same view rotation"
)


@dataclass(frozen=True)
class TrialPolicy:
    version: str
    seed: int
    impostor_claims_per_query: int  # sampled without replacement, deterministic
    query_variant_id: str = "raw"
    min_reference_originals: int = 1

    def validate(self):
        if not self.version or type(self.seed) is not int:
            raise ValueError("Trial policy needs a version and an integer seed")
        if type(self.impostor_claims_per_query) is not int or (
            self.impostor_claims_per_query < 1
        ):
            raise ValueError("impostor_claims_per_query must be a positive integer")
        if self.min_reference_originals < 1:
            raise ValueError("At least one reference original is required")


def _trial_id(version, query_asset_id, claim, refs):
    key = "|".join((version, query_asset_id, claim, *refs))
    return hashlib.sha256(key.encode()).hexdigest()[:24]


def _references(registry, claim, probe):
    return tuple(
        o
        for o in registry.originals(claim)
        if o.view != probe.view and o.original_take_id != probe.original_take_id
    )


def _bound_split(registry, split):
    from .splits import validate_split

    validate_split(split, registry.subjects())
    if split["status"] != "LOCKED":
        raise ValueError("Trials need a LOCKED subject split")
    for a in registry.assets():
        registry.recording(a.asset_id, split["assignments"][a.subject_id])


def generate_verification_trials(registry, split, policy, role, authorization=None):
    """Deterministic genuine + sampled impostor claims for ONE split role.

    Each registered query original (or its single registered `query_variant_id`
    derivative) yields one genuine claim and `impostor_claims_per_query` wrong claims.
    Wrong claims substitute only `claimed_subject_id`; the walker's subject is used
    solely to label the trial and is never part of a verifier input.
    """
    from .splits import require_roles

    policy.validate()
    _bound_split(registry, split)
    require_roles(split, (role,), authorization)
    subjects = sorted(s for s, r in split["assignments"].items() if r == role)
    common = dict(
        fold_id="",
        split_role=role,
        trial_policy_version=policy.version,
        dataset_fingerprint=registry.fingerprint,
        split_fingerprint=split["fingerprint"],
    )
    trials, skipped, shortfall = [], {}, 0
    for subject in subjects:
        for probe in registry.originals(subject):
            query = (
                probe
                if policy.query_variant_id == "raw"
                else registry.variant_of(
                    probe.original_take_id, policy.query_variant_id
                )
            )
            if query is None:
                skipped["NO_QUERY_VARIANT"] = skipped.get("NO_QUERY_VARIANT", 0) + 1
                continue
            eligible = {
                c: refs
                for c in subjects
                if len(refs := _references(registry, c, probe))
                >= policy.min_reference_originals
            }
            impostors = sorted(
                (c for c in eligible if c != subject),
                key=lambda c: hashlib.sha256(
                    f"{policy.seed}|{probe.original_take_id}|{c}".encode()
                ).hexdigest(),
            )[: policy.impostor_claims_per_query]
            shortfall += policy.impostor_claims_per_query - len(impostors)
            claims = ([subject] if subject in eligible else []) + impostors
            if subject not in eligible:
                skipped["GENUINE_TOO_FEW_REFERENCES"] = (
                    skipped.get("GENUINE_TOO_FEW_REFERENCES", 0) + 1
                )
            for claim in claims:
                refs = eligible[claim]
                ref_ids = tuple(r.original_take_id for r in refs)
                trials.append(
                    VerificationTrial(
                        _trial_id(policy.version, query.asset_id, claim, ref_ids),
                        probe.original_take_id,
                        query.variant_id,
                        subject,
                        claim,
                        ref_ids,
                        claim == subject,
                        probe.view,
                        tuple(r.view for r in refs),
                        probe.session_id,
                        **common
                        | {"fold_id": f"probe-view:{probe.view}"}
                        | {"query_asset_id": query.asset_id},
                    )
                )
    trials.sort(key=lambda t: t.trial_id)
    genuine = sum(t.label_same_identity for t in trials)
    doc = {
        "schema_version": TRIAL_SCHEMA,
        "trial_policy": asdict(policy) | {"reference_rule": REFERENCE_RULE},
        "split_role": role,
        "dataset_fingerprint": registry.fingerprint,
        "split_fingerprint": split["fingerprint"],
        "counts": {
            "trials": len(trials),
            "genuine": genuine,
            "impostor": len(trials) - genuine,
            "query_subjects": len({t.query_subject_id for t in trials}),
            "query_originals": len({t.query_original_take_id for t in trials}),
            "impostor_shortfall": shortfall,
            "skipped_queries": dict(sorted(skipped.items())),
        },
        "independence_note": "Trials are correlated within subjects and originals; "
        "the independent unit is the participant.",
        "trials": [asdict(t) for t in trials],
        "manifest_sha256": manifest_hash(trials),
    }
    validate_trial_manifest(doc, registry, split, authorization)
    return doc


def load_trials(doc):
    rows = []
    for row in doc["trials"]:
        row = dict(row)
        for key in ("reference_original_take_ids", "enroll_views"):
            row[key] = tuple(row[key])
        rows.append(VerificationTrial(**row))
    return rows


def validate_trial_manifest(doc, registry, split, authorization=None):
    """Re-check every invariant of a frozen trial manifest against its inputs."""
    from .splits import require_roles

    if doc.get("schema_version") != TRIAL_SCHEMA:
        raise ValueError("Unsupported trial manifest")
    if doc["dataset_fingerprint"] != registry.fingerprint:
        raise ValueError("Trial manifest belongs to a different capture registry")
    if doc["split_fingerprint"] != split["fingerprint"]:
        raise ValueError("Trial manifest belongs to a different split")
    _bound_split(registry, split)
    role = doc["split_role"]
    require_roles(split, (role,), authorization)
    trials = load_trials(doc)
    if manifest_hash(trials) != doc["manifest_sha256"]:
        raise ValueError("Trial manifest content does not match its hash")
    version = doc["trial_policy"]["version"]
    takes = {o.original_take_id: o for o in registry.originals()}
    ids, semantic = set(), set()
    for t in trials:
        if t.trial_id in ids:
            raise ValueError("Duplicate trial ID")
        ids.add(t.trial_id)
        if (t.query_asset_id, t.claimed_subject_id) in semantic:
            raise ValueError("Duplicate query/claim trial")
        semantic.add((t.query_asset_id, t.claimed_subject_id))
        query = registry.asset(t.query_asset_id)
        probe = registry.original(query.asset_id)
        assignments = split["assignments"]
        if (
            probe.original_take_id != t.query_original_take_id
            or query.variant_id != t.query_variant_id
            or probe.subject_id != t.query_subject_id
            or probe.view != t.probe_view
            or probe.session_id != t.query_session_id
        ):
            raise ValueError("Trial provenance disagrees with the capture registry")
        if t.label_same_identity != (t.claimed_subject_id == t.query_subject_id):
            raise ValueError("Incorrect genuine/impostor label")
        if t.split_role != role or {
            assignments[t.query_subject_id],
            assignments.get(t.claimed_subject_id),
        } != {role}:
            raise ValueError("Trial crosses split roles")
        if not t.reference_original_take_ids or len(
            set(t.reference_original_take_ids)
        ) != len(t.reference_original_take_ids):
            raise ValueError("Missing or repeated reference originals")
        lineage = registry.lineage_ids(query.asset_id)
        for take in t.reference_original_take_ids:
            ref = takes.get(take)
            if ref is None or ref.subject_id != t.claimed_subject_id:
                raise ValueError("Reference is not an original of the claimed subject")
            if take in lineage or ref.sha256 in lineage:
                raise ValueError("Query and reference share an original recording")
        if t.enroll_views != tuple(
            takes[k].view for k in t.reference_original_take_ids
        ):
            raise ValueError("Enrollment views disagree with references")
        if t.trial_policy_version != version or t.trial_id != _trial_id(
            version,
            t.query_asset_id,
            t.claimed_subject_id,
            t.reference_original_take_ids,
        ):
            raise ValueError("Trial ID does not match its content")
        if (t.dataset_fingerprint, t.split_fingerprint) != (
            doc["dataset_fingerprint"],
            doc["split_fingerprint"],
        ):
            raise ValueError("Trial fingerprints disagree with the manifest")
    return trials


SCORED_STATUSES = (Status.OK, Status.POLICY_NOT_READY)


def verification_metrics(
    labels, scores, statuses=None, threshold=None, target_far=0.05, target_fars=()
):
    """Tie-aware empirical ROC over trials WITH A SCORE; higher score = claim supported.

    A score may accompany `OK` (required) or `POLICY_NOT_READY` (valid score, no
    authorized threshold); every other status is an abstention and must carry None.
    `coverage` is the PRD operational coverage (status OK / submitted); `score_coverage`
    counts scored trials. TAR uses achievable FAR<=target (no interpolation). EER
    interpolates the FAR/FRR crossing and is descriptive only: it installs nothing.
    Undefined metrics stay None and carry an explicit reason in `undefined`.
    """
    y = np.asarray(labels)
    if y.ndim != 1 or not np.isin(y, [0, 1]).all() or len(scores) != len(y):
        raise ValueError("Invalid labels/scores")
    fars = tuple(sorted({target_far, *target_fars}))
    if not all(0 <= f <= 1 for f in fars):
        raise ValueError("Invalid target FAR")
    if threshold is not None and not np.isfinite(threshold):
        raise ValueError("Invalid threshold")
    states = [Status.OK] * len(y) if statuses is None else [Status(s) for s in statuses]
    if len(states) != len(y):
        raise ValueError("Status count mismatch")
    scored = np.zeros(len(y), dtype=bool)
    for i, (score, state) in enumerate(zip(scores, states)):
        if score is None:
            if state == Status.OK:
                raise ValueError("OK requires a finite score")
            continue
        if state not in SCORED_STATUSES:
            raise ValueError("Abstentions must not carry scores")
        if not np.isfinite(score):
            raise ValueError("Nonfinite score")
        scored[i] = True
    ok = np.array([s == Status.OK for s in states], dtype=bool)

    def share(mask, c=None):
        sel = np.ones(len(y), bool) if c is None else y == c
        return float(mask[sel].mean()) if sel.any() else None

    undefined = {}
    result = {
        "total": len(y),
        "genuine_total": int((y == 1).sum()),
        "impostor_total": int((y == 0).sum()),
        "evaluable": int(scored.sum()),
        "score_coverage": share(scored),
        "score_coverage_by_class": {str(c): share(scored, c) for c in (0, 1)},
        "operational_decisions": int(ok.sum()),
        "coverage": share(ok),
        "coverage_by_class": {str(c): share(ok, c) for c in (0, 1)},
        "status_counts": {s.value: states.count(s) for s in Status},
        "roc_auc": None,
        "eer": None,
        "tar_at_far": None,
        "tar_at_fars": {},
        "target_far": target_far,
        "undefined": undefined,
    }
    y = y[scored].astype(int)
    scores = np.asarray([s for s, keep in zip(scores, scored) if keep], dtype=float)
    p, n = int(y.sum()), int((1 - y).sum())
    result.update(genuine_scored=p, impostor_scored=n)
    if threshold is not None:
        accepted = scores >= threshold  # boundary: equal score accepts the claim
        tp, fp = int(np.sum(accepted & (y == 1))), int(np.sum(accepted & (y == 0)))
        result.update(tp=tp, fp=fp, fn=p - tp, tn=n - fp)
        result["far"] = fp / n if n else None
        result["frr"] = (p - tp) / p if p else None
        if not n:
            undefined["far"] = "UNDEFINED_METRIC: no scored impostor trials"
        if not p:
            undefined["frr"] = "UNDEFINED_METRIC: no scored genuine trials"
    if p == 0 or n == 0:
        reason = (
            "UNDEFINED_METRIC: both scored claim classes required "
            f"(genuine={p}, impostor={n})"
        )
        undefined.update(roc_auc=reason, eer=reason, tar_at_far=reason)
        result["undefined_reason"] = reason
        return result
    order = np.argsort(-scores, kind="stable")
    sy, ss = y[order], scores[order]
    ends = np.r_[
        np.flatnonzero(np.diff(ss) != 0), len(ss) - 1
    ]  # tied scores move together
    tpr = np.r_[0, np.cumsum(sy)[ends] / p]
    fpr = np.r_[0, np.cumsum(1 - sy)[ends] / n]
    result["roc_auc"] = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
    diff = fpr + tpr - 1  # FAR - FRR
    i = int(np.flatnonzero(diff >= 0)[0])
    a = 0.0 if diff[i] == 0 else -diff[i - 1] / (diff[i] - diff[i - 1])
    result["eer"] = float(
        fpr[i] if diff[i] == 0 else fpr[i - 1] + a * (fpr[i] - fpr[i - 1])
    )
    warnings = []
    for f in fars:
        keep = fpr <= f
        result["tar_at_fars"][str(f)] = {
            "tar": float(tpr[keep].max()),
            "empirical_far": float(fpr[keep].max()),
        }
        if 0 < f < 1 / n:
            warnings.append(
                f"TARGET_FAR_{f}_BELOW_RESOLUTION_1/{n}: only FAR=0 reachable"
            )
    result["tar_at_far"] = result["tar_at_fars"][str(target_far)]["tar"]
    # Pair scores are correlated within subjects; these are not independent people.
    result["impostor_score_resolution"] = 1 / n
    result["genuine_score_resolution"] = 1 / p
    result["warnings"] = warnings
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


BOOTSTRAP_METRICS = ("roc_auc", "eer", "tar_at_far")


def _cluster_draws(subject_ids, repetitions, seed):
    """Resample probe subjects WITH all their trials (sessions stay nested)."""
    ids = np.asarray(subject_ids)
    subjects = sorted(set(subject_ids))
    members = {s: np.flatnonzero(ids == s) for s in subjects}
    rng = np.random.default_rng(seed)
    for _ in range(repetitions):
        pick = rng.choice(len(subjects), len(subjects), replace=True)
        yield np.concatenate([members[subjects[k]] for k in pick])


def _interval(draws, requested, min_valid_fraction):
    if not draws:
        return None, "UNDEFINED_METRIC: no bootstrap replicate had both claim classes"
    if len(draws) < min_valid_fraction * requested:
        return None, (
            f"UNSTABLE: only {len(draws)}/{requested} replicates defined; "
            "no interval reported"
        )
    return np.quantile(draws, [0.025, 0.975]).tolist(), None


def subject_bootstrap(
    labels,
    scores,
    statuses,
    subject_ids,
    repetitions=1000,
    seed=0,
    target_far=0.05,
    min_valid_fraction=0.9,
):
    """Percentile CI from resampling probe SUBJECTS, never pair scores or windows.

    This is a probe-subject cluster CI. Claimed subjects are crossed/reused, so it
    is not a complete two-way cluster model; report that limitation in studies.
    """
    if repetitions < 1 or len(subject_ids) != len(labels):
        raise ValueError("Invalid bootstrap parameters")
    verification_metrics(labels, scores, statuses)
    n_subjects = len(set(subject_ids))
    if n_subjects < 2:
        return {
            "status": "UNDEFINED_METRIC",
            "reason": "At least two probe subjects required",
        }
    draws = {m: [] for m in BOOTSTRAP_METRICS}
    for idx in _cluster_draws(subject_ids, repetitions, seed):
        m = verification_metrics(
            [labels[i] for i in idx],
            [scores[i] for i in idx],
            [statuses[i] for i in idx],
            target_far=target_far,
        )
        if m["roc_auc"] is not None:
            for name in BOOTSTRAP_METRICS:
                draws[name].append(m[name])
    ci, reasons = {}, {}
    for name in BOOTSTRAP_METRICS:
        ci[name], reason = _interval(draws[name], repetitions, min_valid_fraction)
        if reason:
            reasons[name] = reason
    return {
        "status": "OK" if not reasons else reasons["roc_auc"].split(":")[0],
        "seed": seed,
        "cluster": "probe_subject",
        "subjects": n_subjects,
        "valid_replicates": len(draws["roc_auc"]),
        "requested_replicates": repetitions,
        "ci95": ci,
        "roc_auc_ci95": ci["roc_auc"],
        "limitations": reasons,
        "warnings": ["FEW_CLUSTERS: interval is not a reliable population estimate"]
        if n_subjects < 10
        else [],
    }


def paired_comparison(rows_a, rows_b, repetitions=1000, seed=0, target_far=0.05):
    """Descriptive paired difference on IDENTICAL trials; never selects a model.

    Rows: dicts with trial_id, label, query_subject_id, claimed_subject_id, split_role,
    status, score. Coverage is reported on the full common manifest; performance on
    the explicitly identified subset that BOTH systems scored. Nothing is discarded
    silently: abstentions stay in the full-manifest coverage.
    """
    a, b = {r["trial_id"]: r for r in rows_a}, {r["trial_id"]: r for r in rows_b}
    if len(a) != len(rows_a) or len(b) != len(rows_b) or a.keys() != b.keys():
        raise ValueError("Paired comparison needs exactly the same trial IDs")
    keys = ("label", "query_subject_id", "claimed_subject_id", "split_role")
    if any(tuple(a[t][k] for k in keys) != tuple(b[t][k] for k in keys) for t in a):
        raise ValueError("Trial labels/subjects/roles differ between systems")
    if any(r["split_role"] == "locked_final" for r in a.values()):
        from .splits import LockedTestAccessDenied

        raise LockedTestAccessDenied(
            "Locked-test scores are never used to compare models"
        )
    ids = sorted(a)
    common = [t for t in ids if a[t]["score"] is not None and b[t]["score"] is not None]

    def metrics(rows, subset):
        return verification_metrics(
            [rows[t]["label"] for t in subset],
            [rows[t]["score"] for t in subset],
            [rows[t]["status"] for t in subset],
            target_far=target_far,
        )

    full_a, full_b = metrics(a, ids), metrics(b, ids)
    common_a, common_b = metrics(a, common), metrics(b, common)
    delta = {
        m: None
        if common_a[m] is None or common_b[m] is None
        else common_a[m] - common_b[m]
        for m in BOOTSTRAP_METRICS
    }
    boot = {"status": "UNDEFINED_METRIC", "reason": "No common scoreable trials"}
    subjects = [a[t]["query_subject_id"] for t in common]
    if len(set(subjects)) >= 2:
        draws = {m: [] for m in BOOTSTRAP_METRICS}
        for idx in _cluster_draws(subjects, repetitions, seed):
            picked = [common[i] for i in idx]
            ma, mb = metrics(a, picked), metrics(b, picked)
            if ma["roc_auc"] is not None:
                for m in BOOTSTRAP_METRICS:
                    draws[m].append(ma[m] - mb[m])
        ci, reasons = {}, {}
        for m in BOOTSTRAP_METRICS:
            ci[m], reason = _interval(draws[m], repetitions, 0.9)
            if reason:
                reasons[m] = reason
        boot = {
            "status": "OK" if not reasons else "LIMITED",
            "cluster": "probe_subject",
            "seed": seed,
            "subjects": len(set(subjects)),
            "valid_replicates": len(draws["roc_auc"]),
            "delta_ci95": ci,
            "limitations": reasons,
        }
    elif common:
        boot["reason"] = "At least two probe subjects required"
    return {
        "trials": len(ids),
        "common_scoreable_trials": len(common),
        "common_scoreable_trial_ids_sha256": hashlib.sha256(
            "\n".join(common).encode()
        ).hexdigest(),
        "system_a": {"full_manifest": full_a, "common_scoreable": common_a},
        "system_b": {"full_manifest": full_b, "common_scoreable": common_b},
        "difference_a_minus_b": delta,
        "paired_bootstrap": boot,
        "interpretation": "Descriptive paired difference on identical trials; "
        "no model is selected and EER/TAR are not operating points.",
    }
