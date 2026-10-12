"""Restricted, research-only offline evaluation over a FROZEN trial manifest.

Every submitted trial is preserved with separate flags:
  score_available                 a claim-only similarity was computed
  quality_evaluable               the query passed pose/window/encoder quality
  operational_decision_available  status OK under an authorized policy (none exists)
`POLICY_NOT_READY` with a score is a valid scored trial with no threshold; it is
never a quality failure and never a TRUE/FALSE verdict.

A scorer receives a `ClaimRequest` (query lineage + claimed ID + reference takes). It
never sees the walker's subject or the trial label, so the same runner can host the
frozen encoders today and an E0 pairwise adapter later on identical trials.
Outputs are private local research artifacts (0600): per-trial scores never belong
in public reports, and no embedding, consent record or file path is written to them.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from .capture_registry import ProvenanceError, write_new_json
from .contracts import GaitEmbedding, QualityAssessment, Status
from .evaluation import (
    paired_comparison,
    subject_bootstrap,
    validate_trial_manifest,
    verification_metrics,
)
from .media import InvalidMedia, sha256_file
from .skeleton_contracts import InsufficientGait
from .verifier import ClaimVerifier

QUERY_SCHEMA = "query-embedding-v1"
RUN_SCHEMA = "offline-evaluation-v1"
EVIDENCE_CLASSES = (
    "SYNTHETIC_FIXTURE",
    "ENGINEERING_SMOKE",
    "STUDY_DEVELOPMENT",
    "STUDY_CALIBRATION",
    "STUDY_LOCKED_FINAL",
)
_PERSISTED = ("EMBEDDED", "INSUFFICIENT_GAIT", "INVALID_INPUT")


@dataclass(frozen=True)
class ClaimRequest:
    """Operational input of one trial: no walker identity, no label.

    A scorer returns (Status, reason, score-or-None, verdict-or-None, quality_evaluable).
    """

    trial_id: str
    query_asset_id: str
    query_original_take_id: str
    query_variant_id: str
    claimed_subject_id: str
    reference_original_take_ids: tuple


class QueryEmbeddingStore:
    """One private JSON per (query asset, encoder). Holds no subject identity."""

    def __init__(self, root):
        self.root = Path(root)

    def _path(self, asset_id, encoder_key):
        key = hashlib.sha256(f"{asset_id}|{encoder_key}".encode()).hexdigest()[:32]
        return self.root / f"{key}.json"

    def get(self, asset_id, encoder_key):
        path = self._path(asset_id, encoder_key)
        if not path.exists():
            return None
        record = json.loads(path.read_text())
        if (
            record.get("schema_version") != QUERY_SCHEMA
            or record["asset_id"] != asset_id
            or record["encoder_key"] != encoder_key
        ):
            raise ValueError("Query embedding record does not match its key")
        return record

    def put(self, record):
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        return write_new_json(
            self._path(record["asset_id"], record["encoder_key"]), record
        )


def embed_query(video_path, extractor, encoder, quality_assessor):
    """Same extractor/quality/encoder path as ClaimVerifier, without any claim."""
    try:
        sequence = extractor.extract(video_path)
        quality = quality_assessor(sequence)
        if not quality.usable:
            return "INSUFFICIENT_GAIT", "QUALITY_GATE", None, list(quality.reasons)
        embedding = encoder.encode(sequence)
        embedding.normalized()
        return "EMBEDDED", "OK", embedding, []
    except (InvalidMedia, FileNotFoundError):
        return "INVALID_INPUT", "INVALID_MEDIA", None, []
    except InsufficientGait as exc:
        return "INSUFFICIENT_GAIT", "ENCODER_ABSTAINED", None, str(exc).split(",")
    except Exception as exc:
        return "PROCESSING_ERROR", type(exc).__name__, None, []


def _cuda_peak(start):
    import sys

    torch = sys.modules.get("torch")  # in-process models only; workers not measured
    if torch is None or not torch.cuda.is_available():
        return None
    if start:
        torch.cuda.reset_peak_memory_stats()
        return None
    return torch.cuda.max_memory_allocated() / 2**20


def embed_registered_queries(
    registry,
    asset_ids,
    data_root,
    extractor_factory,
    encoder,
    quality_assessor,
    store,
):
    """Embed EXPLICIT registered query assets once per encoder (idempotent).

    Pose/quality/media outcomes are persisted (including abstentions); transient
    runtime errors and missing/altered files are reported but not persisted, so a
    rerun retries them. `extractor_factory` receives lineage only.
    """
    encoder_key = encoder.metadata.model_id
    results = []
    for asset_id in sorted(set(asset_ids)):
        row = {"asset_id": asset_id}
        results.append(row)
        try:
            asset = registry.asset(asset_id)
            if store.get(asset_id, encoder_key) is not None:
                row["status"] = "EXISTS"
                continue
            path = registry.path_of(asset, data_root)
            if not path.is_file() or sha256_file(path) != asset.sha256:
                row.update(status="REFUSED", reason="SOURCE_MISSING_OR_CHANGED")
                continue
        except (ProvenanceError, ValueError, OSError) as exc:
            row.update(status="REFUSED", reason=f"{type(exc).__name__}: {exc}")
            continue
        _cuda_peak(True)
        began = time.perf_counter()
        status, reason, embedding, reasons = embed_query(
            path,
            extractor_factory(asset.original_take_id, asset.variant_id),
            encoder,
            quality_assessor,
        )
        row.update(status=status, reason=reason)
        if status not in _PERSISTED:
            continue
        values = embedding.normalized().tolist() if embedding is not None else None
        store.put(
            {
                "schema_version": QUERY_SCHEMA,
                "asset_id": asset_id,
                "original_take_id": asset.original_take_id,
                "variant_id": asset.variant_id,
                "source_sha256": asset.sha256,
                "encoder_key": encoder_key,
                "model_id": embedding.model_id if embedding is not None else None,
                "status": status,
                "reason": reason,
                "quality_reasons": reasons,
                "values": values,
                "runtime_sec": time.perf_counter() - began,
                "peak_cuda_mib": _cuda_peak(False),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    return results


class _TrialReferences:
    """Store view holding exactly one claim's selected references for one trial."""

    def __init__(self, claim, templates):
        self.claim, self.templates = claim, tuple(templates)

    def get_claim(self, claim):
        if claim != self.claim:
            raise LookupError("Only the claimed subject is available")
        return self.templates


class EmbeddingTrialScorer:
    """Claim-only scoring of stored query embeddings through ClaimVerifier."""

    def __init__(
        self,
        registry,
        split,
        store,
        query_store,
        encoder_key,
        policy=None,
        allow_test_policy=False,
        model_ready=False,
    ):
        if policy is not None and not (
            policy.ready(allow_test_policy) and (policy.test_only or model_ready)
        ):
            raise ValueError("No authorized operational policy for this model")
        self.registry, self.split, self.store = registry, split, store
        self.query_store, self.encoder_key = query_store, encoder_key
        self.policy, self.allow_test_policy = policy, allow_test_policy
        self.recipes, self.runtime, self.lookups = set(), {}, []
        self._claims = {}
        self._originals = {o.original_take_id: o for o in registry.originals()}

    def _references(self, claim):
        if claim not in self._claims:
            self.lookups.append(claim)
            try:
                self._claims[claim] = tuple(self.store.get_claim(claim))
            except Exception:
                self._claims[claim] = None
        return self._claims[claim]

    def __call__(self, request):
        r = request
        record = self.query_store.get(r.query_asset_id, self.encoder_key)
        asset = self.registry.asset(r.query_asset_id)
        if record is not None:
            self.runtime[r.query_asset_id] = (
                record.get("runtime_sec"),
                record.get("peak_cuda_mib"),
            )
            if (record["source_sha256"], record["original_take_id"]) != (
                asset.sha256,
                r.query_original_take_id,
            ):
                return (
                    Status.INVALID_INPUT,
                    "QUERY_PROVENANCE_MISMATCH",
                    None,
                    None,
                    False,
                )
            if record["status"] == "INVALID_INPUT":
                return Status.INVALID_INPUT, record["reason"], None, None, False
        quality = record is not None and record["status"] == "EMBEDDED"
        templates = self._references(r.claimed_subject_id)
        if templates is None:
            return Status.PROCESSING_ERROR, "STORE_ERROR", None, None, quality
        if not templates:
            return Status.CLAIM_NOT_ENROLLED, "NO_REFERENCES", None, None, quality
        wanted = set(r.reference_original_take_ids)
        if not wanted <= {t.recording.original_take_id for t in templates}:
            return (
                Status.CLAIM_NOT_ENROLLED,
                "ENROLLMENT_INCOMPLETE",
                None,
                None,
                quality,
            )
        if record is None:
            return Status.PROCESSING_ERROR, "QUERY_EMBEDDING_MISSING", None, None, False
        if not quality:
            return Status(record["status"]), record["reason"], None, None, False
        recipe = record["model_id"]
        self.recipes.add(recipe)
        selected = [
            t
            for t in templates
            if t.recording.original_take_id in wanted and t.embedding.model_id == recipe
        ]
        if {t.recording.original_take_id for t in selected} != wanted:
            return Status.CLAIM_NOT_ENROLLED, "RECIPE_MISMATCH", None, None, quality
        role = self.split["assignments"][r.claimed_subject_id]
        for t in selected:
            ref = self._originals[t.recording.original_take_id]
            if (t.recording.source_sha256, t.recording.split_role) != (
                ref.sha256,
                role,
            ):
                return (
                    Status.CLAIM_NOT_ENROLLED,
                    "ENROLLMENT_PROVENANCE_MISMATCH",
                    None,
                    None,
                    quality,
                )
        probe = GaitEmbedding(
            np.asarray(record["values"], dtype=np.float32),
            recipe,
            record["original_take_id"],
            record["source_sha256"],
        )
        verifier = ClaimVerifier(
            _TrialReferences(r.claimed_subject_id, selected),
            policy=self.policy,
            allow_test_policy=self.allow_test_policy,
        )
        result, diag = verifier.verify_embedding_detailed(
            probe,
            r.claimed_subject_id,
            QualityAssessment(True),
            self.registry.lineage_ids(r.query_asset_id),
        )
        return result.status, diag["reason"], diag.get("score"), result.verdict, True


def _git_revision():
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True
            ).stdout.strip()
        )
        return {"head": head, "dirty_working_tree": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"head": None, "dirty_working_tree": None}


def _group(outcomes, key, target_fars):
    groups = {}
    for o in outcomes:
        groups.setdefault(str(o[key]), []).append(o)
    return {k: _metrics(v, target_fars) for k, v in sorted(groups.items())}


def _metrics(outcomes, target_fars):
    return verification_metrics(
        [o["label"] for o in outcomes],
        [o["score"] for o in outcomes],
        [o["status"] for o in outcomes],
        target_far=target_fars[0],
        target_fars=target_fars,
    )


def run_evaluation(
    trial_doc,
    registry,
    split,
    scorer,
    system,
    evidence_class,
    authorization=None,
    target_fars=(0.05,),
    bootstrap_repetitions=1000,
    bootstrap_seed=0,
):
    """Score every trial of a validated frozen manifest; return (report, outcomes).

    `system` = {label, encoder_key, training_dataset}. The scorer only ever gets a
    ClaimRequest. Nothing is dropped: missing data become explicit statuses.
    """
    if evidence_class not in EVIDENCE_CLASSES:
        raise ValueError("Unknown evidence class")
    if (evidence_class == "STUDY_LOCKED_FINAL") != (
        trial_doc["split_role"] == "locked_final"
    ):
        raise ValueError("Evidence class disagrees with the trial split role")
    trials = validate_trial_manifest(trial_doc, registry, split, authorization)
    outcomes = []
    for t in trials:
        status, reason, score, verdict, quality = scorer(
            ClaimRequest(
                t.trial_id,
                t.query_asset_id,
                t.query_original_take_id,
                t.query_variant_id,
                t.claimed_subject_id,
                t.reference_original_take_ids,
            )
        )
        status = Status(status)
        if score is not None and status not in (Status.OK, Status.POLICY_NOT_READY):
            raise ValueError("Scorer returned a score with an abstention status")
        outcomes.append(
            {
                "trial_id": t.trial_id,
                "label": int(t.label_same_identity),
                "query_subject_id": t.query_subject_id,
                "claimed_subject_id": t.claimed_subject_id,
                "query_asset_id": t.query_asset_id,
                "split_role": t.split_role,
                "probe_view": t.probe_view,
                "query_session": f"{t.query_subject_id}/{t.query_session_id}",
                "fold_id": t.fold_id,
                "status": status.value,
                "reason": reason,
                "score": score,
                "verdict": verdict,
                "score_available": score is not None,
                "quality_evaluable": bool(quality),
                "operational_decision_available": status == Status.OK,
            }
        )
    recipes = sorted(getattr(scorer, "recipes", ()))
    if len(recipes) > 1:
        raise ValueError("One run must use exactly one model recipe")
    reasons = {}
    for o in outcomes:
        key = f"{o['status']}/{o['reason']}"
        cls = "genuine" if o["label"] else "impostor"
        reasons.setdefault(key, {"genuine": 0, "impostor": 0})[cls] += 1
    n = len(outcomes)
    overall = _metrics(outcomes, target_fars)
    sessions = {}
    for o in outcomes:
        sessions.setdefault(o["query_subject_id"], set()).add(o["query_session"])
    warnings = [
        "Descriptive offline metrics only: no threshold, calibration or model "
        "selection is implied; EER/TAR are not operating points."
    ]
    if not evidence_class.startswith("STUDY_"):
        warnings.append(
            f"{evidence_class}: NOT biometric performance evidence for any cohort."
        )
    if not any(len(s) > 1 for s in sessions.values()):
        warnings.append(
            "No subject has more than one session: results are cross-view / "
            "different-take, not cross-session."
        )
    runtime = [v for v in getattr(scorer, "runtime", {}).values() if v[0] is not None]
    peaks = [v[1] for v in runtime if v[1] is not None]
    report = {
        "schema_version": RUN_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evidence_class": evidence_class,
        "system": dict(system) | {"recipe_model_ids": recipes},
        "inputs": {
            "registry_fingerprint": registry.fingerprint,
            "split_fingerprint": split["fingerprint"],
            "split_status": split["status"],
            "trial_manifest_sha256": trial_doc["manifest_sha256"],
            "trial_policy": trial_doc["trial_policy"],
            "split_role": trial_doc["split_role"],
        },
        "code": _git_revision(),
        "counts": {
            "submitted_trials": n,
            "genuine": sum(o["label"] for o in outcomes),
            "impostor": n - sum(o["label"] for o in outcomes),
            "query_subjects": len(sessions),
            "claimed_subjects": len({o["claimed_subject_id"] for o in outcomes}),
            "score_available": sum(o["score_available"] for o in outcomes),
            "quality_evaluable": sum(o["quality_evaluable"] for o in outcomes),
            "operational_decisions": sum(
                o["operational_decision_available"] for o in outcomes
            ),
        },
        "coverage": {
            "score": overall["score_coverage"],
            "quality": sum(o["quality_evaluable"] for o in outcomes) / n if n else None,
            "operational": overall["coverage"],
        },
        "status_reason_counts": dict(sorted(reasons.items())),
        "metrics": {
            "overall": overall,
            "by_probe_view": _group(outcomes, "probe_view", target_fars),
            "by_query_subject": _group(outcomes, "query_subject_id", target_fars),
            "by_query_session": _group(outcomes, "query_session", target_fars),
        },
        "bootstrap": subject_bootstrap(
            [o["label"] for o in outcomes],
            [o["score"] for o in outcomes],
            [o["status"] for o in outcomes],
            [o["query_subject_id"] for o in outcomes],
            bootstrap_repetitions,
            bootstrap_seed,
            target_fars[0],
        )
        if n
        else {"status": "UNDEFINED_METRIC", "reason": "No trials"},
        "runtime": {
            "queries_measured": len(runtime),
            "runtime_sec_median": float(np.median([v[0] for v in runtime]))
            if runtime
            else None,
            "runtime_sec_max": max(v[0] for v in runtime) if runtime else None,
            "peak_cuda_mib_max": max(peaks) if peaks else None,
        },
        "warnings": warnings,
    }
    return report, outcomes


def write_run(out_dir, report, outcomes):
    """New private directory: report.json + restricted per-trial outcomes.jsonl."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    lines = "".join(
        json.dumps(o, sort_keys=True, allow_nan=False) + "\n" for o in outcomes
    )
    report = report | {
        "outcomes_sha256": hashlib.sha256(lines.encode()).hexdigest(),
        "outcomes_file": "outcomes.jsonl (RESTRICTED: per-trial raw scores)",
    }
    write_new_json(out / "report.json", report)
    fd = os.open(out / "outcomes.jsonl", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(lines)
    return report


def read_run(run_dir):
    run = Path(run_dir)
    report = json.loads((run / "report.json").read_text())
    lines = (run / "outcomes.jsonl").read_text()
    if hashlib.sha256(lines.encode()).hexdigest() != report["outcomes_sha256"]:
        raise ValueError("Outcomes do not match their report")
    return report, [json.loads(line) for line in lines.splitlines()]


def compare_runs(run_a, run_b, repetitions=1000, seed=0, target_far=0.05):
    """Paired descriptive comparison of two systems on the identical frozen manifest.

    Refuses different manifests/registries/splits, the same system twice, and
    different pretraining datasets (GPGait++ CASIA-B stays outside the matched GREW
    comparison). Never names a winner.
    """
    (ra, oa), (rb, ob) = run_a, run_b
    if ra["inputs"] != rb["inputs"]:
        raise ValueError("Runs used different registries, splits or trial manifests")
    if ra["system"]["label"] == rb["system"]["label"]:
        raise ValueError("Compare two different systems")
    if ra["system"]["training_dataset"] != rb["system"]["training_dataset"]:
        raise ValueError(
            "Different pretraining datasets are not a matched comparison "
            "(CASIA-B is reported separately from GREW candidates)"
        )
    comparison = paired_comparison(oa, ob, repetitions, seed, target_far)
    return {
        "systems": [ra["system"], rb["system"]],
        "inputs": ra["inputs"],
        "evidence_class": sorted({ra["evidence_class"], rb["evidence_class"]}),
        **comparison,
    }
