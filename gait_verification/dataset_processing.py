"""Phase 2 dataset processing: explicit-scope plan (dry run) and resumable runner.

Registry (selected asset IDs) -> native pose artifact -> temporal quality record ->
per-model query embedding. This module orchestrates the Phase 1A-1D components; it adds
no registry, split, threshold or scoring of its own.

* Scope is explicit: only listed asset IDs are touched; nothing is ever discovered.
* A plan is read-only. It never creates a directory, enrolls, fits or locks anything.
* An artifact is reused only when it is sealed AND its recorded inputs, recipe and
  checksums still match. An untrusted artifact is renamed aside (kept), never overwritten.
* A different checkpoint, pose recipe, window policy or pose variant yields a different
  task ID / store directory, so artifacts of different recipes never mix.
* GPU work is sequential with an explicit cross-process slot bound; models load lazily,
  once per stage, and are released afterwards. No CPU fallback is ever attempted.
* Nothing here says a recording is good science: the pose gate is the PROVISIONAL Phase 1A
  engineering gate and an unusable recording is never an identity mismatch.
"""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Callable
from .capture_registry import ProvenanceError, write_new_json
from .decisions import DECISIONS
from .media import InvalidMedia, probe_video, sha256_file
from .offline_eval import (
    QueryEmbeddingStore,
    RecordIntegrityError,
    _cuda_peak,
    _git_revision,
    canonical,
    digest,
    embed_query,
    seal,
    sealed_ok,
)
from .pipeline import (
    AGGREGATION,
    CacheIntegrityError,
    CachedPoseExtractor,
    check_pose_artifact,
    extractor_config,
    pose_cache_key,
)
from .pose import load_pose
from .pose_quality import WindowQualityPolicy, analyze_pose, assess_window_quality

PLAN_SCHEMA = "dataset-processing-plan-v1"
RUN_SCHEMA = "dataset-processing-run-v1"
QUALITY_SCHEMA = "pose-quality-record-v1"
EVIDENCE = "ENGINEERING_PROCESSING: not biometric performance evidence"
GATE_NOTE = "PROVISIONAL Phase 1A engineering gate; not a validated biometric minimum"
KINDS = ("pose", "quality", "embed")
VARIANTS = ("vidstab-s10", "vidstab-s15")  # RAW is the control and is never derived
# Uncompressed arrays of one person per frame (see build_plan); compressed NPZ is smaller.
POSE_BYTES_PER_FRAME = (
    17 * 2 * 8 + 17 * 8 + 17 + 8 * 3 + 32 + 4 * 8 + (17 * 2 * 8 + 17 * 8 + 17)
)


class GpuBusy(RuntimeError):
    """Every GPU slot is held by another process; waiting would exceed the bound."""


# --------------------------------------------------------------------------- helpers


def _now():
    return datetime.now(timezone.utc).isoformat()


def quarantine(*paths):
    """Rename untrusted artifacts aside, preserving them; returns the new names."""
    existing = [Path(p) for p in paths if Path(p).exists()]
    n = 1
    while any(p.with_name(f"{p.name}.invalid-{n}").exists() for p in existing):
        n += 1
    return [p.rename(p.with_name(f"{p.name}.invalid-{n}")) for p in existing]


def _safe_name(value):
    if not value or value.startswith(".") or "/" in value:
        raise ProvenanceError("Unsafe path component")
    return value


class WorkDir:
    """Private derived-artifact location, always separate from the data root."""

    def __init__(self, root):
        self.root = Path(root)
        self.pose, self.quality = self.root / "pose", self.root / "quality"
        self.embeddings, self.runs = self.root / "embeddings", self.root / "runs"
        self.ledger = Ledger(self.root / "ledger.jsonl")

    def embed_dir(self, model, recipe):
        return self.embeddings / f"{_safe_name(model)}-{recipe}"

    def ensure(self):
        for d in (self.root, self.pose, self.quality, self.embeddings, self.runs):
            d.mkdir(parents=True, exist_ok=True)
            os.chmod(d, 0o700)


class Ledger:
    """Append-only audit log of task events (JSON lines, 0600). Never rewritten."""

    def __init__(self, path):
        self.path = Path(path)

    def append(self, **event):
        import fcntl

        line = canonical({"ts": _now(), **event}) + "\n"
        fd = os.open(
            self.path, os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600
        )
        with os.fdopen(fd, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            if os.fstat(fd).st_nlink != 1:
                raise ValueError("Ledger must not be a hard-linked file")
            os.fchmod(fd, 0o600)
            if (
                os.fstat(fd).st_size
                and os.pread(fd, 1, os.fstat(fd).st_size - 1) != b"\n"
            ):
                f.write(
                    "\n"
                )  # preserve a torn event, but separate the next complete event
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    def read(self):
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text().splitlines():
            try:
                row = json.loads(line)
                if (
                    isinstance(row, dict)
                    and isinstance(row.get("task_id"), str)
                    and row["task_id"]
                    and isinstance(row.get("state"), str)
                ):
                    rows.append(row)
            except ValueError:  # a torn final line from a crash is ignored, not trusted
                continue
        return rows

    def latest(self):
        out = {}
        for row in self.read():
            if row.get("state") in (
                "DONE",
                "SKIPPED_VALID",
                "FAILED_RETRYABLE",
                "FAILED_PERMANENT",
            ):
                out[row["task_id"]] = row
        return out


@contextmanager
def gpu_slot(lock_dir, limit=1):
    """Hold one of `limit` host-wide GPU slots (flock); fail fast instead of oversubscribing."""
    import fcntl

    if not isinstance(limit, int) or limit < 1:
        raise ValueError("GPU slot limit must be a positive integer")
    lock_dir = Path(lock_dir)
    lock_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(lock_dir, 0o700)
    for i in range(limit):
        fd = os.open(lock_dir / f"slot-{i}.lock", os.O_WRONLY | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            continue
        try:
            yield i
        finally:
            os.close(fd)  # closing releases the flock
        return
    raise GpuBusy(f"All {limit} GPU slot(s) are in use")


def default_gpu_lock_dir():
    return Path(tempfile.gettempdir()) / f"gait-v2-gpu-{os.getuid()}"


def cuda_available():
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def release_gpu_memory():
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextmanager
def interrupt_flag():
    """SIGINT/SIGTERM set a flag polled between tasks; state is already saved per task."""
    flag, old = {"stop": False}, {}

    def handler(signum, frame):
        flag["stop"] = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        old[sig] = signal.signal(sig, handler)
    try:
        yield lambda: flag["stop"]
    finally:
        for sig, previous in old.items():
            signal.signal(sig, previous)


def classify(exc):
    """(state, reason): permanent only when retrying the same input cannot help."""
    if isinstance(exc, InvalidMedia):
        return "FAILED_PERMANENT", "INVALID_MEDIA"
    if isinstance(exc, FileNotFoundError):
        return "FAILED_PERMANENT", "SOURCE_MISSING"
    if isinstance(exc, CacheIntegrityError):
        return "FAILED_RETRYABLE", f"CACHE_INTEGRITY:{exc.reason}"
    text = str(exc).lower().replace(" ", "")
    if isinstance(exc, MemoryError) or "outofmemory" in text:
        return "FAILED_RETRYABLE", "CUDA_OR_HOST_OUT_OF_MEMORY"
    return "FAILED_RETRYABLE", f"PROCESSING_ERROR:{type(exc).__name__}"


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class ModelSpec:
    name: str
    encoder_key: str  # pinned base model ID (models.json); QueryEmbeddingStore key
    factory: Callable  # () -> window-aggregating encoder; called lazily, at most once


@dataclass(frozen=True)
class ProcessingConfig:
    pose_backend_id: str  # expected detector identity; the loaded detector must match
    quality_policy: WindowQualityPolicy
    models: tuple = ()
    stages: tuple = ("pose", "quality")

    def __post_init__(self):
        self.quality_policy.validate()
        if not set(self.stages) <= set(KINDS) or not self.stages:
            raise ValueError(f"Stages must be a non-empty subset of {KINDS}")
        if len({m.name for m in self.models}) != len(self.models):
            raise ValueError("Duplicate model")
        if "embed" in self.stages and not self.models:
            raise ValueError("The embed stage needs at least one explicit model")

    def recipe(self, model):
        """Everything that changes an embedding's meaning, besides the pose artifact."""
        return digest(
            dict(
                encoder_key=model.encoder_key,
                window_policy=asdict(self.quality_policy),
                pose_backend_id=self.pose_backend_id,
                extractor=extractor_config(),
                aggregation=AGGREGATION,
            ),
            16,
        )


# --------------------------------------------------------------------------- states
# Each returns (state, reason, record); read-only, never modifies an artifact.


def pose_key(asset, config):
    return pose_cache_key(
        asset.sha256, config.pose_backend_id, asset.variant_id, asset.original_take_id
    )


def pose_state(work, asset, config):
    npz = work.pose / f"{pose_key(asset, config)}.npz"
    side = npz.with_suffix(".json")
    if not npz.exists():
        return (
            ("INVALID", "ORPHAN_SIDECAR", None)
            if side.exists()
            else ("MISSING", None, None)
        )
    try:
        record = check_pose_artifact(
            npz,
            dict(
                source_sha256=asset.sha256,
                backend_id=config.pose_backend_id,
                variant_id=asset.variant_id,
                original_take_id=asset.original_take_id,
                extractor=extractor_config(),
            ),
        )
        return "VALID", None, record
    except CacheIntegrityError as exc:
        return "INVALID", exc.reason, None


def quality_path(work, asset, config):
    k = digest(
        dict(
            pose=pose_key(asset, config),
            policy=asdict(config.quality_policy),
            schema=QUALITY_SCHEMA,
        )
    )
    return work.quality / f"quality-{k}.json"


def _read_sealed(path):
    try:
        record = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None, "UNREADABLE"
    return (record, None) if sealed_ok(record) else (None, "SEAL_MISMATCH")


def quality_state(work, asset, config, pose_record):
    path = quality_path(work, asset, config)
    if not path.exists():
        return "MISSING", None, None
    record, reason = _read_sealed(path)
    if record is None:
        return "INVALID", reason, None
    if record.get("schema_version") != QUALITY_SCHEMA:
        return "INVALID", "SCHEMA", None
    summary_fields = {
        "frames",
        "pts_span_sec",
        "accepted_window_count",
        "rejected_window_count",
        "rejected_window_reasons",
        "full_body_visibility_fraction",
        "valid_ankle_fraction",
        "person_detection_fraction",
        "ambiguous_detection_fraction",
        "tracking_break_count",
        "missing_joint_fraction",
    }
    if (
        not isinstance(record.get("summary"), dict)
        or not summary_fields <= record["summary"].keys()
        or not isinstance(record.get("reasons"), list)
        or record.get("outcome")
        not in (
            "USABLE_BY_PROVISIONAL_GATE",
            "NOT_USABLE_BY_PROVISIONAL_GATE",
        )
    ):
        return "INVALID", "SCHEMA", None
    if (
        record.get("policy") != asdict(config.quality_policy)
        or record.get("source_sha256") != asset.sha256
    ):
        return "INVALID", "STALE_POLICY_OR_SOURCE", None
    if any(
        record.get(k) != v
        for k, v in dict(
            asset_id=asset.asset_id,
            original_take_id=asset.original_take_id,
            variant_id=asset.variant_id,
            pose_key=pose_key(asset, config),
        ).items()
    ):
        return "INVALID", "STALE_LINEAGE", None
    if (
        pose_record is not None
        and record.get("pose_npz_sha256") != pose_record["npz_sha256"]
    ):
        return "INVALID", "STALE_POSE_ARTIFACT", None
    return "VALID", None, record


def embed_store(work, config, model):
    return QueryEmbeddingStore(
        work.embed_dir(model.name, config.recipe(model)), require_integrity=True
    )


def embed_state(work, asset, config, model, pose_record):
    try:
        record = embed_store(work, config, model).get(asset.asset_id, model.encoder_key)
    except RecordIntegrityError:
        return "INVALID", "SEAL_MISMATCH", None
    except (OSError, ValueError, KeyError, TypeError):
        return "INVALID", "UNREADABLE", None
    if record is None:
        return "MISSING", None, None
    if not sealed_ok(record):  # unsealed records are never accepted by processing
        return "INVALID", "SEAL_MISMATCH", None
    if record.get("status") not in ("EMBEDDED", "INSUFFICIENT_GAIT", "INVALID_INPUT"):
        return "INVALID", "SCHEMA", None
    if record["status"] == "EMBEDDED":
        try:
            from .contracts import GaitEmbedding

            GaitEmbedding(
                record["values"],
                record["model_id"],
                asset.original_take_id,
                asset.sha256,
            ).normalized()
        except (KeyError, TypeError, ValueError):
            return "INVALID", "EMBEDDING_VALUES", None
    if record.get("source_sha256") != asset.sha256 or record.get(
        "recipe"
    ) != config.recipe(model):
        return "INVALID", "STALE_RECIPE_OR_SOURCE", None
    if (
        record.get("original_take_id") != asset.original_take_id
        or record.get("variant_id") != asset.variant_id
    ):
        return "INVALID", "STALE_LINEAGE", None
    if (
        pose_record is not None
        and record.get("pose_npz_sha256") != pose_record["npz_sha256"]
    ):
        return "INVALID", "STALE_POSE_ARTIFACT", None
    return "VALID", None, record


def _task(kind, asset, model, task_id, state, reason, ledger, retry_permanent):
    prior = ledger.get(task_id)
    if (
        state in ("MISSING", "INVALID")
        and prior
        and prior["state"] == "FAILED_PERMANENT"
        and not retry_permanent
    ):
        state, reason = "PRIOR_PERMANENT_FAILURE", prior.get("reason")
    return dict(
        task_id=task_id,
        kind=kind,
        asset_id=asset.asset_id,
        model=model.name if model else None,
        needs_cuda=kind in ("pose", "embed"),
        state=state,
        reason=reason,
    )


# --------------------------------------------------------------------------- plan


def source_issue(registry, asset, data_root):
    """Verify selected bytes, every ancestor, and the registered transform chain."""
    for ancestor in registry.ancestry(asset.asset_id):
        try:
            path = registry.path_of(ancestor, data_root)
            if not path.is_file():
                return (
                    "SOURCE_MISSING" if ancestor == asset else "ANCESTOR_SOURCE_MISSING"
                )
            if sha256_file(path) != ancestor.sha256:
                return (
                    "SOURCE_CHECKSUM_MISMATCH"
                    if ancestor == asset
                    else "ANCESTOR_CHECKSUM_MISMATCH"
                )
            if not ancestor.is_original:
                manifest = registry._transformation(ancestor, data_root)
                if (
                    manifest["input"]["sha256"]
                    != registry.asset(ancestor.parent_asset_id).sha256
                    or manifest["output"]["sha256"] != ancestor.sha256
                ):
                    return "TRANSFORMATION_LINEAGE_MISMATCH"
        except ProvenanceError:
            return "SOURCE_PATH_REJECTED"
        except (OSError, ValueError, KeyError, TypeError):
            return "SOURCE_PROVENANCE_INVALID"
    return None


def protect_sources(registry, data_root, output_root):
    """Refuse an artifact directory that could mutate a registered source tree."""
    output_root = Path(output_root).resolve()
    for asset in registry.assets():
        try:
            source = registry.path_of(asset, data_root)
        except ProvenanceError:
            continue  # the source validation reports invalid paths separately
        if source.is_relative_to(output_root):
            raise ValueError("Artifact directory contains registered source files")


def final_test_denied(asset, split=None, authorization=None):
    """True when the asset belongs to the protected locked-final group.

    Uses the supplied split (a subject absent from it is denied) or, without a split,
    the registry's declared role. The CLI offers no authorization, so it is always denied.
    """
    from .splits import LockedTestAccessDenied, require_roles

    if split is None:
        return asset.declared_split_role == "locked_final"
    role = split["assignments"].get(asset.subject_id)
    if role is None:
        return True  # a subject missing from the supplied split is never assumed safe
    if asset.declared_split_role not in (None, role):
        return True  # conflicting registry/split declarations never authorize access
    if role != "locked_final":
        return False
    try:
        require_roles(split, ("locked_final",), authorization)
        return False
    except LockedTestAccessDenied:
        return True


def build_plan(
    registry,
    data_root,
    work_root,
    asset_ids,
    config,
    *,
    measure=False,
    retry_permanent=False,
    split=None,
    authorization=None,
):
    """Deterministic, read-only processing plan for EXPLICITLY selected assets.

    Verifies each source's SHA-256 (a mismatch BLOCKS its tasks) and classifies every
    task as VALID / MISSING / INVALID / PENDING_UPSTREAM / BLOCKED. Creates nothing.
    """
    ids = sorted(set(asset_ids))
    if not ids:
        raise ValueError("Explicit scope required: no assets were selected")
    protect_sources(registry, data_root, work_root)
    work = WorkDir(work_root)
    ledger = work.ledger.latest()
    blocked, tasks, assets = [], [], []
    frames = 0
    for asset_id in ids:
        try:
            asset = registry.asset(asset_id)
        except ProvenanceError:
            blocked.append(dict(asset_id=asset_id, reason="UNKNOWN_ASSET"))
            continue
        if final_test_denied(asset, split, authorization):
            blocked.append(dict(asset_id=asset_id, reason="LOCKED_FINAL_ACCESS_DENIED"))
            continue
        assets.append(asset)
        reason = source_issue(registry, asset, data_root)
        if reason:
            blocked.append(dict(asset_id=asset_id, reason=reason))
        pose, p_reason, p_record = (
            ("BLOCKED", reason, None) if reason else pose_state(work, asset, config)
        )
        if reason is None and measure and pose != "VALID":
            try:
                path = registry.path_of(asset, data_root)
                frames += len(probe_video(path).timestamps)
            except (InvalidMedia, OSError):
                pass  # reported when the task actually runs
        if "pose" in config.stages:
            tasks.append(
                _task(
                    "pose",
                    asset,
                    None,
                    f"pose-{pose_key(asset, config)}",
                    pose,
                    p_reason,
                    ledger,
                    retry_permanent,
                )
            )
        if "quality" in config.stages:
            if pose == "VALID":
                q, q_reason, _ = quality_state(work, asset, config, p_record)
            else:
                q, q_reason = (
                    ("BLOCKED", reason) if reason else ("PENDING_UPSTREAM", None)
                )
            tasks.append(
                _task(
                    "quality",
                    asset,
                    None,
                    quality_path(work, asset, config).stem,
                    q,
                    q_reason,
                    ledger,
                    retry_permanent,
                )
            )
        if "embed" in config.stages:
            for model in sorted(config.models, key=lambda m: m.name):
                if pose == "VALID":
                    e, e_reason, _ = embed_state(work, asset, config, model, p_record)
                else:
                    e, e_reason = (
                        ("BLOCKED", reason) if reason else ("PENDING_UPSTREAM", None)
                    )
                tid = "embed-" + digest(
                    dict(
                        asset=asset.asset_id,
                        sha=asset.sha256,
                        pose=pose_key(asset, config),
                        recipe=config.recipe(model),
                    )
                )
                tasks.append(
                    _task(
                        "embed", asset, model, tid, e, e_reason, ledger, retry_permanent
                    )
                )
    order = {k: i for i, k in enumerate(KINDS)}
    tasks.sort(key=lambda t: (order[t["kind"]], t["asset_id"], t["model"] or ""))
    originals = [a for a in assets if a.is_original]
    pending = [
        t for t in tasks if t["state"] in ("MISSING", "INVALID", "PENDING_UPSTREAM")
    ]
    states = {}
    for t in tasks:
        states[t["state"]] = states.get(t["state"], 0) + 1
    probe = Path(work_root).resolve()
    while not probe.exists():
        probe = probe.parent
    warnings = []
    if (
        any(not a.is_original for a in assets)
        and config.models
        and "embed" in config.stages
    ):
        warnings.append(
            "Derivative assets selected: their embeddings carry a ':variant=' recipe suffix "
            "and cannot match RAW references; the RAW-versus-stabilized policy (OWN-08) is unset."
        )
    plan_body = dict(
        schema_version=PLAN_SCHEMA,
        registry_fingerprint=registry.fingerprint,
        assets=[(a.asset_id, a.sha256) for a in assets],
        unknown_assets=[
            b["asset_id"] for b in blocked if b["reason"] == "UNKNOWN_ASSET"
        ],
        requested_assets=ids,
        split_fingerprint=split["fingerprint"] if split else None,
        pose_backend_id=config.pose_backend_id,
        extractor=extractor_config(),
        quality_policy=asdict(config.quality_policy),
        models=[
            (m.name, m.encoder_key) for m in sorted(config.models, key=lambda m: m.name)
        ],
        stages=[s for s in KINDS if s in config.stages],
    )
    return dict(
        plan_body,
        plan_id=digest(plan_body, 32),
        evidence_class=EVIDENCE,
        dry_run_guarantee="Read-only: no directory, enrollment, split, threshold or model was created or loaded.",
        selection=dict(
            requested=len(ids),
            assets=len(assets),
            originals=len(originals),
            derivatives=len(assets) - len(originals),
            subjects=len({a.subject_id for a in assets}),
            sessions=len({(a.subject_id, a.session_id) for a in assets}),
            views=sorted({a.view for a in assets}),
        ),
        embedding_stores={
            m.name: str(work.embed_dir(m.name, config.recipe(m)).relative_to(work.root))
            for m in sorted(config.models, key=lambda m: m.name)
        },
        tasks=tasks,
        task_states=dict(sorted(states.items())),
        cuda=dict(
            jobs_requiring_cuda=sum(t["needs_cuda"] for t in pending),
            policy="sequential; explicit slot bound; no CPU fallback; models load lazily",
        ),
        blocked=blocked,
        storage=dict(
            disk_free_bytes=shutil.disk_usage(probe).free,
            pose_cache_uncompressed_estimate_bytes=frames * POSE_BYTES_PER_FRAME
            if measure
            else None,
            estimate_basis=(
                "measured frame count x one-person array bytes, for pose tasks not yet valid; "
                "the compressed NPZ is smaller"
                if measure
                else "not measured (use --measure); embedding and quality sizes are not estimated"
            ),
        ),
        warnings=warnings,
        owner_decisions_unset={k: v for k, v in DECISIONS.items()},
        owner_decisions_needed_for_this_operation=[],
        quality_gate_note=GATE_NOTE,
    )


# --------------------------------------------------------------------------- run


class _Fixed:
    """Extractor that returns an already-validated pose sequence."""

    def __init__(self, sequence):
        self.sequence = sequence

    def extract(self, _path):
        return self.sequence


def _usable(report):
    s = report["summary"]
    reasons = list(s["video_rejection_reasons"]) + sorted(s["rejected_window_reasons"])
    return bool(s["usable"]), ([] if s["usable"] else reasons)


def run_processing(
    registry,
    data_root,
    work_root,
    asset_ids,
    config,
    *,
    detector_factory=None,
    cuda_probe=cuda_available,
    gpu_lock_dir=None,
    max_gpu_jobs=1,
    should_stop=lambda: False,
    retry_permanent=False,
    expect_plan_id=None,
    split=None,
    authorization=None,
):
    """Execute the plan for the selected assets; rerunning resumes and is idempotent."""
    plan = build_plan(
        registry,
        data_root,
        work_root,
        asset_ids,
        config,
        retry_permanent=retry_permanent,
        split=split,
        authorization=authorization,
    )
    if expect_plan_id and expect_plan_id != plan["plan_id"]:
        raise ValueError(
            "Plan ID differs from the reviewed dry run; refusing to execute"
        )
    work = WorkDir(work_root)
    work.ensure()
    run_id = f"run-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{os.urandom(4).hex()}"
    started, clock = _now(), time.perf_counter()
    history = work.ledger.read()
    attempts = {}
    for row in history:
        if str(row.get("state", "")).startswith("FAILED"):
            attempts[row["task_id"]] = attempts.get(row["task_id"], 0) + 1
    results, runtime = (
        {},
        dict(gpu_slot=None, pose_backend_id=config.pose_backend_id, models={}),
    )
    interrupted = [False]

    def log(task, state, reason=None, **extra):
        row = dict(
            task_id=task["task_id"],
            kind=task["kind"],
            asset_id=task["asset_id"],
            model=task["model"],
            state=state,
            reason=reason,
            run_id=run_id,
        )
        if state.startswith("FAILED"):
            attempts[task["task_id"]] = attempts.get(task["task_id"], 0) + 1
            row["attempt"] = attempts[task["task_id"]]
        work.ledger.append(**row, **extra)
        results[task["task_id"]] = dict(
            task, final_state=state, final_reason=reason, **extra
        )

    def halt():
        if should_stop():
            interrupted[0] = True
        return interrupted[0]

    def finish(task, fn):
        """Run one task with failure isolation; fn returns (state, reason, extra)."""
        began = time.perf_counter()
        try:
            state, reason, extra = fn()
        except Exception as exc:  # isolated: one bad recording never stops the batch
            state, reason = classify(exc)
            extra = {}
        log(task, state, reason, seconds=round(time.perf_counter() - began, 3), **extra)

    def skip_or_pending(kind, extra_ok=("VALID",)):
        todo = []
        for t in plan["tasks"]:
            if t["kind"] != kind or t["task_id"] in results:
                continue
            if t["state"] in extra_ok:
                log(t, "SKIPPED_VALID")
            elif t["state"] in ("BLOCKED", "PRIOR_PERMANENT_FAILURE"):
                log(t, "BLOCKED", t["reason"])
            else:
                todo.append(t)
        return todo

    def asset_of(t):
        return registry.asset(t["asset_id"])

    def source_path(asset):
        path = registry.path_of(asset, data_root)
        if reason := source_issue(registry, asset, data_root):
            raise ProvenanceError(reason)
        return path

    def block_all(todo, reason):
        for t in todo:
            log(t, "FAILED_RETRYABLE", reason)

    def gpu_stage(todo, body):
        """Run `body(todo)` holding a GPU slot, after a CUDA preflight; else block."""
        if not todo or interrupted[0]:
            return
        if not cuda_probe():
            return block_all(todo, "CUDA_UNAVAILABLE_NO_CPU_FALLBACK")
        try:
            with gpu_slot(gpu_lock_dir or default_gpu_lock_dir(), max_gpu_jobs) as slot:
                runtime["gpu_slot"] = slot
                body(todo)
        except GpuBusy:
            block_all([t for t in todo if t["task_id"] not in results], "GPU_BUSY")
        except Exception as exc:
            _, reason = classify(exc)
            # Stage setup failures are resources/runtime problems, not evidence
            # that an input is permanently invalid (e.g. a missing checkpoint).
            if isinstance(exc, FileNotFoundError):
                reason = "MODEL_RESOURCE_MISSING"
            for t in todo:
                if t["task_id"] not in results:
                    log(t, "FAILED_RETRYABLE", reason)
            release_gpu_memory()

    # ---- pose
    def pose_body(todo):
        if detector_factory is None:
            return block_all(todo, "NO_DETECTOR_FACTORY")
        detector = detector_factory()
        try:
            if detector.backend_id != config.pose_backend_id:
                return block_all(todo, "DETECTOR_DOES_NOT_MATCH_PLAN")
            runtime["pose_backend_id"] = detector.backend_id
            for t in todo:
                if halt():
                    break
                asset = asset_of(t)
                state, reason, _ = pose_state(work, asset, config)
                if state == "VALID":  # produced meanwhile (e.g. by an interrupted run)
                    log(t, "SKIPPED_VALID")
                    continue
                if state == "INVALID":
                    quarantine(
                        *(
                            work.pose / f"{pose_key(asset, config)}{s}"
                            for s in (".npz", ".json")
                        )
                    )
                    work.ledger.append(
                        task_id=t["task_id"],
                        kind="pose",
                        asset_id=asset.asset_id,
                        state="INVALIDATED",
                        reason=reason,
                        run_id=run_id,
                    )

                def work_fn():
                    path = source_path(asset)
                    CachedPoseExtractor(
                        detector,
                        work.pose,
                        asset.original_take_id,
                        asset.variant_id,
                        require_integrity=True,
                    ).extract(path)
                    state, reason, record = pose_state(work, asset, config)
                    if state != "VALID":
                        raise CacheIntegrityError(
                            reason or "WRITTEN_ARTIFACT_NOT_VALID"
                        )
                    return "DONE", None, dict(frames=record["frames"])

                finish(t, work_fn)
        finally:
            detector = None  # drop the reference so memory can be released
            release_gpu_memory()

    todo = skip_or_pending("pose") if "pose" in config.stages else []
    gpu_stage(todo, pose_body)

    def pose_ready(asset):
        state, _, record = pose_state(work, asset, config)
        return record if state == "VALID" else None

    # ---- quality (CPU)
    if "quality" in config.stages:
        for t in skip_or_pending("quality", ("VALID",)):
            if halt():
                break
            asset = asset_of(t)
            record = pose_ready(asset)
            if record is None:
                log(t, "BLOCKED", "POSE_ARTIFACT_NOT_READY")
                continue
            current = quality_state(work, asset, config, record)[0]
            if current == "VALID":  # an upstream rebuild reproduced the same bytes
                log(t, "SKIPPED_VALID")
                continue
            if current == "INVALID":
                quarantine(quality_path(work, asset, config))

            def quality_fn(asset=asset, record=record):
                sequence = load_pose(work.pose / f"{pose_key(asset, config)}.npz")
                report = analyze_pose(sequence, config.quality_policy)
                usable, reasons = _usable(report)
                body = dict(
                    schema_version=QUALITY_SCHEMA,
                    asset_id=asset.asset_id,
                    original_take_id=asset.original_take_id,
                    variant_id=asset.variant_id,
                    source_sha256=asset.sha256,
                    pose_key=pose_key(asset, config),
                    pose_npz_sha256=record["npz_sha256"],
                    policy=asdict(config.quality_policy),
                    gate_note=GATE_NOTE,
                    outcome="USABLE_BY_PROVISIONAL_GATE"
                    if usable
                    else "NOT_USABLE_BY_PROVISIONAL_GATE",
                    reasons=reasons,
                    summary=report["summary"],
                    windows=report["windows"],
                    accepted_segments=report["accepted_segments"],
                    created_at=_now(),
                )
                write_new_json(quality_path(work, asset, config), seal(body))
                return "DONE", None, dict(outcome=body["outcome"])

            finish(t, quality_fn)

    # ---- embeddings, one model at a time
    if "embed" in config.stages:
        for model in sorted(config.models, key=lambda m: m.name):
            todo = skip_or_pending_model(plan, results, log, model)

            def embed_body(todo, model=model):
                encoder = model.factory()
                try:
                    if encoder.metadata.model_id != model.encoder_key:
                        return block_all(todo, "ENCODER_DOES_NOT_MATCH_PLAN")
                    runtime["models"][model.name] = dict(
                        model_id=encoder.metadata.model_id,
                        checkpoint_sha256=encoder.metadata.checkpoint_sha256,
                        recipe=config.recipe(model),
                        store=str(
                            work.embed_dir(
                                model.name, config.recipe(model)
                            ).relative_to(work.root)
                        ),
                    )
                    store = embed_store(work, config, model)
                    for t in todo:
                        if halt():
                            break
                        asset = asset_of(t)
                        record = pose_ready(asset)
                        if record is None:
                            log(t, "BLOCKED", "POSE_ARTIFACT_NOT_READY")
                            continue
                        current = embed_state(work, asset, config, model, record)[0]
                        if current == "VALID":
                            log(t, "SKIPPED_VALID")
                            continue
                        if current == "INVALID":
                            store.quarantine(asset.asset_id, model.encoder_key)

                        def embed_fn(asset=asset, record=record):
                            path = source_path(asset)
                            sequence = load_pose(
                                work.pose / f"{pose_key(asset, config)}.npz"
                            )
                            _cuda_peak(True)
                            began = time.perf_counter()
                            status, reason, embedding, reasons = embed_query(
                                path,
                                _Fixed(sequence),
                                encoder,
                                lambda s: assess_window_quality(
                                    s, config.quality_policy
                                ),
                            )
                            if status == "PROCESSING_ERROR":
                                raise RuntimeError(reason)
                            store.put(
                                seal(
                                    dict(
                                        schema_version="query-embedding-v1",
                                        asset_id=asset.asset_id,
                                        original_take_id=asset.original_take_id,
                                        variant_id=asset.variant_id,
                                        source_sha256=asset.sha256,
                                        encoder_key=model.encoder_key,
                                        model_id=embedding.model_id
                                        if embedding is not None
                                        else None,
                                        status=status,
                                        reason=reason,
                                        quality_reasons=reasons,
                                        values=embedding.normalized().tolist()
                                        if embedding is not None
                                        else None,
                                        runtime_sec=time.perf_counter() - began,
                                        peak_cuda_mib=_cuda_peak(False),
                                        created_at=_now(),
                                        recipe=config.recipe(model),
                                        pose_npz_sha256=record["npz_sha256"],
                                    )
                                )
                            )
                            return "DONE", None, dict(outcome=status)

                        finish(t, embed_fn)
                finally:
                    encoder = None
                    release_gpu_memory()

            gpu_stage(todo, embed_body)

    # ---- close out
    changed = []
    checked_assets = {
        ancestor.asset_id: ancestor
        for a in map(asset_of, plan["tasks"])
        for ancestor in registry.ancestry(a.asset_id)
    }
    for a in checked_assets.values():
        try:
            if sha256_file(registry.path_of(a, data_root)) != a.sha256:
                changed.append(a.asset_id)
        except (OSError, ProvenanceError):
            if not any(b["asset_id"] == a.asset_id for b in plan["blocked"]):
                changed.append(a.asset_id)
    for t in plan["tasks"]:
        if t["task_id"] not in results:
            results[t["task_id"]] = dict(
                t,
                final_state="NOT_RUN",
                final_reason="INTERRUPTED" if interrupted[0] else t["reason"],
            )
    rows = sorted(
        results.values(),
        key=lambda r: (KINDS.index(r["kind"]), r["asset_id"], r["model"] or ""),
    )
    counts = {}
    for r in rows:
        counts[r["final_state"]] = counts.get(r["final_state"], 0) + 1
    failed = sum(
        v for k, v in counts.items() if k.startswith("FAILED") or k == "BLOCKED"
    )
    status = (
        "INTERRUPTED"
        if interrupted[0]
        else "COMPLETE_WITH_FAILURES"
        if failed or plan["blocked"] or changed
        else "COMPLETE"
    )
    report = dict(
        schema_version=RUN_SCHEMA,
        run_id=run_id,
        status=status,
        started_at=started,
        finished_at=_now(),
        wall_seconds=round(time.perf_counter() - clock, 3),
        plan_id=plan["plan_id"],
        registry_fingerprint=registry.fingerprint,
        selection=plan["selection"],
        blocked_inputs=plan["blocked"],
        task_counts=dict(sorted(counts.items())),
        tasks=rows,
        failures=[
            r
            for r in rows
            if r["final_state"].startswith("FAILED") or r["final_state"] == "BLOCKED"
        ],
        originals=dict(
            checked=sum(a.is_original for a in checked_assets.values()),
            assets_checked=len(checked_assets),
            changed_during_processing=sorted(changed),
        ),
        runtime=dict(
            runtime,
            ffmpeg=_ffmpeg_version(),
            python=platform.python_version(),
            torch=getattr(sys.modules.get("torch"), "__version__", None),
            onnxruntime=getattr(sys.modules.get("onnxruntime"), "__version__", None),
            code=_git_revision(),
            peak_memory_note="peak CUDA memory is recorded per embedding only for in-process models; otherwise null",
        ),
        evidence_class=EVIDENCE,
        quality_gate_note=GATE_NOTE,
        restricted=True,
    )
    write_new_json(work.runs / f"{run_id}.json", report)
    return report


def skip_or_pending_model(plan, results, log, model):
    """Embed tasks of one model: log reusable/blocked ones, return those to compute."""
    todo = []
    for t in plan["tasks"]:
        if t["kind"] != "embed" or t["model"] != model.name or t["task_id"] in results:
            continue
        if t["state"] == "VALID":
            log(t, "SKIPPED_VALID")
        elif t["state"] in ("BLOCKED", "PRIOR_PERMANENT_FAILURE"):
            log(t, "BLOCKED", t["reason"])
        else:
            todo.append(t)
    return todo


def _ffmpeg_version():
    try:
        return subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[
            0
        ]
    except (OSError, subprocess.CalledProcessError):
        return None


# --------------------------------------------------------------------------- report


def coverage_report(
    registry, work_root, asset_ids, config, split=None, authorization=None
):
    """Read-only per-recording processing/quality summary for the selected assets.

    The four layers stay separate: media and operator metadata are `validate`'s job;
    here are the pose ENGINEERING gate (provisional) and processing failures. Scientific
    capture suitability is NOT ASSESSED. An unusable recording is not an identity mismatch.
    """
    work = WorkDir(work_root)
    ledger = work.ledger.latest()
    rows, outcomes, reasons, failures = [], {}, {}, []
    denied = []
    for asset_id in sorted(set(asset_ids)):
        asset = registry.asset(asset_id)
        if final_test_denied(asset, split, authorization):
            denied.append(asset_id)
            continue
        p_state, p_reason, p_record = pose_state(work, asset, config)
        row = dict(
            asset_id=asset_id,
            original_take_id=asset.original_take_id,
            variant_id=asset.variant_id,
            view=asset.view,
            pose=dict(state=p_state, reason=p_reason),
            pose_engineering_gate=dict(status="NOT_PROCESSED", provisional=True),
            embeddings={},
            scientific_capture_suitability=dict(
                status="NOT_ASSESSED",
                reason="needs real pilot evidence (register SCI-01/SCI-02)",
            ),
        )
        if p_state == "VALID":
            q_state, q_reason, q = quality_state(work, asset, config, p_record)
            if q_state == "VALID":
                s = q["summary"]
                row["pose_engineering_gate"] = dict(
                    status=q["outcome"],
                    provisional=True,
                    policy_version=q["policy"]["version"],
                    reasons=q["reasons"],
                    frames=s["frames"],
                    pts_span_sec=s["pts_span_sec"],
                    accepted_windows=s["accepted_window_count"],
                    rejected_windows=s["rejected_window_count"],
                    rejected_window_reasons=s["rejected_window_reasons"],
                    full_body_visibility_fraction=s["full_body_visibility_fraction"],
                    valid_ankle_fraction=s["valid_ankle_fraction"],
                    person_detection_fraction=s["person_detection_fraction"],
                    ambiguous_detection_fraction=s["ambiguous_detection_fraction"],
                    tracking_break_count=s["tracking_break_count"],
                    missing_joint_fraction=s["missing_joint_fraction"],
                )
                outcomes[q["outcome"]] = outcomes.get(q["outcome"], 0) + 1
                for r in q["reasons"]:
                    reasons[r] = reasons.get(r, 0) + 1
            else:
                row["pose_engineering_gate"] = dict(
                    status=f"QUALITY_RECORD_{q_state}",
                    reason=q_reason,
                    provisional=True,
                )
            for model in sorted(config.models, key=lambda m: m.name):
                e_state, e_reason, e = embed_state(work, asset, config, model, p_record)
                row["embeddings"][model.name] = dict(
                    state=e_state, reason=e_reason, status=e["status"] if e else None
                )
        rows.append(row)
    for tid, ev in sorted(ledger.items()):
        if ev["state"].startswith("FAILED") and ev["asset_id"] in {
            r["asset_id"] for r in rows
        }:
            failures.append(
                {
                    k: ev.get(k)
                    for k in (
                        "task_id",
                        "kind",
                        "asset_id",
                        "model",
                        "state",
                        "reason",
                        "attempt",
                        "run_id",
                    )
                }
            )
    return dict(
        schema_version="dataset-coverage-report-v1",
        evidence_class=EVIDENCE,
        registry_fingerprint=registry.fingerprint,
        recordings=rows,
        locked_final_access_denied=denied,
        pose_gate_outcomes=dict(sorted(outcomes.items())),
        pose_gate_reason_counts=dict(sorted(reasons.items())),
        processing_failures=failures,
        notes=[
            GATE_NOTE,
            "A recording that fails the gate is unusable for this pipeline, not an identity mismatch and not discarded.",
            "Processing failures (decoder, CUDA, I/O) are listed separately from quality outcomes.",
        ],
        restricted=True,
    )


# --------------------------------------------------------------------------- derivatives


def derive_variants(
    registry, data_root, asset_ids, variants, stabilizer=None, *, split=None
):
    """Create vid.stab derivatives of EXPLICIT original assets; return (results, new_rows).

    Each lands in <data_root>/derived/<variant>/<asset>/ with FFmpeg's transformation
    manifest; the original is only read (its SHA-256 is rechecked afterwards). Reruns reuse
    a derivative whose manifest and bytes verify; a failed attempt is renamed aside. The
    registry is NOT changed: `new_rows` are proposed rows for a new registry file.
    """
    from .stabilization import StabilizationConfig, stabilize

    stabilizer = stabilizer or stabilize
    if not asset_ids or not variants:
        raise ValueError("Explicit assets and variants are required")
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(
            f"Unsupported variant(s) {unknown}; RAW is the control, never derived"
        )
    results, rows = [], []
    base = Path(data_root).resolve()
    for asset_id in sorted(set(asset_ids)):
        for variant in sorted(set(variants)):
            row = dict(asset_id=asset_id, variant=variant)
            results.append(row)
            try:
                asset = registry.asset(asset_id)
                if final_test_denied(registry.original(asset_id), split):
                    raise ProvenanceError("LOCKED_FINAL_ACCESS_DENIED")
                if not asset.is_original:
                    raise ProvenanceError("Only originals are stabilized")
                existing = registry.variant_of(asset.original_take_id, variant)
                if existing:
                    if reason := source_issue(registry, existing, data_root):
                        raise ProvenanceError(reason)
                    row.update(status="ALREADY_REGISTERED")
                    continue
                source = registry.path_of(asset, data_root)
                if not source.is_file() or sha256_file(source) != asset.sha256:
                    raise ProvenanceError("Source missing or checksum mismatch")
                work = base / "derived" / variant / _safe_name(asset_id)
                if not work.resolve().is_relative_to(base):
                    raise ProvenanceError("Derivative path escapes data root")
                protect_sources(registry, data_root, work)
                reused = _verified_manifest(work, asset, variant)
                if not reused:
                    if work.exists():
                        quarantine(work)
                    recording = registry.recording(
                        asset_id, asset.declared_split_role or "unassigned"
                    )
                    stabilizer(
                        source,
                        work,
                        recording,
                        StabilizationConfig(enabled=True, smoothing=int(variant[-2:])),
                    )
                if not _verified_manifest(work, asset, variant):
                    raise ProvenanceError(
                        "Derivative artifact or recipe failed validation"
                    )
                manifest = json.loads((work / "manifest.json").read_text())
                if sha256_file(source) != asset.sha256:
                    raise ProvenanceError("Original changed during processing")
                new = dict(
                    asset_id=f"{asset_id}.{variant}",
                    variant_id=variant,
                    parent_asset_id=asset_id,
                    original_take_id=asset.original_take_id,
                    subject_id=asset.subject_id,
                    sha256=manifest["output"]["sha256"],
                    source_file_uri=str((work / "stabilized.mp4").relative_to(base)),
                    transformation_manifest_ref=str(
                        (work / "manifest.json").relative_to(base)
                    ),
                    transformation_manifest_sha256=sha256_file(work / "manifest.json"),
                )
                rows.append(new)
                row.update(
                    status="REUSED" if reused else "CREATED",
                    derivative_asset_id=new["asset_id"],
                )
            except Exception as exc:
                row.update(status="FAILED", reason=f"{type(exc).__name__}: {exc}")
    return results, rows


def _verified_manifest(work, asset, variant):
    """True when `work` holds a completed derivative of this exact original."""
    try:
        from .stabilization import StabilizationConfig

        manifest = json.loads((work / "manifest.json").read_text())
        return (
            manifest["status"] == "OK"
            and manifest.get("variant_id") == variant
            and manifest.get("config")
            == asdict(StabilizationConfig(enabled=True, smoothing=int(variant[-2:])))
            and manifest["input"]["sha256"] == asset.sha256
            and sha256_file(work / "stabilized.mp4") == manifest["output"]["sha256"]
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False
