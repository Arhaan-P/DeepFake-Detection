"""TWO_SUBJECT_PILOT_DIAGNOSTIC orchestration over the accepted Phase 2C components.

Engineering pipeline acceptance only: intake -> registry -> controlled derivatives ->
pose/quality/embeddings -> independent-original enrollment -> claim-only scoring ->
private reports. No component is reimplemented: lineage, ownership, split, cache and
scoring checks are the existing ones. Two people cannot validate a biometric system;
successful scores stay POLICY_NOT_READY with a null verdict and raw scores stay in
restricted files. Nothing here fits, thresholds, ranks or selects anything.
"""

from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from .capture_registry import (
    CaptureRegistry,
    canonical_sha256,
    load_registry,
    write_new_json,
)
from .contracts import Status, VerificationResult
from .dataset_processing import (
    VARIANTS,
    GpuBusy,
    WorkDir,
    build_plan,
    coverage_report,
    cuda_available,
    default_gpu_lock_dir,
    derive_variants,
    embed_store,
    gpu_slot,
    pose_key,
    pose_state,
    protect_sources,
    release_gpu_memory,
    run_processing,
    source_issue,
)
from .enrollments import DirectoryEnrollmentStore
from .offline_eval import EmbeddingTrialScorer, write_run
from .pilot_intake import EVIDENCE, NOTICE, VIEWS, validate_intake
from .pilot_protocol import (
    claim_request,
    pilot_split,
    pilot_trials,
    validate_pilot_trials,
)
from .pipeline import CachedPoseExtractor, enroll_registered
from .stabilization import AugmentationConfig

SCHEMA = "two-subject-pilot-workflow-v1"
MEDIA_KINDS = ("REAL_PARTICIPANT_RECORDINGS", "SYNTHETIC_TEST_MEDIA")
DIRECTIONS = ("set1-to-set2", "set2-to-set1")
WAITING = "WAITING_FOR_RECORDINGS_OR_METADATA"
# Measured from the existing quality record; +1 = higher is better for pose engineering.
MEASURES = dict(
    accepted_window_count=1,
    full_body_visibility_fraction=1,
    valid_ankle_fraction=1,
    person_detection_fraction=1,
    ambiguous_detection_fraction=-1,
    tracking_break_count=-1,
    missing_joint_fraction=-1,
)
MATCHED_VARIANT = dict(
    status="RESTRICTED",
    reason="RECIPE_MISMATCH",
    explanation=(
        "Enrollment references are RAW independent originals only (Phase 1C rule). "
        "Derivative queries carry a ':variant=' recipe and cannot match them; no "
        "variant-aware reference store is installed. Derivatives receive pose/quality "
        "analysis only, and RAW/derivative scores are never mixed."
    ),
)
PROHIBITED_USE = (
    "Exploratory two-person engineering diagnostic. Not a scientific partition, not "
    "locked-final evidence, not a model comparison, and not an operating point."
)
REMAINING = (
    "Consent-backed approved pilot/SOP review (EXT-01..05, OWN-09/10)",
    "Approved ~40-person cohort with real sessions (EXT-06/07)",
    "Owner split/trial/FAR decisions (OWN-01..07) and RAW-vs-derivative policy (OWN-08)",
    "Validated pose-quality gate (SCI-01/02) and stabilization evidence (SCI-03)",
    "Development-only calibration/threshold and model selection (SCI-04..08)",
    "Licence, retention, encryption and consent-mapping custody (SEC-01..05)",
)


def _now_id():
    return f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}-{os.urandom(3).hex()}"


def _sets(direction):
    if direction not in DIRECTIONS:
        raise ValueError("Unknown pilot direction")
    return (1, 2) if direction == DIRECTIONS[0] else (2, 1)


def _strip_measured(rows):
    measured = (
        "sha256",
        "duration_sec",
        "resolution",
        "nominal_fps",
        "actual_frame_timestamps_ref",
    )
    return [
        dict(r, capture={k: v for k, v in r["capture"].items() if k not in measured})
        for r in rows
    ]


class _CacheOnlyDetector:
    """Enrollment reuses processing's sealed pose; it never runs a second detector."""

    layout = "coco17"

    def __init__(self, backend_id):
        self.backend_id = backend_id

    def __call__(self, frame):
        raise RuntimeError("POSE_NOT_PROCESSED: run `process` first")


class PilotWorkflow:
    """Config-driven stages; each is resumable and re-checks sources before access.

    Injected dependencies keep the library free of model loading: `processing` is
    the existing ProcessingConfig (pretrained ModelSpecs), `detector_factory` loads
    RTMPose, `e0_extractor`/`e0_backend_factory` are the native MediaPipe extractor
    and frozen E0 head. The CLI supplies the real ones; tests inject fixtures.
    """

    def __init__(
        self,
        cfg,
        *,
        processing,
        detector_factory=None,
        cuda_probe=cuda_available,
        e0_extractor=None,
        e0_backend_factory=None,
        gpu_lock_dir=None,
        max_gpu_jobs=1,
        should_stop=lambda: False,
    ):
        if cfg.get("schema_version") != SCHEMA or cfg.get("evidence_class") != EVIDENCE:
            raise ValueError("Unsupported pilot configuration")
        if cfg.get("media_kind") not in MEDIA_KINDS:
            raise ValueError(f"media_kind must be one of {MEDIA_KINDS}")
        models = list(cfg.get("models", []))
        pretrained = {m.name for m in processing.models}
        if len(set(models)) != len(models) or set(models) - {"e0"} != pretrained:
            raise ValueError(
                "Configured models must equal the processing models (+ optional e0)"
            )
        if "e0" in models and (e0_extractor is None or e0_backend_factory is None):
            raise ValueError("E0 needs its native extractor and frozen backend")
        if set(cfg.get("variants", [])) - set(VARIANTS):
            raise ValueError(
                f"Stabilization variants are {VARIANTS}; RAW is never derived"
            )
        self.cfg, self.processing = cfg, processing
        self.recipes = {}
        for item in cfg.get("augmentations", []):
            config = AugmentationConfig(item["kind"], item["value"])
            self.recipes[config.variant_id] = config
        self.data, self.work = Path(cfg["data_root"]), Path(cfg["work_dir"])
        self.registry_path = Path(cfg["registry"])
        self.measured_path = self.registry_path.with_name(
            self.registry_path.name + ".intake-measured.json"
        )
        self.detector_factory, self.cuda_probe = detector_factory, cuda_probe
        self.e0_extractor, self.e0_backend_factory = e0_extractor, e0_backend_factory
        self.gpu_lock_dir = gpu_lock_dir or default_gpu_lock_dir()
        self.max_gpu_jobs, self.should_stop = max_gpu_jobs, should_stop

    # ------------------------------------------------------------- bookkeeping
    @property
    def models(self):
        return tuple(sorted(self.cfg["models"]))

    def _intake_doc(self):
        return json.loads(Path(self.cfg["intake"]).read_text())

    def _save(self, stage, payload):
        folder = self.work / "pilot" / stage
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        write_new_json(folder / f"{_now_id()}.json", payload)
        return payload

    def _latest(self, stage):
        folder = self.work / "pilot" / stage
        files = sorted(folder.glob("*.json")) if folder.is_dir() else []
        return json.loads(files[-1].read_text()) if files else None

    def _intake(self):
        """Explicit intake only; the configured asset IDs must be its 16 rows."""
        doc = self._intake_doc()
        ids = sorted(r["capture"].get("asset_id") for r in doc.get("rows", []))
        if sorted(self.cfg["asset_ids"]) != ids or sorted(
            self.cfg["subjects"]
        ) != sorted(doc["subjects"]):
            raise ValueError(
                "Configured subjects/asset IDs must exactly match the intake sheet"
            )
        report, registry, measured = validate_intake(doc, self.data)
        if (
            registry is not None
            and self.cfg["media_kind"] == "REAL_PARTICIPANT_RECORDINGS"
        ):
            if any(
                "SYNTHETIC" in str(a.consent_ref) + str(a.session_id)
                for a in registry.assets()
            ):
                raise ValueError(
                    "Synthetic-labelled metadata declared as real recordings"
                )
        return report | dict(media_kind=self.cfg["media_kind"]), registry, measured

    def _registered(self):
        """Load the registered pilot and re-check every original before any access."""
        if not self.registry_path.exists():
            raise ValueError("Pilot not registered: run `validate`/`register` first")
        registry = load_registry(self.registry_path)
        sheet = json.loads(self.measured_path.read_text())
        if _strip_measured(sheet["rows"]) != _strip_measured(
            self._intake_doc()["rows"]
        ):
            raise ValueError(
                "Intake changed after registration; register into a new registry path"
            )
        protect_sources(registry, self.data, self.work)
        issues = {
            a.asset_id: r
            for a in registry.originals()
            if (r := source_issue(registry, a, self.data))
        }
        if issues:
            raise ValueError(f"Registered originals failed source checks: {issues}")
        return registry, sheet, pilot_split(registry)

    def _derived(self, registry, split, create):
        """(full registry, results): derivatives verified by manifest + bytes."""
        variants = sorted(self.cfg.get("variants", [])) + sorted(self.recipes)
        if not variants:
            return registry, []

        def refuse(*_):
            raise RuntimeError("DERIVATIVE_NOT_CREATED: run `augment` first")

        results, rows = derive_variants(
            registry,
            self.data,
            [a.asset_id for a in registry.originals()],
            variants,
            None if create else refuse,
            split=split,
            recipes=self.recipes,
        )
        full = CaptureRegistry(
            dict(
                schema_version="capture-registry-v1",
                assets=registry.document["assets"] + rows,
            )
        )
        return full, results

    def _originals(self, registry, sheet, number=None):
        ids = {r["capture"]["asset_id"]: r["set_number"] for r in sheet["rows"]}
        return sorted(
            a.asset_id
            for a in registry.originals()
            if number is None or ids[a.asset_id] == number
        )

    # ------------------------------------------------------------------ stages
    def validate(self):
        report, registry, _ = self._intake()
        return report | dict(status="VALID" if registry else WAITING)

    def register(self):
        report, registry, measured = self._intake()
        if registry is None:
            return dict(status=WAITING, intake=report)
        if self.registry_path.exists():
            if load_registry(self.registry_path).document != registry.document:
                raise ValueError(
                    "A different registry exists at this path; never overwritten"
                )
        else:
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            write_new_json(self.measured_path, measured)
            write_new_json(self.registry_path, registry.document)
        self._save("intake", report)
        return dict(
            status="REGISTERED",
            intake=report,
            registry_fingerprint=registry.fingerprint,
        )

    def plan(self):
        """Read-only: no directory, registry, derivative or model is created/loaded."""
        if self.registry_path.exists():
            registry, sheet, _ = self._registered()
            intake = None
        else:
            intake, registry, sheet = self._intake()
            if registry is None:
                return dict(status=WAITING, intake=intake)
            protect_sources(registry, self.data, self.work)
        split = pilot_split(registry)
        processing = replace(self.processing, stages=self._stages())
        plan = build_plan(
            registry,
            self.data,
            self.work,
            self._originals(registry, sheet),
            processing,
            split=split,
        )
        derivatives = sorted(self.cfg.get("variants", [])) + sorted(self.recipes)
        body = dict(
            schema_version="two-subject-pilot-plan-v1",
            evidence_class=EVIDENCE,
            registry_fingerprint=registry.fingerprint,
            processing_plan_id=plan["plan_id"],
            derivative_variants=derivatives,
            augmentation_recipes={k: asdict(v) for k, v in self.recipes.items()},
            models=self.models,
            directions=list(DIRECTIONS),
        )
        return body | dict(
            plan_id=canonical_sha256(body)[:32],
            status="PLANNED",
            notice=NOTICE,
            participants=len(registry.subjects()),
            independent_originals=len(registry.originals()),
            planned_derivatives=len(registry.originals()) * len(derivatives),
            trials_per_direction=len(registry.originals()),
            original_processing_task_states=plan["task_states"],
            blocked=plan["blocked"],
            cuda=plan["cuda"],
            storage=plan["storage"],
            intake_validated_in_memory=intake is not None,
            dry_run_guarantee=plan["dry_run_guarantee"],
        )

    def _stages(self):
        return ("pose", "quality") + (("embed",) if self.processing.models else ())

    def augment(self):
        registry, _, split = self._registered()
        full, results = self._derived(registry, split, create=True)
        rows = []
        for raw in registry.originals():
            path = registry.path_of(raw, self.data)
            rows.append(
                dict(
                    asset_id=raw.asset_id,
                    original_take_id=raw.original_take_id,
                    variant_id="raw",
                    transformation="NONE (reference condition)",
                    input_sha256=raw.sha256,
                    output_sha256=raw.sha256,
                    checksum_after=source_issue(registry, raw, self.data)
                    or "UNCHANGED",
                    storage_bytes=path.stat().st_size,
                    status="PRESERVED",
                )
            )
        for r in results:
            row = dict(
                asset_id=r.get("derivative_asset_id"),
                parent_asset_id=r["asset_id"],
                variant_id=r["variant"],
                status=r["status"],
                reason=r.get("reason"),
            )
            if r.get("derivative_asset_id"):
                child = full.asset(r["derivative_asset_id"])
                manifest = json.loads(
                    (self.data / child.transformation_manifest_ref).read_text()
                )
                row.update(
                    original_take_id=child.original_take_id,
                    transformation=manifest["config"],
                    commands=manifest["commands"],
                    ffmpeg_version=manifest["ffmpeg_version"],
                    input_sha256=manifest["input"]["sha256"],
                    output_sha256=manifest["output"]["sha256"],
                    duration_sec=manifest["output"]["duration_sec"],
                    frame_count=len(manifest["output"]["timestamps"]),
                    source_frame_count=len(manifest["input"]["timestamps"]),
                    timing_transform=manifest["timing_transform"],
                    storage_bytes=full.path_of(child, self.data).stat().st_size,
                    inherited=dict(
                        subject=child.subject_id,
                        session=child.session_id,
                        view=child.view,
                        split_role=child.declared_split_role,
                    ),
                )
            rows.append(row)
        failed = [r for r in results if r["status"] == "FAILED"]
        return self._save(
            "augment",
            dict(
                status="COMPLETE_WITH_FAILURES" if failed else "COMPLETE",
                evidence_class=EVIDENCE,
                independent_originals=len(registry.originals()),
                derivatives=len(full.assets()) - len(registry.originals()),
                note="Derivatives are analysis copies of the same 16 originals, never new takes.",
                assets=rows,
            ),
        )

    def process(self):
        registry, sheet, split = self._registered()
        full, derived = self._derived(registry, split, create=False)
        originals = self._originals(registry, sheet)
        kw = dict(
            detector_factory=self.detector_factory,
            cuda_probe=self.cuda_probe,
            gpu_lock_dir=self.gpu_lock_dir,
            max_gpu_jobs=self.max_gpu_jobs,
            should_stop=self.should_stop,
            split=split,
        )
        runs = [
            run_processing(
                full,
                self.data,
                self.work,
                originals,
                replace(self.processing, stages=self._stages()),
                **kw,
            )
        ]
        children = sorted(a.asset_id for a in full.assets() if not a.is_original)
        if children:
            runs.append(
                run_processing(
                    full,
                    self.data,
                    self.work,
                    children,
                    replace(self.processing, stages=("pose", "quality")),
                    **kw,
                )
            )
        reuse = {
            m.name: sum(
                t["kind"] == "embed"
                and t["model"] == m.name
                and t["final_state"] == "SKIPPED_VALID"
                for t in runs[0]["tasks"]
            )
            for m in self.processing.models
        }
        e0 = None
        if "e0" in self.models:
            from .e0_processing import process_e0

            e0 = process_e0(
                registry,
                split,
                self.data,
                self._e0_store(),
                originals,
                self.e0_extractor,
                self.should_stop,
            )
            reuse["e0"] = sum(r["status"] == "SKIPPED_VALID" for r in e0["results"])
        statuses = [r["status"] for r in runs] + ([e0["status"]] if e0 else [])
        missing = [r for r in derived if r["status"] == "FAILED"]
        return self._save(
            "process",
            dict(
                status="COMPLETE"
                if set(statuses) == {"COMPLETE"} and not missing
                else "COMPLETE_WITH_FAILURES",
                run_ids=[r["run_id"] for r in runs],
                task_counts=[r["task_counts"] for r in runs],
                failures=[f for r in runs for f in r["failures"]],
                originals_changed=[
                    c for r in runs for c in r["originals"]["changed_during_processing"]
                ],
                derivatives_not_available=missing,
                e0=e0,
                cache_reused=reuse,
                runtime=runs[0]["runtime"],
                wall_seconds=[r["wall_seconds"] for r in runs],
            ),
        )

    def _e0_store(self):
        from .e0_processing import E0FeatureStore, feature_recipe

        return E0FeatureStore(self.work / "e0-features", feature_recipe())

    def enroll(self, direction=DIRECTIONS[0]):
        """Enroll the reference set's RAW originals; one store per direction."""
        registry, sheet, split = self._registered()
        ref_set = _sets(direction)[0]
        refs = self._originals(registry, sheet, ref_set)
        work = WorkDir(self.work)
        store = DirectoryEnrollmentStore(self.work / "enrollments" / direction)
        policy, rows = self.processing.quality_policy, []
        for model in sorted(self.processing.models, key=lambda m: m.name):
            results, need = {}, []
            enrolled = {
                t.recording.original_take_id: t.embedding.model_id
                for s in registry.subjects()
                for t in store.get_claim(s)
                if t.embedding.model_id.startswith(model.encoder_key)
            }
            for asset_id in refs:
                asset = registry.asset(asset_id)
                p_state, _, p_record = pose_state(work, asset, self.processing)
                query = (
                    embed_store(work, self.processing, model).get(
                        asset_id, model.encoder_key
                    )
                    if p_state == "VALID"
                    else None
                )
                if query is None:
                    results[asset_id] = dict(
                        status="REFUSED", reason="POSE_OR_EMBEDDING_NOT_PROCESSED"
                    )
                elif query["status"] != "EMBEDDED":
                    results[asset_id] = dict(
                        status="REFUSED", reason=f"QUALITY:{query['reason']}"
                    )
                elif enrolled.get(asset.original_take_id) == query["model_id"]:
                    results[asset_id] = dict(status="ALREADY_ENROLLED")  # no model load
                else:
                    need.append(asset_id)
            if need:
                results |= self._enroll_model(
                    registry, split, need, model, store, policy
                )
            rows += self._enrollment_rows(registry, model.name, results)
        if "e0" in self.models:
            features = self._e0_store()
            results = {}
            for asset_id in refs:
                record = features.get(registry.asset(asset_id))
                results[asset_id] = (
                    dict(status="FEATURE_REFERENCE_READY")
                    if record and record["status"] == "FEATURED"
                    else dict(
                        status="REFUSED",
                        reason=record["reason"]
                        if record
                        else "E0_FEATURE_NOT_PROCESSED",
                    )
                )
            rows += self._enrollment_rows(registry, "e0", results)
        ok = all(r["independent_reference_count"] == len(VIEWS) for r in rows)
        return self._save(
            f"enroll-{direction}",
            dict(
                status="COMPLETE" if ok else "COMPLETE_WITH_FAILURES",
                direction=direction,
                reference_set=ref_set,
                evidence_class=EVIDENCE,
                rows=rows,
                note="One reference per independent RAW original and recipe; no embeddings exposed.",
            ),
        )

    def _enroll_model(self, registry, split, need, model, store, policy):
        def failed(reason):
            return {a: dict(status="FAILED_RETRYABLE", reason=reason) for a in need}

        if not self.cuda_probe():
            return failed("CUDA_UNAVAILABLE_NO_CPU_FALLBACK")
        backend = _CacheOnlyDetector(self.processing.pose_backend_id)
        try:
            with gpu_slot(self.gpu_lock_dir, self.max_gpu_jobs):
                encoder = model.factory()
                try:
                    if encoder.metadata.model_id != model.encoder_key:
                        return failed("ENCODER_DOES_NOT_MATCH_PLAN")
                    out = enroll_registered(
                        registry,
                        split,
                        need,
                        self.data,
                        lambda take, variant: CachedPoseExtractor(
                            backend,
                            self.work / "pose",
                            take,
                            variant,
                            require_integrity=True,
                        ),
                        [encoder],
                        store,
                        policy,
                    )
                finally:
                    encoder = None
                    release_gpu_memory()
        except GpuBusy:
            return failed("GPU_BUSY")
        return {r["asset_id"]: r for r in out}

    @staticmethod
    def _enrollment_rows(registry, model, results):
        rows = []
        for subject in registry.subjects():
            mine = {
                a: r
                for a, r in results.items()
                if registry.asset(a).subject_id == subject
            }
            ready = sorted(
                registry.asset(a).original_take_id
                for a, r in mine.items()
                if r["status"]
                in ("ENROLLED", "ALREADY_ENROLLED", "FEATURE_REFERENCE_READY")
            )
            rows.append(
                dict(
                    claimed_subject_id=subject,
                    model=model,
                    reference_original_take_ids=ready,
                    independent_reference_count=len(ready),
                    reused_without_model_load=sum(
                        r["status"] == "ALREADY_ENROLLED" for r in mine.values()
                    ),
                    failures=[
                        dict(asset_id=a, status=r["status"], reason=r.get("reason"))
                        for a, r in sorted(mine.items())
                        if registry.asset(a).original_take_id not in ready
                    ],
                )
            )
        return rows

    def trials(self, direction=DIRECTIONS[0]):
        """Deterministic claim manifest; an existing file must regenerate identically."""
        registry, sheet, split = self._registered()
        manifest = pilot_trials(registry, split, sheet, direction)
        path = self.work / "pilot" / f"trials-{direction}.json"
        if path.exists():
            return validate_pilot_trials(
                json.loads(path.read_text()), registry, split, sheet
            )
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        write_new_json(path, manifest)
        return manifest

    def _scorer(self, name, registry, split, direction):
        if name == "e0":
            from .e0_scorer import E0TrialScorer

            return E0TrialScorer(
                registry, split, self.data, self._e0_store(), self.e0_backend_factory
            )
        model = next(m for m in self.processing.models if m.name == name)
        return EmbeddingTrialScorer(
            registry,
            split,
            DirectoryEnrollmentStore(self.work / "enrollments" / direction),
            embed_store(WorkDir(self.work), self.processing, model),
            model.encoder_key,
            data_root=self.data,
        )

    def score(self, direction=DIRECTIONS[0]):
        """Claim-only scoring of every trial for every configured model."""
        manifest = self.trials(direction)
        registry, sheet, split = self._registered()
        sets = {r["capture"]["asset_id"]: r["set_number"] for r in sheet["rows"]}
        summaries = {}
        for name in self.models:
            scorer = self._scorer(name, registry, split, direction)
            outcomes = []
            for t in manifest["trials"]:
                status, reason, score, verdict, quality = scorer(claim_request(t))
                status = Status(status)
                VerificationResult(t["claimed_subject_id"], status, verdict)
                if status == Status.OK or verdict is not None:
                    raise ValueError("No operational policy may exist in the pilot")
                if score is not None and (
                    status != Status.POLICY_NOT_READY or quality is not True
                ):
                    raise ValueError(
                        "Scored outcome must be POLICY_NOT_READY and quality-evaluable"
                    )
                outcomes.append(
                    dict(
                        trial_id=t["trial_id"],
                        label=int(t["label_same_identity"]),
                        query_subject_id=t["query_subject_id"],
                        claimed_subject_id=t["claimed_subject_id"],
                        query_asset_id=t["query_asset_id"],
                        probe_view=t["probe_view"],
                        query_set=sets[t["query_asset_id"]],
                        status=status.value,
                        reason=reason,
                        verdict=None,
                        score=score,
                        score_available=score is not None,
                        quality_evaluable=bool(quality),
                    )
                )
            summary = self._summarize(name, scorer, outcomes, direction, manifest)
            out = self.work / "pilot" / "runs" / direction / f"{name}-{_now_id()}"
            out.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            write_run(out, summary, outcomes)
            summaries[name] = summary | dict(run_dir=str(out.relative_to(self.work)))
        return self._save(f"score-{direction}", summaries)

    @staticmethod
    def _summarize(name, scorer, outcomes, direction, manifest):
        def breakdown(key):
            groups = {}
            for o in outcomes:
                g = groups.setdefault(str(o[key]), {})
                k = f"{o['status']}/{o['reason']}"
                g[k] = g.get(k, 0) + 1
            return dict(sorted(groups.items()))

        reasons = {}
        for o in outcomes:
            cls = "genuine" if o["label"] else "wrong_claim"
            reasons.setdefault(
                f"{o['status']}/{o['reason']}", {"genuine": 0, "wrong_claim": 0}
            )[cls] += 1
        runtime = [v[0] for v in scorer.runtime.values() if v[0] is not None]
        n, genuine = len(outcomes), sum(o["label"] for o in outcomes)
        scored = sum(o["score_available"] for o in outcomes)
        return dict(
            schema_version="two-subject-pilot-run-v1",
            evidence_class=EVIDENCE,
            notice=NOTICE,
            model=name,
            direction=direction,
            contract_sha256=manifest["contract_sha256"],
            recipe_model_ids=sorted(scorer.recipes),
            counts=dict(
                submitted_trials=n,
                genuine=genuine,
                wrong_claim=n - genuine,
                score_available=scored,
                unscored_or_abstained=n - scored,
            ),
            status_reason_counts=dict(sorted(reasons.items())),
            by_probe_view=breakdown("probe_view"),
            by_query_set=breakdown("query_set"),
            by_query_subject=breakdown("query_subject_id"),
            outcomes=[{k: v for k, v in o.items() if k != "score"} for o in outcomes],
            runtime=dict(
                queries_measured=len(runtime),
                runtime_sec_max=max(runtime, default=None),
                scope="cached per-query preparation only",
            ),
            public_result="POLICY_NOT_READY with null verdict; raw scores only in restricted outcomes.jsonl",
            prohibited_use=PROHIBITED_USE,
        )

    def previews(self):
        """Private start/middle/end (+ flagged) overlays for EXPLICIT preview assets."""
        from .pilot import write_overlay
        from .pose import load_pose

        registry, _, split = self._registered()
        full, _ = self._derived(registry, split, create=False)
        work, rows = WorkDir(self.work), []
        for asset_id in self.cfg.get("preview_assets", []):
            asset = full.asset(asset_id)
            state, reason, _ = pose_state(work, asset, self.processing)
            out = self.work / "previews" / asset_id
            if state != "VALID":
                rows.append(
                    dict(asset_id=asset_id, status="POSE_NOT_READY", reason=reason)
                )
                continue
            sequence = load_pose(work.pose / f"{pose_key(asset, self.processing)}.npz")
            n = len(sequence.timestamps)
            events = (
                sequence.tracking_events if sequence.tracking_events is not None else []
            )
            flagged = [i for i, e in enumerate(events) if e][
                : self.cfg.get("preview_flagged_frames", 0)
            ]
            chosen = sorted({0, n // 2, n - 1, *flagged})
            if not out.exists():
                out.mkdir(parents=True, mode=0o700)
                write_overlay(full.path_of(asset, self.data), sequence, out, chosen)
            rows.append(
                dict(
                    asset_id=asset_id,
                    status="PRIVATE_PREVIEW",
                    frames=chosen,
                    flagged_frames=flagged,
                    directory=str(out.relative_to(self.work)),
                )
            )
        return self._save(
            "previews",
            dict(
                rows=rows, privacy="Private local frames; may show faces; never upload."
            ),
        )

    # ------------------------------------------------------------------ reports
    def report(self):
        registry, sheet, split = self._registered()
        full, _ = self._derived(registry, split, create=False)
        set_of = {r["capture"]["asset_id"]: r["set_number"] for r in sheet["rows"]}
        coverage = coverage_report(
            full, self.work, [a.asset_id for a in full.assets()], self.processing, split
        )
        raw_rows = {}
        for row in coverage["recordings"]:
            asset = full.asset(row["asset_id"])
            row.update(
                subject_id=asset.subject_id,
                set_number=set_of[full.original(asset.asset_id).asset_id],
            )
            gate = row["pose_engineering_gate"]
            row["recommend_manual_review"] = bool(
                gate.get("status") != "USABLE_BY_PROVISIONAL_GATE"
                or gate.get("ambiguous_detection_fraction")
                or gate.get("tracking_break_count")
            )
            if asset.is_original:
                raw_rows[asset.original_take_id] = gate
        for row in coverage["recordings"]:
            raw = raw_rows.get(row["original_take_id"], {})
            if row["variant_id"] != "raw":
                row["delta_from_raw"] = {
                    k: (
                        None
                        if raw.get(k) is None
                        or row["pose_engineering_gate"].get(k) is None
                        else "UNCHANGED"
                        if row["pose_engineering_gate"][k] == raw[k]
                        else "IMPROVED"
                        if (row["pose_engineering_gate"][k] - raw[k]) * sign > 0
                        else "WORSENED"
                    )
                    for k, sign in MEASURES.items()
                }
        process = self._latest("process") or {}
        verification = {}
        for direction in DIRECTIONS:
            runs = self._latest(f"score-{direction}") or {}
            verification[direction] = {
                k: v | dict(cache_reused=process.get("cache_reused", {}).get(k))
                for k, v in runs.items()
            }
        report = dict(
            schema_version="two-subject-pilot-report-v1",
            evidence_class=EVIDENCE,
            notice=NOTICE,
            media_kind=self.cfg["media_kind"],
            prohibited_use=PROHIBITED_USE,
            participants=len(registry.subjects()),
            independent_originals=len(registry.originals()),
            derivatives=len(full.assets()) - len(registry.originals()),
            intake=self._latest("intake"),
            augmentation=self._latest("augment"),
            processing=process,
            quality=coverage,
            enrollment=(self._latest(f"enroll-{DIRECTIONS[0]}") or {}).get("rows", []),
            enrollment_set_swap=(self._latest(f"enroll-{DIRECTIONS[1]}") or {}).get(
                "rows", []
            ),
            verification=verification[DIRECTIONS[0]],
            verification_set_swap=verification[DIRECTIONS[1]],
            matched_variant_scoring=MATCHED_VARIANT,
            previews=self._latest("previews"),
            scientific_status="SCIENTIFICALLY_UNVALIDATED",
            remaining_before_main_study=REMAINING,
        )
        out = self.work / "pilot" / "reports" / _now_id()
        out.mkdir(parents=True, mode=0o700)
        write_new_json(out / "report.json", report)
        fd = os.open(out / "summary.md", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(summary_markdown(report))
        return report | dict(report_dir=str(out.relative_to(self.work)))

    def all(self, set_swap=None):
        """Every stage with the same checks as the individual commands."""
        registered = self.register()
        if registered["status"] != "REGISTERED":
            return registered
        stages = [self.augment(), self.process()]
        directions = (
            DIRECTIONS
            if (self.cfg.get("set_swap") if set_swap is None else set_swap)
            else DIRECTIONS[:1]
        )
        for direction in directions:
            stages.append(self.enroll(direction))
            self.score(direction)
        stages.append(self.previews())
        report = self.report()
        status = (
            "COMPLETE"
            if all(s.get("status", "COMPLETE") == "COMPLETE" for s in stages)
            else "COMPLETE_WITH_FAILURES"
        )
        return dict(status=status, report=report)


def summary_markdown(report):
    """Plain-language student summary; engineering pass/fail, never identity correctness."""
    rec = report["quality"]["recordings"]
    lines = [
        "# Two-subject gait pilot — engineering summary",
        "",
        f"**{NOTICE}**",
        "",
        f"Media: `{report['media_kind']}`. Participants: {report['participants']}. "
        f"Independent original takes: {report['independent_originals']}. "
        f"Derivative analysis copies: {report['derivatives']} (not extra takes).",
        "",
        "Engineering pass/fail below says whether software could process a recording. "
        "It says nothing about whether any identity claim is correct.",
        "",
        "## Original recordings",
        "",
        "| Asset | Subject | View | Set | Pose gate | Manual review |",
        "|---|---|---|---|---|---|",
    ]
    for r in rec:
        if r["variant_id"] == "raw":
            lines.append(
                f"| {r['asset_id']} | {r['subject_id']} | {r['view']} | {r['set_number']} | "
                f"{r['pose_engineering_gate'].get('status')} | {'yes' if r['recommend_manual_review'] else 'no'} |"
            )
    recapture = [
        r["asset_id"]
        for r in rec
        if r["variant_id"] == "raw"
        and r["pose_engineering_gate"].get("status") != "USABLE_BY_PROVISIONAL_GATE"
    ]
    lines += [
        "",
        "**Need review or recapture:** "
        + (", ".join(recapture) or "none by the provisional gate"),
        "",
    ]
    views = {}
    for r in rec:
        if r["variant_id"] == "raw" and r["recommend_manual_review"]:
            views[r["view"]] = views.get(r["view"], 0) + 1
    lines += [
        "## Camera views with processing flags",
        "",
        ", ".join(f"{v}: {n}" for v, n in sorted(views.items())) or "No view flagged.",
        "",
        "## Transformations versus RAW (measured pose diagnostics only)",
        "",
    ]
    deltas = {}
    for r in rec:
        for k, verdict in r.get("delta_from_raw", {}).items():
            if verdict:
                deltas.setdefault(r["variant_id"], {}).setdefault(verdict, 0)
                deltas[r["variant_id"]][verdict] += 1
    for variant, counts in sorted(deltas.items()):
        lines.append(
            f"- `{variant}`: "
            + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
        )
    lines += ["", "## Enrollment (technical availability)", ""]
    for r in report["enrollment"]:
        lines.append(
            f"- {r['claimed_subject_id']} / {r['model']}: {r['independent_reference_count']} "
            f"of {len(VIEWS)} independent references"
        )
    lines += ["", "## Claim scoring availability (primary Set 1 → Set 2)", ""]
    for name, s in sorted(report["verification"].items()):
        c = s["counts"]
        lines.append(
            f"- {name}: {c['score_available']}/{c['submitted_trials']} claims produced a restricted "
            f"score; abstained/unscored {c['unscored_or_abstained']}; reasons "
            + ", ".join(
                f"{k}={sum(v.values())}" for k, v in s["status_reason_counts"].items()
            )
        )
    lines += [
        "",
        f"Matched-variant scoring: {report['matched_variant_scoring']['status']} "
        f"({report['matched_variant_scoring']['reason']}).",
        "",
        "Every scored claim is `POLICY_NOT_READY` with a null verdict. No threshold, "
        "winner or performance figure exists.",
        "",
        "## Still needed before the 40-person study",
        "",
    ]
    lines += [f"- {item}" for item in report["remaining_before_main_study"]]
    return "\n".join(lines) + "\n"
