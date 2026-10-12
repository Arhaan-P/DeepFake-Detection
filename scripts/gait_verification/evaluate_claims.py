"""Phase 1D restricted offline verification evaluation (research-only).

  validate  capture registry structure (+ optional file/transform verification)
  split     DRAFT subject split from EXPLICIT counts and seed (nothing defaulted)
  lock      lock a DRAFT split (later changes need a recorded protocol revision)
  trials    frozen genuine + sampled impostor trial manifest for ONE role
  embed     embed the trial manifest's registered query assets (frozen CUDA encoder)
  run       claim-only scoring of every trial -> report.json + restricted outcomes
  compare   paired descriptive comparison of two runs on the identical manifest

Phase 2 dataset processing (needs no split, trials or scientific decision):
  validate  also writes per-recording media / metadata layers when --data-root is given
  plan      read-only dry run for EXPLICITLY selected assets (--asset / --assets-file)
  process   execute that plan; rerunning the same command resumes (idempotent)
  derive    vid.stab S10/S15 derivatives of explicit originals + a NEW proposed registry
  report    per-recording pose-gate / failure summary of processed assets

Development and calibration roles only: there is deliberately no locked-final path
here. No threshold, calibration, model selection or public verdict is produced.
Every output is a NEW private file; nothing is overwritten.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gait_verification.capture_registry import (  # noqa: E402
    ProvenanceError,
    load_registry,
    write_new_json,
)
from gait_verification.decisions import load_decisions, require_decision  # noqa: E402
from gait_verification.dataset_ingest import diagnose_rows, ingest_report  # noqa: E402
from gait_verification.enrollments import DirectoryEnrollmentStore  # noqa: E402
from gait_verification.evaluation import (  # noqa: E402
    TrialPolicy,
    generate_verification_trials,
)
from gait_verification.offline_eval import (  # noqa: E402
    EVIDENCE_CLASSES,
    EmbeddingTrialScorer,
    QueryEmbeddingStore,
    compare_runs,
    read_run,
    run_evaluation,
    write_run,
)
from gait_verification.registry import MODELS, model_entry  # noqa: E402
from gait_verification.splits import (  # noqa: E402
    SplitPolicy,
    assign_split,
    lock_split,
    validate_split,
)

ROLES = ("development", "calibration")
DECISIONS_CONFIG = "configs/gait_verification/phase1d_evaluation.json"
STAGES = ("pose", "quality", "embed")


def selection(args):
    """Explicit asset IDs only; there is deliberately no 'all' and no discovery."""
    ids = list(args.asset or [])
    if args.assets_file:
        ids += [
            line.strip()
            for line in Path(args.assets_file).read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    if not ids:
        raise ValueError("Explicit scope required: pass --asset and/or --assets-file")
    return ids


def processing_config(args):
    """Pinned recipe from configs; models are loaded lazily by the runner."""
    from gait_verification.dataset_processing import ModelSpec, ProcessingConfig
    from gait_verification.registry import (
        load_encoder,
        load_json,
        load_quality_policy,
        model_entry,
    )

    rt = load_json("configs/gait_verification/phase1a.json")["rtmpose"]
    policy = load_quality_policy()
    models = tuple(
        ModelSpec(
            m,
            model_entry(m)["model_id"],
            lambda m=m: load_encoder(m, args.source_root, args.checkpoint_root, policy),
        )
        for m in sorted(set(args.model or ()))
    )
    stages = tuple(
        args.stages or (("pose", "quality") + (("embed",) if models else ()))
    )
    backend = f"rtmpose-coco17:{rt['detector']['sha256']}:{rt['pose']['sha256']}"
    return ProcessingConfig(backend, policy, models, stages)


def add_selection(sp):
    sp.add_argument("--registry", required=True)
    sp.add_argument("--data-root", required=True)
    sp.add_argument("--work-dir", required=True, help="private derived-artifact root")
    sp.add_argument("--asset", action="append", help="asset ID (repeatable)")
    sp.add_argument("--assets-file", help="file with one asset ID per line")
    sp.add_argument(
        "--split", help="split manifest; locked-final assets are then denied"
    )
    sp.add_argument("--model", action="append", choices=MODELS)
    sp.add_argument("--stages", nargs="+", choices=STAGES)
    sp.add_argument("--source-root", default=".cache/gait_verification/sources")
    sp.add_argument("--checkpoint-root", default=".cache/gait_verification/checkpoints")


def read(path):
    return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--registry", required=True)
    v.add_argument("--data-root", help="also verify file bytes and transformations")
    v.add_argument("--measure", action="store_true", help="ffprobe timing facts")
    v.add_argument("--out", required=True)
    s = sub.add_parser("split")
    s.add_argument("--registry", required=True)
    s.add_argument("--policy-version", required=True)
    s.add_argument("--seed", type=int, help="OWN-02; else the decisions config")
    for role in ("development", "calibration", "locked-final"):
        s.add_argument(f"--{role}", type=int, help="OWN-01; else the decisions config")
    s.add_argument("--decisions", default=DECISIONS_CONFIG)
    s.add_argument("--pin-development", nargs="*", default=[])
    s.add_argument("--out", required=True)
    lk = sub.add_parser("lock")
    lk.add_argument("--split", required=True)
    lk.add_argument("--locked-by", required=True)
    lk.add_argument("--out", required=True)
    t = sub.add_parser("trials")
    t.add_argument("--registry", required=True)
    t.add_argument("--split", required=True)
    t.add_argument("--role", required=True, choices=ROLES)
    t.add_argument("--policy-version", required=True)
    t.add_argument("--seed", type=int, help="OWN-04; else the decisions config")
    t.add_argument("--impostors-per-query", type=int, help="OWN-03; else the config")
    t.add_argument("--decisions", default=DECISIONS_CONFIG)
    t.add_argument("--query-variant", default="raw")
    t.add_argument("--out", required=True)
    e = sub.add_parser("embed")
    e.add_argument("--registry", required=True)
    e.add_argument("--trials", required=True)
    e.add_argument("--data-root", required=True)
    e.add_argument("--model", required=True, choices=MODELS)
    e.add_argument("--query-store", required=True)
    e.add_argument("--pose-cache-dir")
    e.add_argument("--rtmpose-dir", default="~/.cache/rtmlib/hub/checkpoints")
    e.add_argument("--source-root", default=".cache/gait_verification/sources")
    e.add_argument("--checkpoint-root", default=".cache/gait_verification/checkpoints")
    e.add_argument("--out", required=True, help="new ingestion report")
    r = sub.add_parser("run")
    r.add_argument("--registry", required=True)
    r.add_argument("--split", required=True)
    r.add_argument("--trials", required=True)
    r.add_argument("--enrollments")
    r.add_argument("--query-store")
    r.add_argument("--model", required=True, choices=(*MODELS, "e0"))
    r.add_argument("--data-root", help="required for E0 immutable source verification")
    r.add_argument("--e0-feature-store")
    r.add_argument("--e0-checkpoint")
    r.add_argument("--e0-sha256")
    r.add_argument("--evidence-class", required=True, choices=EVIDENCE_CLASSES[:-1])
    r.add_argument(
        "--target-far", type=float, nargs="+", help="declared FAR(s); OWN-05"
    )
    r.add_argument("--bootstrap", type=int, default=1000)
    r.add_argument("--bootstrap-seed", type=int, default=0)
    r.add_argument("--out", required=True, help="new private run directory")
    c = sub.add_parser("compare")
    c.add_argument("--run", action="append", required=True)
    c.add_argument("--target-far", type=float, help="declared FAR; OWN-05")
    c.add_argument("--bootstrap", type=int, default=1000)
    c.add_argument("--bootstrap-seed", type=int, default=0)
    c.add_argument("--out", required=True)
    c.add_argument(
        "--comparison-policy",
        choices=("matched-pretraining-v1", "historical-e0-versus-grew-architecture-v1"),
        default="matched-pretraining-v1",
    )
    ea = sub.add_parser(
        "e0-audit", help="safe extraction/audit of the exact shipped E0 artifact"
    )
    ea.add_argument(
        "--checkpoint", default="outputs/checkpoints/checkpoint_epoch_46_best.pth"
    )
    ea.add_argument(
        "--extract-to", required=True, help="new private tensor-only checkpoint"
    )
    ea.add_argument("--out", required=True, help="new extraction provenance report")
    ef = sub.add_parser(
        "e0-features",
        help="prepare/resume native features for explicit original assets",
    )
    ef.add_argument("--registry", required=True)
    ef.add_argument("--split", help="protected roles remain inaccessible")
    ef.add_argument("--data-root", required=True)
    ef.add_argument("--asset", action="append")
    ef.add_argument("--assets-file")
    ef.add_argument("--e0-feature-store", required=True)
    ef.add_argument(
        "--pose-model",
        default=".cache/gait_verification/checkpoints/pose_landmarker_lite.task",
    )
    ef.add_argument("--out", required=True)
    ev = sub.add_parser(
        "e0-validate-features",
        help="validate native MediaPipe33 xyz fixtures; never load a pickle",
    )
    ev.add_argument(
        "--native-pose", required=True, help="NPZ with pose and observed arrays"
    )
    ev.add_argument("--out", required=True)
    cr = sub.add_parser(
        "run-common",
        help="score one frozen manifest with explicit per-system artifact locations",
    )
    cr.add_argument("--registry", required=True)
    cr.add_argument("--split", required=True)
    cr.add_argument("--trials", required=True)
    cr.add_argument("--data-root")
    cr.add_argument(
        "--systems",
        required=True,
        help="JSON: systems list with model and per-model store/checkpoint locations",
    )
    cr.add_argument("--evidence-class", required=True, choices=EVIDENCE_CLASSES[:-1])
    cr.add_argument("--target-far", type=float, nargs="+", required=True)
    cr.add_argument(
        "--comparison-policy",
        choices=("matched-pretraining-v1", "historical-e0-versus-grew-architecture-v1"),
        default="matched-pretraining-v1",
    )
    cr.add_argument("--bootstrap", type=int, default=1000)
    cr.add_argument("--bootstrap-seed", type=int, default=0)
    cr.add_argument("--out", required=True)
    pl = sub.add_parser("plan", help="read-only dry run")
    add_selection(pl)
    pl.add_argument(
        "--measure", action="store_true", help="ffprobe frame counts for storage"
    )
    pl.add_argument("--out", help="new plan file (stdout summary only when omitted)")
    pr = sub.add_parser("process", help="execute the plan; rerun to resume")
    add_selection(pr)
    pr.add_argument("--rtmpose-dir", default="~/.cache/rtmlib/hub/checkpoints")
    pr.add_argument("--max-gpu-jobs", type=int, default=1)
    pr.add_argument("--retry-permanent", action="store_true")
    pr.add_argument("--expect-plan-id", help="plan_id of a reviewed dry run")
    dv = sub.add_parser("derive", help="vid.stab derivatives of explicit originals")
    dv.add_argument("--registry", required=True)
    dv.add_argument("--data-root", required=True)
    dv.add_argument("--split", help="split manifest; locked-final assets are denied")
    dv.add_argument("--asset", action="append")
    dv.add_argument("--assets-file")
    dv.add_argument("--variant", action="append", required=True)
    dv.add_argument("--out-registry", required=True, help="NEW registry file")
    dv.add_argument("--out", required=True, help="derivation report")
    rp = sub.add_parser("report", help="processing / quality summary")
    add_selection(rp)
    rp.add_argument("--out", required=True)
    a = p.parse_args()

    try:
        if a.command.startswith("e0-") or a.command == "run-common":
            return phase2b(a)
        if a.command == "validate":
            try:
                registry = load_registry(a.registry)
            except ProvenanceError as exc:
                rows = (
                    read(a.registry).get("assets") if Path(a.registry).is_file() else []
                )
                bad = {
                    "ok": False,
                    "error": str(exc),
                    "diagnostics": diagnose_rows(rows),
                }
                write_new_json(a.out, bad)
                print(json.dumps(bad, indent=2))
                return 2
            out = {"summary": registry.summary()}
            if a.data_root:
                out["ingest"] = ingest_report(registry, a.data_root, a.measure)
            write_new_json(a.out, out)
            ok = out.get("ingest", {"ok": True})["ok"]
            print(json.dumps({"ok": ok, **registry.summary()}, indent=2))
            return 0 if ok else 2
        if a.command in ("plan", "process", "report"):
            return phase2(a)
        if a.command == "derive":
            from gait_verification.dataset_processing import derive_variants

            registry = load_registry(a.registry)
            results, rows = derive_variants(
                registry,
                a.data_root,
                selection(a),
                a.variant,
                split=validate_split(read(a.split)) if a.split else None,
            )
            write_new_json(a.out, {"results": results})
            if rows:  # validated before it is written; the old registry is untouched
                from gait_verification.capture_registry import CaptureRegistry

                document = {
                    **registry.document,
                    "assets": registry.document["assets"] + rows,
                }
                CaptureRegistry(document)  # construction validates lineage
                write_new_json(a.out_registry, document)
            counts = {}
            for r in results:
                counts[r["status"]] = counts.get(r["status"], 0) + 1
            print(json.dumps(counts))
            return 2 if "FAILED" in counts else 0
        if a.command == "split":
            registry = load_registry(a.registry)
            cfg = load_decisions(a.decisions)
            counts = {
                "development": a.development,
                "calibration": a.calibration,
                "locked_final": a.locked_final,
            }
            if any(v is None for v in counts.values()):
                counts = require_decision(cfg["OWN-01"], "OWN-01", "split")
            seed = a.seed if a.seed is not None else cfg["OWN-02"]
            policy = SplitPolicy(
                a.policy_version,
                require_decision(seed, "OWN-02", "split"),
                counts,
                tuple(a.pin_development),
            )
            split = assign_split(registry.subjects(), policy)
            write_new_json(a.out, split)
            print(json.dumps({"status": "DRAFT", "fingerprint": split["fingerprint"]}))
        elif a.command == "lock":
            split = lock_split(read(a.split), a.locked_by)
            write_new_json(a.out, split)
            print(json.dumps({"status": "LOCKED", "fingerprint": split["fingerprint"]}))
        elif a.command == "trials":
            registry, split = load_registry(a.registry), read(a.split)
            cfg = load_decisions(a.decisions)
            seed = a.seed if a.seed is not None else cfg["OWN-04"]
            k = a.impostors_per_query
            policy = TrialPolicy(
                a.policy_version,
                require_decision(seed, "OWN-04", "trials"),
                require_decision(
                    k if k is not None else cfg["OWN-03"], "OWN-03", "trials"
                ),
                a.query_variant,
            )
            doc = generate_verification_trials(registry, split, policy, a.role)
            write_new_json(a.out, doc)
            print(json.dumps(doc["counts"] | {"sha256": doc["manifest_sha256"]}))
        elif a.command == "embed":
            from gait_verification.offline_eval import embed_registered_queries
            from gait_verification.pipeline import CachedPoseExtractor
            from gait_verification.pose_quality import assess_window_quality
            from gait_verification.registry import (
                load_encoder,
                load_quality_policy,
                load_rtmpose,
            )

            registry, doc = load_registry(a.registry), read(a.trials)
            if doc["dataset_fingerprint"] != registry.fingerprint:
                raise ValueError("Trial manifest belongs to another registry")
            if doc["split_role"] not in ROLES:
                raise PermissionError("No locked-final access in Phase 1D")
            policy = load_quality_policy()
            detector = load_rtmpose(a.rtmpose_dir)
            results = embed_registered_queries(
                registry,
                [t["query_asset_id"] for t in doc["trials"]],
                a.data_root,
                lambda take, variant: CachedPoseExtractor(
                    detector, a.pose_cache_dir, take, variant
                ),
                load_encoder(a.model, a.source_root, a.checkpoint_root, policy),
                lambda s: assess_window_quality(s, policy),
                QueryEmbeddingStore(a.query_store),
            )
            write_new_json(a.out, {"model": a.model, "results": results})
            counts = {}
            for row in results:
                counts[row["status"]] = counts.get(row["status"], 0) + 1
            print(json.dumps(counts))
        elif a.command == "run":
            registry, split = load_registry(a.registry), validate_split(read(a.split))
            system, scorer = make_scorer(
                registry,
                split,
                a.data_root,
                {
                    "model": a.model,
                    "enrollments": a.enrollments,
                    "query_store": a.query_store,
                    "feature_store": a.e0_feature_store,
                    "checkpoint": a.e0_checkpoint,
                    "sha256": a.e0_sha256,
                },
            )
            report, outcomes = run_evaluation(
                read(a.trials),
                registry,
                split,
                scorer,
                system,
                a.evidence_class,
                target_fars=tuple(a.target_far or ()),
                bootstrap_repetitions=a.bootstrap,
                bootstrap_seed=a.bootstrap_seed,
            )
            write_run(a.out, report, outcomes)
            print(
                json.dumps(
                    report["counts"] | {"status": report["status_reason_counts"]}
                )
            )
        else:
            if len(a.run) < 2:
                raise ValueError("compare needs at least two --run directories")
            runs = [read_run(path) for path in a.run]
            if len(runs) == 2:
                result = compare_runs(
                    *runs,
                    a.bootstrap,
                    a.bootstrap_seed,
                    a.target_far,
                    comparison_policy=a.comparison_policy,
                )
            else:
                from gait_verification.common_eval import compare_common_runs

                result = compare_common_runs(
                    runs,
                    a.target_far,
                    a.comparison_policy,
                    a.bootstrap,
                    a.bootstrap_seed,
                )
            write_new_json(a.out, result)
            print(
                json.dumps(
                    {"common_scoreable_trials": result["common_scoreable_trials"]}
                )
            )
    except (
        OSError,
        ValueError,
        KeyError,
    ) as exc:  # includes access/provenance refusals
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        return 3
    return 0


def make_scorer(registry, split, data_root, spec):
    """Build one research scorer from explicit artifacts; no model-selection policy."""
    model = spec["model"]
    if model == "e0":
        from gait_verification.e0 import AGGREGATION, SCORE_SEMANTICS, FrozenE0
        from gait_verification.e0_checkpoint import SHIPPED_SHA256, TRAINING_ORIGIN
        from gait_verification.e0_processing import E0FeatureStore, feature_recipe
        from gait_verification.e0_scorer import E0TrialScorer

        if not all(
            (
                data_root,
                spec.get("feature_store"),
                spec.get("checkpoint"),
                spec.get("sha256"),
            )
        ):
            raise ValueError(
                "E0 requires data-root, feature-store, checkpoint and sha256"
            )
        checksum = spec["sha256"]
        from gait_verification.registry import load_json

        if (
            checksum
            != load_json("configs/gait_verification/e0.json")["model_only_sha256"]
        ):
            raise ValueError(
                "E0 run requires the registry-pinned extracted checkpoint hash"
            )
        key = f"e0:{checksum}:native60x78-v1:{AGGREGATION}"
        store = E0FeatureStore(spec["feature_store"], feature_recipe())
        scorer = E0TrialScorer(
            registry,
            split,
            data_root,
            store,
            lambda: FrozenE0(spec["checkpoint"], checksum),
        )
        system = {
            "label": "e0",
            "encoder_key": key,
            "model_family": "e0_tcn",
            "training_dataset": TRAINING_ORIGIN,
            "training_origin": TRAINING_ORIGIN,
            "source_checkpoint_sha256": SHIPPED_SHA256,
            "checkpoint_sha256": checksum,
            "score_semantics": SCORE_SEMANTICS,
            "reference_aggregation": AGGREGATION,
            "feature_recipe_id": store.recipe_id,
            "readiness": "BLOCKED",
            "research_only": True,
        }
        return system, scorer
    if not spec.get("enrollments") or not spec.get("query_store"):
        raise ValueError("Pretrained run requires enrollments and query-store")
    entry = model_entry(model)
    return {
        "label": model,
        "encoder_key": entry["model_id"],
        "training_dataset": entry["training_dataset"],
        "readiness": entry["readiness"],
        "score_semantics": "cosine similarity; higher supports claim; uncalibrated",
    }, EmbeddingTrialScorer(
        registry,
        split,
        DirectoryEnrollmentStore(spec["enrollments"]),
        QueryEmbeddingStore(spec["query_store"]),
        entry["model_id"],
    )


def phase2b(a):
    from gait_verification.e0_processing import (
        E0FeatureStore,
        MediaPipeE0Extractor,
        feature_recipe,
        process_e0,
    )

    if a.command == "e0-audit":
        from gait_verification.e0_checkpoint import SHIPPED_SHA256, extract_e0

        report = extract_e0(a.checkpoint, a.extract_to, SHIPPED_SHA256)
        write_new_json(a.out, report)
        print(json.dumps(report))
        return 0
    if a.command == "e0-validate-features":
        import numpy as np
        from gait_verification.e0 import native_features

        with np.load(a.native_pose, allow_pickle=False) as poses:
            values = native_features(poses["pose"], poses["observed"])
        write_new_json(
            a.out,
            {
                "shape": list(values.shape),
                "finite": bool(np.isfinite(values).all()),
                "recipe": feature_recipe(),
                "evidence": "native feature contract only",
            },
        )
        print(json.dumps({"shape": list(values.shape), "finite": True}))
        return 0
    registry = load_registry(a.registry)
    split = validate_split(read(a.split)) if a.split else None
    if a.command == "e0-features":
        store = E0FeatureStore(a.e0_feature_store, feature_recipe())
        # Lazy construction: cached/denied runs need no pose checkpoint or runtime.
        extractor = None

        def extract(path):
            nonlocal extractor
            if extractor is None:
                extractor = MediaPipeE0Extractor(a.pose_model, store.recipe)
            return extractor(path)

        report = process_e0(registry, split, a.data_root, store, selection(a), extract)
        write_new_json(a.out, report)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "requested_assets": report["requested_assets"],
                }
            )
        )
        return 0 if report["status"] == "COMPLETE" else 2
    from gait_verification.common_eval import compare_common_runs, run_common_evaluation
    from gait_verification.evaluation import validate_trial_manifest

    doc = read(a.trials)
    validate_trial_manifest(
        doc, registry, split
    )  # refusal before scorer/artifact access
    specs = read(a.systems)["systems"]
    systems = [make_scorer(registry, split, a.data_root, spec) for spec in specs]
    runs = run_common_evaluation(
        doc,
        registry,
        split,
        systems,
        a.evidence_class,
        target_fars=tuple(a.target_far),
        bootstrap_repetitions=a.bootstrap,
        bootstrap_seed=a.bootstrap_seed,
    )
    comparison = compare_common_runs(
        runs, a.target_far[0], a.comparison_policy, a.bootstrap, a.bootstrap_seed
    )
    Path(a.out).mkdir(parents=True, exist_ok=False, mode=0o700)
    for report, rows in runs:
        write_run(Path(a.out) / report["system"]["label"], report, rows)
    write_new_json(Path(a.out) / "comparison.json", comparison)
    print(
        json.dumps(
            {
                "systems": len(runs),
                "common_scoreable_trials": comparison["common_scoreable_trials"],
            }
        )
    )
    return 0


def phase2(a):
    from gait_verification.dataset_processing import (
        build_plan,
        coverage_report,
        interrupt_flag,
        run_processing,
    )

    registry, config = load_registry(a.registry), processing_config(a)
    ids = selection(a)
    split = validate_split(read(a.split)) if a.split else None
    if a.command == "plan":
        plan = build_plan(
            registry,
            a.data_root,
            a.work_dir,
            ids,
            config,
            measure=a.measure,
            split=split,
        )
        if a.out:
            write_new_json(a.out, plan)
        print(
            json.dumps(
                {
                    k: plan[k]
                    for k in (
                        "plan_id",
                        "selection",
                        "task_states",
                        "cuda",
                        "blocked",
                        "storage",
                        "warnings",
                    )
                },
                indent=2,
            )
        )
        return 2 if plan["blocked"] else 0
    if a.command == "report":
        write_new_json(a.out, coverage_report(registry, a.work_dir, ids, config, split))
        return 0
    from gait_verification.registry import load_rtmpose

    with interrupt_flag() as stop:
        report = run_processing(
            registry,
            a.data_root,
            a.work_dir,
            ids,
            config,
            detector_factory=lambda: load_rtmpose(a.rtmpose_dir),
            max_gpu_jobs=a.max_gpu_jobs,
            should_stop=stop,
            retry_permanent=a.retry_permanent,
            expect_plan_id=a.expect_plan_id,
            split=split,
        )
    print(
        json.dumps(
            {k: report[k] for k in ("run_id", "status", "task_counts", "originals")},
            indent=2,
        )
    )
    return 0 if report["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    sys.exit(main())
