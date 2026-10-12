"""Phase 1D restricted offline verification evaluation (research-only).

  validate  capture registry structure (+ optional file/transform verification)
  split     DRAFT subject split from EXPLICIT counts and seed (nothing defaulted)
  lock      lock a DRAFT split (later changes need a recorded protocol revision)
  trials    frozen genuine + sampled impostor trial manifest for ONE role
  embed     embed the trial manifest's registered query assets (frozen CUDA encoder)
  run       claim-only scoring of every trial -> report.json + restricted outcomes
  compare   paired descriptive comparison of two runs on the identical manifest

Development and calibration roles only: there is deliberately no locked-final path
here. No threshold, calibration, model selection or public verdict is produced.
Every output is a NEW private file; nothing is overwritten.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gait_verification.capture_registry import load_registry, write_new_json  # noqa: E402
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
    s.add_argument("--seed", type=int, required=True)
    for role in ("development", "calibration", "locked-final"):
        s.add_argument(f"--{role}", type=int, required=True)
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
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--impostors-per-query", type=int, required=True)
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
    r.add_argument("--enrollments", required=True)
    r.add_argument("--query-store", required=True)
    r.add_argument("--model", required=True, choices=MODELS)
    r.add_argument("--evidence-class", required=True, choices=EVIDENCE_CLASSES[:-1])
    r.add_argument("--target-far", type=float, nargs="+", default=[0.05])
    r.add_argument("--bootstrap", type=int, default=1000)
    r.add_argument("--bootstrap-seed", type=int, default=0)
    r.add_argument("--out", required=True, help="new private run directory")
    c = sub.add_parser("compare")
    c.add_argument("--run", action="append", required=True)
    c.add_argument("--bootstrap", type=int, default=1000)
    c.add_argument("--bootstrap-seed", type=int, default=0)
    c.add_argument("--out", required=True)
    a = p.parse_args()

    try:
        if a.command == "validate":
            registry = load_registry(a.registry)
            out = {"summary": registry.summary()}
            if a.data_root:
                out["verification"] = registry.verify_files(a.data_root, a.measure)
            write_new_json(a.out, out)
            ok = out.get("verification", {"ok": True})["ok"]
            print(json.dumps({"ok": ok, **registry.summary()}, indent=2))
            return 0 if ok else 2
        if a.command == "split":
            registry = load_registry(a.registry)
            counts = {
                "development": a.development,
                "calibration": a.calibration,
                "locked_final": a.locked_final,
            }
            policy = SplitPolicy(
                a.policy_version, a.seed, counts, tuple(a.pin_development)
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
            policy = TrialPolicy(
                a.policy_version, a.seed, a.impostors_per_query, a.query_variant
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
            entry = model_entry(a.model)
            scorer = EmbeddingTrialScorer(
                registry,
                split,
                DirectoryEnrollmentStore(a.enrollments),
                QueryEmbeddingStore(a.query_store),
                entry["model_id"],  # deliberately no decision policy is passed
            )
            system = {
                "label": a.model,
                "encoder_key": entry["model_id"],
                "training_dataset": entry["training_dataset"],
                "readiness": entry["readiness"],
            }
            report, outcomes = run_evaluation(
                read(a.trials),
                registry,
                split,
                scorer,
                system,
                a.evidence_class,
                target_fars=tuple(a.target_far),
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
            if len(a.run) != 2:
                raise ValueError("compare needs exactly two --run directories")
            result = compare_runs(
                read_run(a.run[0]), read_run(a.run[1]), a.bootstrap, a.bootstrap_seed
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


if __name__ == "__main__":
    sys.exit(main())
