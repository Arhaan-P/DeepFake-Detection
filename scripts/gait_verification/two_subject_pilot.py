"""Two-person, four-view, two-take gait pilot (TWO_SUBJECT_PILOT_DIAGNOSTIC).

  init      write an editable 16-row intake sheet and a pilot config (new files only)
  validate  measure/decode the explicitly listed originals; list missing files/metadata
  register  write the capture-registry-v1 file once every row validates (never overwrites)
  plan      read-only dry run (nothing created, no model loaded)
  augment   S10/S15 and opt-in brightness/contrast/blur/compression derivatives
  process   pose + provisional quality for originals and derivatives; frozen embeddings
            and native E0 features for originals only
  enroll    RAW reference-set originals -> claimed-subject references (--direction)
  trials    deterministic genuine/wrong-claim manifest, cross-view rule (--direction)
  score     claim-only scoring for every configured model (--direction)
  previews  private skeleton overlays for the config's explicit preview assets
  report    intake/augmentation/quality/enrollment/verification report + summary.md
  all       every stage above, with the same checks

Engineering pilot only: two people support no accuracy, error-rate, threshold or model
choice. Successful scores are POLICY_NOT_READY with a null verdict; raw scores stay in
restricted files under the work directory. Rerunning any command resumes from validated
caches. Nothing is discovered from folders: the intake sheet lists every file.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gait_verification.capture_registry import ProvenanceError, write_new_json  # noqa: E402
from gait_verification.pilot_intake import EVIDENCE, intake_template  # noqa: E402
from gait_verification.two_subject_pilot import (  # noqa: E402
    DIRECTIONS,
    MEDIA_KINDS,
    SCHEMA,
    PilotWorkflow,
)

ALL_MODELS = ["e0", "gaitpt", "gpgait", "gpgaitpp", "gpgaitpp_casia"]
RUNTIME = dict(
    rtmpose_dir="~/.cache/rtmlib/hub/checkpoints",
    source_root=".cache/gait_verification/sources",
    checkpoint_root=".cache/gait_verification/checkpoints",
    e0_pose_model=".cache/gait_verification/checkpoints/pose_landmarker_lite.task",
    e0_checkpoint=".cache/gait_verification/checkpoints/e0-shipped-head-v1.pt",
)


def build_workflow(cfg, max_gpu_jobs=1, should_stop=lambda: False):
    """Real dependencies, all loaded lazily: planning and cached reruns load nothing."""
    from gait_verification.registry import load_json, load_rtmpose
    from scripts.gait_verification.evaluate_claims import processing_config

    rt = RUNTIME | cfg.get("runtime", {})
    pretrained = [m for m in cfg["models"] if m != "e0"]
    processing = processing_config(
        argparse.Namespace(
            model=pretrained,
            source_root=rt["source_root"],
            checkpoint_root=rt["checkpoint_root"],
            stages=None,
        )
    )
    extractor = None

    def e0_extract(path):
        nonlocal extractor
        if extractor is None:
            from gait_verification.e0_processing import (
                MediaPipeE0Extractor,
                feature_recipe,
            )

            extractor = MediaPipeE0Extractor(rt["e0_pose_model"], feature_recipe())
        return extractor(path)

    def e0_backend():
        from gait_verification.e0 import FrozenE0

        # The checksum is the registry pin; a path override cannot change the weights.
        pin = load_json("configs/gait_verification/e0.json")["model_only_sha256"]
        return FrozenE0(rt["e0_checkpoint"], pin)

    return PilotWorkflow(
        cfg,
        processing=processing,
        detector_factory=lambda: load_rtmpose(rt["rtmpose_dir"]),
        e0_extractor=e0_extract,
        e0_backend_factory=e0_backend,
        max_gpu_jobs=max_gpu_jobs,
        should_stop=should_stop,
    )


def init(a):
    template = intake_template(tuple(a.subjects))
    config = dict(
        schema_version=SCHEMA,
        evidence_class=EVIDENCE,
        media_kind=a.media_kind,
        data_root=a.data_root,
        work_dir=a.work_dir,
        intake=a.intake_out,
        registry=a.registry,
        subjects=list(a.subjects),
        asset_ids=[r["capture"]["asset_id"] for r in template["rows"]],
        models=ALL_MODELS,
        variants=["vidstab-s10", "vidstab-s15"],
        augmentations=[],
        set_swap=False,
        preview_assets=[],
        preview_flagged_frames=2,
        runtime=RUNTIME,
    )
    write_new_json(a.intake_out, template)
    write_new_json(a.config_out, config)
    return dict(
        status="CREATED",
        intake=a.intake_out,
        config=a.config_out,
        rows=len(template["rows"]),
        next_step="Fill every null field from real records, then run `validate`",
    )


def summarize(command, result):
    """Console view: statuses and counts only (no scores, labels or embeddings)."""
    if command == "trials":
        return dict(
            direction=result["direction"],
            trials=len(result["trials"]),
            genuine=sum(t["label_same_identity"] for t in result["trials"]),
            contract_sha256=result["contract_sha256"],
        )
    if command == "score":
        return {
            m: s["counts"] | dict(statuses=s["status_reason_counts"])
            for m, s in result.items()
        }
    if command == "report" or (command == "all" and "report" in result):
        report = result.get("report", result)
        return dict(
            status=result.get("status", "REPORTED"),
            report_dir=report["report_dir"],
            independent_originals=report["independent_originals"],
            derivatives=report["derivatives"],
            scored={
                m: s["counts"]["score_available"]
                for m, s in report["verification"].items()
            },
        )
    intake = result.get("intake") or result
    if "missing_originals" in intake:
        return dict(
            status=result["status"],
            missing_originals=intake["missing_originals"],
            missing_matrix_rows=intake["missing_matrix_rows"],
            duplicate_sources=intake["duplicate_sources"],
            errors=intake["errors"],
            row_issues={
                r["asset_id"]: r["issues"] for r in intake["recordings"] if r["issues"]
            },
        )
    return {
        k: v
        for k, v in result.items()
        if k
        in (
            "status",
            "plan_id",
            "independent_originals",
            "planned_derivatives",
            "original_processing_task_states",
            "blocked",
            "derivatives",
            "task_counts",
            "failures",
            "cache_reused",
            "run_ids",
            "rows",
        )
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    i = sub.add_parser("init")
    i.add_argument("--subjects", nargs=2, required=True, help="two pseudonymous IDs")
    i.add_argument("--media-kind", choices=MEDIA_KINDS, required=True)
    i.add_argument("--data-root", required=True)
    i.add_argument(
        "--work-dir", required=True, help="private artifacts, outside originals/"
    )
    i.add_argument(
        "--registry", required=True, help="capture registry path to create later"
    )
    i.add_argument("--intake-out", required=True)
    i.add_argument("--config-out", required=True)
    for name in (
        "validate",
        "register",
        "plan",
        "augment",
        "process",
        "enroll",
        "trials",
        "score",
        "previews",
        "report",
        "all",
    ):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--out", help="new JSON file with the full private result")
        p.add_argument("--max-gpu-jobs", type=int, default=1)
        if name in ("enroll", "trials", "score"):
            p.add_argument("--direction", choices=DIRECTIONS, default=DIRECTIONS[0])
        if name == "all":
            p.add_argument(
                "--set-swap",
                action="store_true",
                help="also run the separately labelled Set 2 -> Set 1 experiment",
            )
    a = parser.parse_args()
    try:
        if a.command == "init":
            result = init(a)
        else:
            from gait_verification.dataset_processing import interrupt_flag

            cfg = json.loads(Path(a.config).read_text())
            with interrupt_flag() as stop:
                workflow = build_workflow(cfg, a.max_gpu_jobs, stop)
                method = getattr(workflow, a.command)
                if a.command in ("enroll", "trials", "score"):
                    result = method(a.direction)
                elif a.command == "all":
                    result = method(True if a.set_swap else None)
                else:
                    result = method()
            if a.out:
                write_new_json(a.out, result)
        print(json.dumps(summarize(a.command, result), indent=2, default=str))
    except (OSError, ValueError, KeyError, ProvenanceError) as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        return 3
    status = result.get("status") if isinstance(result, dict) else None
    return (
        0
        if status in (None, "CREATED", "VALID", "REGISTERED", "PLANNED", "COMPLETE")
        else 2
    )


if __name__ == "__main__":
    sys.exit(main())
