"""Phase 1C enrollment: independent ORIGINAL recordings -> per-original reference embeddings.

  check   pose + window quality only (no gait model): is this video enrollable?
  add     store one reference per --models recipe for ONE original take
  show    non-biometric summary of ONE subject's references (never lists other subjects)
  revoke  exclude one reference from lookups (record kept)

Run `add` once per independent original (different take/session/view). Re-adding the
same take, RAW/S10/S15 derivatives of it, or another subject's bytes is refused.
Real participants require a real --consent-ref; never enroll historical clips as if
they consented to the new study (use a scratch --enrollments for engineering smoke).
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gait_verification.enrollments import DirectoryEnrollmentStore
from gait_verification.pipeline import (
    CachedPoseExtractor,
    EnrollmentRefused,
    check_eligibility,
    enroll_original,
)
from gait_verification.registry import (
    MODELS,
    load_encoder,
    load_quality_policy,
    load_rtmpose,
)


def common(p):
    p.add_argument("--rtmpose-dir", default="~/.cache/rtmlib/hub/checkpoints")
    p.add_argument("--pose-cache-dir", help="immutable native pose cache directory")
    p.add_argument("--video", required=True)
    p.add_argument("--take-id", required=True, help="stable original_take_id")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enrollments", default="data/gait_verification/enrollments")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check")
    common(check)
    add = sub.add_parser("add")
    common(add)
    add.add_argument("--subject-id", required=True)
    add.add_argument("--models", nargs="+", choices=MODELS, required=True)
    add.add_argument("--session-id", required=True)
    add.add_argument("--view", required=True)
    add.add_argument(
        "--recorded-at", required=True, help="ISO-8601 actual recording time"
    )
    add.add_argument("--consent-ref", required=True)
    add.add_argument("--split-role", default="development")
    add.add_argument("--source-root", default=".cache/gait_verification/sources")
    add.add_argument(
        "--checkpoint-root", default=".cache/gait_verification/checkpoints"
    )
    show = sub.add_parser("show")
    show.add_argument("--subject-id", required=True)
    revoke = sub.add_parser("revoke")
    revoke.add_argument("--subject-id", required=True)
    revoke.add_argument("--enrollment-id", required=True)
    revoke.add_argument("--reason", required=True)
    args = parser.parse_args()
    store = DirectoryEnrollmentStore(args.enrollments)
    policy = load_quality_policy()
    try:
        if args.command == "show":
            rows = [
                dict(
                    enrollment_id=t.enrollment_id,
                    original_take_id=t.recording.original_take_id,
                    session_id=t.recording.session_id,
                    view=t.recording.view,
                    recorded_at=t.recording.recorded_at,
                    model_id=t.embedding.model_id,
                    dimension=len(t.embedding.values),
                )
                for t in store.get_claim(args.subject_id)
            ]
            print(
                json.dumps(
                    {"subject_id": args.subject_id, "references": rows}, indent=2
                )
            )
        elif args.command == "revoke":
            store.revoke_reference(args.subject_id, args.enrollment_id, args.reason)
            print(json.dumps({"revoked": args.enrollment_id}))
        else:
            extractor = CachedPoseExtractor(
                load_rtmpose(args.rtmpose_dir), args.pose_cache_dir, args.take_id
            )
            if args.command == "check":
                report = check_eligibility(extractor.extract(args.video), policy)
                print(
                    json.dumps(
                        {"eligible": True, "quality": report["summary"]}, indent=2
                    )
                )
            else:
                encoders = [
                    load_encoder(m, args.source_root, args.checkpoint_root, policy)
                    for m in args.models
                ]
                made = enroll_original(
                    args.video,
                    extractor,
                    encoders,
                    store,
                    policy,
                    subject_id=args.subject_id,
                    take_id=args.take_id,
                    session_id=args.session_id,
                    view=args.view,
                    recorded_at=args.recorded_at,
                    consent_ref=args.consent_ref,
                    split_role=args.split_role,
                )
                print(
                    json.dumps(
                        {
                            "enrolled": [
                                {
                                    "enrollment_id": t.enrollment_id,
                                    "model_id": t.embedding.model_id,
                                }
                                for t in made
                            ]
                        },
                        indent=2,
                    )
                )
    except EnrollmentRefused as exc:
        print(json.dumps({"eligible": False, "refused": str(exc)}))
        return 2
    except (OSError, ValueError) as exc:  # includes InvalidMedia, FileExistsError
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
