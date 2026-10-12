"""Phase 1C CLI: 1:1 claimed-identity verification of (video, claimed id).

Only the claimed subject's references are loaded. No threshold exists, so a usable
enrolled claim returns POLICY_NOT_READY (verdict null); this CLI can never install a
policy or enable TEST_ONLY behavior. `--diagnostics-out` writes the RESTRICTED raw
score for engineering; it is never printed and must not be shared with callers.
"""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gait_verification.enrollments import DirectoryEnrollmentStore
from gait_verification.pipeline import (
    CachedPoseExtractor,
    lineage_from_manifest,
    query_verifier,
)
from gait_verification.registry import (
    MODELS,
    load_encoder,
    load_quality_policy,
    load_rtmpose,
)
from gait_verification.scoring import STRATEGIES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--claimed-id", required=True)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--enrollments", default="data/gait_verification/enrollments")
    parser.add_argument("--rtmpose-dir", default="~/.cache/rtmlib/hub/checkpoints")
    parser.add_argument("--source-root", default=".cache/gait_verification/sources")
    parser.add_argument(
        "--checkpoint-root", default=".cache/gait_verification/checkpoints"
    )
    parser.add_argument("--pose-cache-dir")
    parser.add_argument("--query-take-id", help="defaults to the video SHA-256")
    parser.add_argument(
        "--derivative-manifest",
        help="stabilization manifest.json when the query is a derivative of an original",
    )
    parser.add_argument(
        "--scoring-strategy",
        default=STRATEGIES[0],
        choices=STRATEGIES,
        help="non-baseline strategies are development-only",
    )
    parser.add_argument(
        "--diagnostics-out", help="new RESTRICTED json file (mode 0600)"
    )
    args = parser.parse_args()
    policy = load_quality_policy()
    lineage = (
        lineage_from_manifest(json.loads(Path(args.derivative_manifest).read_text()))
        if args.derivative_manifest
        else ()
    )
    verifier = query_verifier(
        DirectoryEnrollmentStore(args.enrollments),
        load_encoder(args.model, args.source_root, args.checkpoint_root, policy),
        CachedPoseExtractor(
            load_rtmpose(args.rtmpose_dir), args.pose_cache_dir, args.query_take_id
        ),
        policy,
        scoring_strategy=args.scoring_strategy,  # deliberately no decision policy is passed
    )
    result, diagnostics = verifier.verify_detailed(args.video, args.claimed_id, lineage)
    if args.diagnostics_out:
        fd = os.open(args.diagnostics_out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(
                {
                    "restricted": True,
                    "claimed_subject_id": args.claimed_id,
                    **diagnostics,
                },
                f,
                indent=2,
                allow_nan=False,
            )
    print(json.dumps(result.public_dict()))
    return 0 if result.verdict is not None else 2


if __name__ == "__main__":
    sys.exit(main())
