"""Phase 1C CLI: 1:1 claimed-identity verification of (video, claimed id).

Only the claimed subject's references are loaded. No threshold exists, so a usable
enrolled claim returns POLICY_NOT_READY (verdict null); this CLI can never install a
policy or enable TEST_ONLY behavior. `--diagnostics-out` writes the RESTRICTED raw
score for engineering; it is never printed and must not be shared with callers.

Managed study (`--capture-registry`): the video is identified by its actual bytes in
the validated capture registry; its original take and derivative lineage come from
there. An unregistered file fails closed (INVALID_INPUT). This is the controlled
evaluation ingestion contract, not a public service for arbitrary uploads.
`--development-unmanaged` keeps the Phase 1C manual lineage flags for development
only; it cannot detect an undeclared derivative and is not valid study evidence.
"""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gait_verification.capture_registry import ProvenanceError, load_registry
from gait_verification.enrollments import DirectoryEnrollmentStore
from gait_verification.pipeline import (
    CachedPoseExtractor,
    lineage_from_manifest,
    query_verifier,
    verify_registered,
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
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--capture-registry",
        help="managed study: lineage resolved from this validated registry by checksum",
    )
    mode.add_argument(
        "--development-unmanaged",
        action="store_true",
        help="DEVELOPMENT ONLY: manual lineage flags below; not study evidence",
    )
    parser.add_argument(
        "--query-take-id", help="development-unmanaged only; defaults to video SHA-256"
    )
    parser.add_argument(
        "--derivative-manifest",
        help="development-unmanaged only: stabilization manifest.json of a derivative",
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
    if args.capture_registry and (args.query_take_id or args.derivative_manifest):
        parser.error("with --capture-registry, lineage comes only from the registry")
    policy = load_quality_policy()
    store = DirectoryEnrollmentStore(args.enrollments)

    def make_verifier(take_id, variant_id):
        # Models load only after the query is resolved; no subject ID is passed.
        return query_verifier(
            store,
            load_encoder(args.model, args.source_root, args.checkpoint_root, policy),
            CachedPoseExtractor(
                load_rtmpose(args.rtmpose_dir), args.pose_cache_dir, take_id, variant_id
            ),
            policy,
            scoring_strategy=args.scoring_strategy,  # deliberately no decision policy
        )

    if args.capture_registry:
        try:
            registry = load_registry(args.capture_registry)
        except ProvenanceError as exc:
            print(json.dumps({"error": f"capture registry rejected: {exc}"}))
            return 3
        result, diagnostics = verify_registered(
            registry, args.video, args.claimed_id, make_verifier
        )
    else:
        lineage = (
            lineage_from_manifest(
                json.loads(Path(args.derivative_manifest).read_text())
            )
            if args.derivative_manifest
            else ()
        )
        result, diagnostics = make_verifier(args.query_take_id, "raw").verify_detailed(
            args.video, args.claimed_id, lineage
        )
        diagnostics = dict(diagnostics, lineage_mode="DEVELOPMENT_UNMANAGED")
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
