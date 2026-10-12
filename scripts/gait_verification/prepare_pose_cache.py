"""Extract one native COCO17 RTMPose cache with PTS and explicit local weights."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gait_verification.pose import RTMPoseDetector, NativePoseExtractor, save_pose
from gait_verification.skeleton_contracts import assess_quality
from gait_verification.pose_quality import analyze_pose, WindowQualityPolicy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True)
    p.add_argument("--take-id", required=True)
    p.add_argument("--detector", required=True)
    p.add_argument("--pose-model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--window-quality",
        action="store_true",
        help="Collect temporal diagnostics without aborting the whole recording",
    )
    p.add_argument("--quality-config", default="configs/gait_verification/phase1a.json")
    args = p.parse_args()
    config = (
        json.loads(Path(args.quality_config).read_text())
        if args.window_quality
        else None
    )
    checks = (
        dict(
            expected_detector_sha256=config["rtmpose"]["detector"]["sha256"],
            expected_pose_sha256=config["rtmpose"]["pose"]["sha256"],
        )
        if config
        else {}
    )
    detector = RTMPoseDetector(args.detector, args.pose_model, **checks)
    sequence = NativePoseExtractor(
        detector, args.take_id, collect_diagnostics=args.window_quality
    ).extract(args.video)
    save_pose(sequence, args.output)
    print(
        json.dumps(
            analyze_pose(sequence, WindowQualityPolicy(**config["quality"]))
            if config
            else asdict(assess_quality(sequence)),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
