"""Extract one native COCO17 RTMPose cache with PTS and explicit local weights."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gait_verification.pose import RTMPoseDetector, NativePoseExtractor, save_pose
from gait_verification.skeleton_contracts import assess_quality


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True)
    p.add_argument("--take-id", required=True)
    p.add_argument("--detector", required=True)
    p.add_argument("--pose-model", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    detector = RTMPoseDetector(args.detector, args.pose_model)
    sequence = NativePoseExtractor(detector, args.take_id).extract(args.video)
    save_pose(sequence, args.output)
    print(json.dumps(asdict(assess_quality(sequence)), indent=2))


if __name__ == "__main__":
    main()
