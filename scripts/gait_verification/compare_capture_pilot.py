"""Run an explicit development-only RAW/S10/S15 list with frozen CUDA RTMPose."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gait_verification.contracts import Recording
from gait_verification.pose import RTMPoseDetector
from gait_verification.pose_quality import WindowQualityPolicy
from gait_verification.pilot import run_pilot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--selection-json",
        required=True,
        help="List of {video,recording}; no dataset discovery",
    )
    p.add_argument("--detector", required=True)
    p.add_argument("--pose-model", required=True)
    p.add_argument("--new-output-dir", required=True)
    p.add_argument("--config", default="configs/gait_verification/phase1a.json")
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    selection = json.loads(Path(args.selection_json).read_text())
    detector = RTMPoseDetector(
        args.detector,
        args.pose_model,
        expected_detector_sha256=config["rtmpose"]["detector"]["sha256"],
        expected_pose_sha256=config["rtmpose"]["pose"]["sha256"],
    )
    policy = WindowQualityPolicy(**config["quality"])
    result = run_pilot(
        [(row["video"], Recording(**row["recording"])) for row in selection],
        detector,
        args.new_output_dir,
        policy,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
