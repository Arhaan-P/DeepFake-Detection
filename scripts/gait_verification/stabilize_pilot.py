"""Opt-in single source pilot with a supplied recording manifest; never batch."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gait_verification.contracts import Recording
from gait_verification.stabilization import StabilizationConfig, stabilize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True)
    p.add_argument("--recording-json", required=True)
    p.add_argument("--new-work-dir", required=True)
    p.add_argument("--enable", action="store_true")
    p.add_argument("--smoothing", type=int, choices=[10, 15], default=10)
    args = p.parse_args()
    with open(args.recording_json) as f:
        recording = Recording(**json.load(f))
    result = stabilize(
        args.video,
        args.new_work_dir,
        recording,
        StabilizationConfig(enabled=args.enable, smoothing=args.smoothing),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "manifest": str(Path(args.new_work_dir) / "manifest.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
