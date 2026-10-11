"""Phase 0 CLI: public 1:1 contract, no calibrated model policy installed yet."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gait_verification.enrollments import DirectoryEnrollmentStore
from gait_verification.verifier import ClaimVerifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--claimed-id", required=True)
    parser.add_argument("--enrollments", default="data/gait_verification/enrollments")
    args = parser.parse_args()
    # Fail closed until source compatibility, pilot QA and calibration gates pass.
    result = ClaimVerifier(DirectoryEnrollmentStore(args.enrollments)).verify(
        args.video, args.claimed_id
    )
    print(json.dumps(result.public_dict()))
    return 0 if result.verdict is not None else 2


if __name__ == "__main__":
    sys.exit(main())
