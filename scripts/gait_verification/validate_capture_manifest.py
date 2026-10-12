"""Validate new-cohort capture rows; preserve unknown acquisition settings as null."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gait_verification.capture_registry import validate_capture_rows as validate  # noqa: E402


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", help="JSON array of original capture rows")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    with open(root / "configs/gait_verification/capture_schema.json") as f:
        schema = json.load(f)
    with open(args.manifest) as f:
        rows = json.load(f)
    print(
        f"Validated {validate(rows, schema)} originals; no session independence inferred"
    )
