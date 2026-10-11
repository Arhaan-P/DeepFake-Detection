"""Validate new-cohort capture rows; preserve unknown acquisition settings as null."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import re


def validate(rows, schema):
    seen = set()
    sources = set()
    for row in rows:
        missing = set(schema["required_fields"]) - row.keys()
        if missing:
            raise ValueError(f"Missing fields: {sorted(missing)}")
        for field in (
            "original_take_id",
            "subject_id",
            "session_id",
            "date",
            "sha256",
            "consent_ref",
            "source_file_uri",
            "actual_frame_timestamps_ref",
        ):
            if not row[field]:
                raise ValueError(f"Missing required provenance: {field}")
        datetime.fromisoformat(row["date"])
        if not re.fullmatch(r"[a-f0-9]{64}", row["sha256"]):
            raise ValueError("Invalid SHA-256")
        if row["original_take_id"] in seen or row["sha256"] in sources:
            raise ValueError("Duplicate original take/content")
        seen.add(row["original_take_id"])
        sources.add(row["sha256"])
        if row["view_label"] not in schema["planned_views"]:
            raise ValueError("Unknown capture view")
        if row["duration_sec"] is not None and row["duration_sec"] <= 0:
            raise ValueError("Invalid duration")
    return len(rows)


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
