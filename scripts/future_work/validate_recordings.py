"""
Validate a recording manifest before data is shared (plan Section 8).
=====================================================================
Checks recordings.csv against DOCUMENTATION/DATA_COLLECTION_PROTOCOL.md:
required columns, allowed values, file naming, that each video opens, its
frame rate and duration, duplicate files, and prints a subject x condition
coverage table so gaps are visible before leaving the recording site.

Exit code 1 if any ERROR is found (warnings do not fail).

Usage:
    python scripts/future_work/validate_recordings.py --manifest recordings.csv --videos_dir videos/

Author: DeepFake Detection Project
"""

import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict

REQUIRED = [
    "file",
    "subject_id",
    "site",
    "session",
    "view",
    "take",
    "camera",
    "fps",
    "distance",
    "speed",
    "lighting",
    "clothing",
    "surface",
    "consent_id",
]
ALLOWED = {
    "view": {"F", "S", "OL", "OR"},
    "distance": {"near", "medium", "far"},
    "speed": {"slow", "natural", "fast"},
    "lighting": {"indoor", "outdoor", "lowlight", "backlit"},
    "clothing": {"normal", "jacket", "loose"},
    "surface": {"tile", "concrete", "grass", "corridor", "other"},
}
NAME_RE = re.compile(r"^(?P<subject>[A-Za-z0-9]+)_(?P<view>[A-Za-z]+)(?P<take>\d+)$")
MIN_SECONDS = 6.0


def main():
    ap = argparse.ArgumentParser(description="Validate a recording manifest")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--videos_dir", default="")
    ap.add_argument("--no_video_check", action="store_true")
    args = ap.parse_args()

    errors, warnings = [], []
    with open(args.manifest, newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED if c not in (reader.fieldnames or [])]
        if missing:
            print(f"ERROR: manifest is missing columns {missing}")
            sys.exit(1)
        rows = list(reader)

    dup = [f for f, n in Counter(r["file"] for r in rows).items() if n > 1]
    errors += [f"duplicate file entry: {f}" for f in dup]

    for i, r in enumerate(rows, start=2):
        where = f"row {i} ({r['file']})"
        for c in REQUIRED:
            if not r[c].strip():
                errors.append(f"{where}: empty '{c}'")
        for c, allowed in ALLOWED.items():
            if r[c] and r[c] not in allowed:
                errors.append(f"{where}: {c}='{r[c]}' not in {sorted(allowed)}")
        stem = os.path.splitext(os.path.basename(r["file"]))[0]
        m = NAME_RE.match(stem)
        if not m:
            errors.append(f"{where}: name must be {{SubjectID}}_{{View}}{{Take}}")
        else:
            if m["subject"] != r["subject_id"]:
                errors.append(f"{where}: file subject '{m['subject']}' != subject_id")
            if m["view"] != r["view"]:
                errors.append(f"{where}: file view '{m['view']}' != view column")
            if m["take"] != r["take"]:
                errors.append(f"{where}: file take '{m['take']}' != take column")

        if args.no_video_check:
            continue
        path = os.path.join(args.videos_dir, r["file"]) if args.videos_dir else r["file"]
        if not os.path.exists(path):
            errors.append(f"{where}: file not found at {path}")
            continue
        import cv2

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            errors.append(f"{where}: cannot be opened")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        cap.release()
        dur = frames / fps if fps else 0.0
        if dur < MIN_SECONDS:
            warnings.append(f"{where}: {dur:.1f} s < {MIN_SECONDS} s recommended")
        if fps and abs(fps - float(r["fps"] or 0)) > 1.5:
            warnings.append(f"{where}: actual fps {fps:.1f} != manifest {r['fps']}")

    # coverage
    by_subject = defaultdict(list)
    for r in rows:
        by_subject[r["subject_id"]].append(r)
    for s, rs in by_subject.items():
        views = Counter(r["view"] for r in rs)
        if views.get("S", 0) < 3 or views.get("F", 0) < 3:
            warnings.append(f"{s}: fewer than 3 side and 3 frontal takes ({dict(views)})")
        if len({r["session"] for r in rs}) < 2:
            warnings.append(f"{s}: single session (a second day is recommended)")

    fields = ["view", "session", "speed", "distance", "lighting", "clothing", "camera"]
    print(f"\n{len(rows)} recordings, {len(by_subject)} subjects\n")
    print(f"{'subject':<12s}" + "".join(f"{f:<22s}" for f in fields))
    for s in sorted(by_subject):
        rs = by_subject[s]
        cells = []
        for f in fields:
            c = Counter(r[f] for r in rs)
            cells.append(",".join(f"{k}:{v}" for k, v in sorted(c.items()))[:21])
        print(f"{s:<12s}" + "".join(f"{c:<22s}" for c in cells))

    for w in warnings:
        print(f"WARNING: {w}")
    for e in errors:
        print(f"ERROR: {e}")
    print(f"\n{len(errors)} error(s), {len(warnings)} warning(s)")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
