"""
Cross-generator face-swap evaluation (plan Section 8 / P6 / E9, RQ7).
=====================================================================
Scores face-swapped clips from any number of generators against the claimed
(face) identity with a deployable verifier bundle, and reports per
generator and per codec:

  rejection_rate      fraction of swaps rejected (IDENTITY_MISMATCH) at the
                      bundle's LOSO-derived threshold -- the detection rate
  body_source_rank1   fraction whose best-matching enrolled gait is the true
                      body source (the "gait survives the swap" check)
  mean scores         vs the claimed face identity and vs the body identity

Optionally re-encodes every clip at several JPEG qualities before pose
estimation (--compress 50 20) to test survival under media degradation.

Manifest CSV columns (header required):
    video,body_identity,face_identity,generator[,codec]
e.g.
    data/deepfake/Arhaan_body_Teja_face.mp4,Arhaan,Teja,facefusion_inswapper128,h264

Usage:
    python scripts/future_work/generator_robustness.py \\
        --checkpoint outputs/future_work/checkpoints/E0_baseline.pt \\
        --manifest data/deepfake/manifest.csv --compress 50 20

Author: DeepFake Detection Project
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from extract_pose_cache import DegradedBackend

from utils.gait_descriptors import clip_from_track
from utils.gait_verifier import GaitVerifier
from utils.pose_backends import create_backend


def main():
    ap = argparse.ArgumentParser(description="Cross-generator face-swap evaluation")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--compress", type=int, nargs="*", default=[])
    ap.add_argument("--out", default="outputs/future_work/generators/results.json")
    args = ap.parse_args()

    verifier = GaitVerifier.load(args.checkpoint)
    with open(args.manifest, newline="") as fh:
        rows = list(csv.DictReader(fh))
    base = create_backend(verifier.cfg.backend)
    conditions = [("original", base)] + [
        (f"jpeg{q}", DegradedBackend(base, f"jpeg:{q}")) for q in args.compress
    ]

    records = []
    for row in rows:
        for cond, backend in conditions:
            track = backend.process_video(row["video"])
            clip = clip_from_track(track, Path(row["video"]).stem, verifier.cfg.source)
            rec = {**row, "condition": cond}
            if clip is None:
                rec["verdict"] = "NO_PERSON"
                records.append(rec)
                continue
            res = verifier.verify(clip, row["face_identity"])
            rec["verdict"] = res["verdict"]
            if "scores_all_identities" in res:
                s = res["scores_all_identities"]
                rec["score_claimed_face"] = s.get(row["face_identity"])
                rec["score_body_source"] = s.get(row["body_identity"])
                rec["best_match"] = res["best_match"]
            records.append(rec)
            print(
                f"{Path(row['video']).name:<40s} {row['generator']:<24s} {cond:<9s} "
                f"{rec['verdict']:<18s} face={rec.get('score_claimed_face', float('nan')):.3f} "
                f"body={rec.get('score_body_source', float('nan')):.3f} "
                f"best={rec.get('best_match', '-')}"
            )

    summary = defaultdict(dict)
    keyf = lambda r: (r["generator"], r.get("codec", ""), r["condition"])  # noqa: E731
    groups = defaultdict(list)
    for r in records:
        groups[keyf(r)].append(r)
    for (gen, codec, cond), rs in sorted(groups.items()):
        scored = [r for r in rs if "best_match" in r]
        summary[gen][f"{codec or 'any'}|{cond}"] = {
            "n": len(rs),
            "rejection_rate": float(
                np.mean([r["verdict"] == "IDENTITY_MISMATCH" for r in rs])
            ),
            "body_source_rank1": (
                float(np.mean([r["best_match"] == r["body_identity"] for r in scored]))
                if scored
                else None
            ),
            "mean_score_claimed_face": (
                float(np.mean([r["score_claimed_face"] for r in scored]))
                if scored
                else None
            ),
            "mean_score_body_source": (
                float(np.mean([r["score_body_source"] for r in scored]))
                if scored
                else None
            ),
            "insufficient_or_no_person": int(
                sum(r["verdict"] in ("INSUFFICIENT_GAIT", "NO_PERSON") for r in rs)
            ),
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "threshold": verifier.threshold,
                "summary": summary,
                "records": records,
            },
            fh,
            indent=1,
            default=float,
        )
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
