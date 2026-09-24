"""
Human-readable forensic verification report (plan Section 10 / P7, RQ8, RQ10).
=============================================================================
Explains WHY a claimed identity was accepted or rejected, not only the
binary verdict:

  1. Verdict       AUTHENTIC / IDENTITY_MISMATCH / UNKNOWN_IDENTITY /
                   INSUFFICIENT_GAIT, calibrated probability, threshold, and
                   the best-matching enrolled identity (likely body source).
  2. Sub-scores    cadence, stride time, stance, step length, symmetry ...
                   compared with the claimed identity's enrolment, each with
                   its test-retest reliability (ICC) so weak cues are flagged.
  3. Joints        gradient x input attribution over the 12 gait joints.
  4. Families      share of attribution per feature family.
  5. When          temporal localisation: which part of the walk, if made to
                   agree with the claim, would raise the score most.
  6. Counterfactual  score change when one feature family is replaced by the
                   claimed identity's -- which family the decision hinges on.

Writes <out>.json, <out>.md and <out>.png.

Usage:
    python scripts/future_work/forensic_report.py \\
        --checkpoint outputs/future_work/checkpoints/E0_baseline.pt \\
        --video data/deepfake/Arhaan_body_Teja_face.mp4 --claimed Teja

    # from an existing pose cache (no pose inference)
    python scripts/future_work/forensic_report.py --checkpoint ... \\
        --pose data/pose_cache/mediapipe_lite/Som_S1.npz --claimed Teja

Author: DeepFake Detection Project
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generate_figures"))


from utils.gait_descriptors import clip_from_track
from utils.gait_verifier import GaitVerifier
from utils.pose_backends import PoseTrack, create_backend

READABLE = {
    "cadence_spm": "Cadence (steps/min)",
    "stride_time_s": "Stride time (s)",
    "stride_time_cv": "Stride-time variability (CV)",
    "step_time_asym": "Step-time asymmetry",
    "step_regularity": "Step regularity",
    "stride_regularity": "Stride regularity",
    "regularity_symmetry": "Regularity symmetry",
    "harmonic_ratio_v": "Harmonic ratio (vertical)",
    "spectral_entropy": "Spectral entropy",
    "phase_offset": "L/R phase offset",
    "phase_locking": "L/R phase locking",
    "stride_length": "Stride length (leg lengths)",
    "step_length_l": "Left step length",
    "step_length_r": "Right step length",
    "step_length_asym": "Step-length asymmetry",
    "step_width": "Step width",
    "stance_frac_l": "Left stance fraction",
    "stance_frac_r": "Right stance fraction",
    "double_support_frac": "Double-support fraction",
    "clearance_l": "Left foot clearance",
    "clearance_r": "Right foot clearance",
}


def load_clip(args, verifier):
    if args.pose:
        track = PoseTrack.load(args.pose)
        name = Path(args.pose).stem
    else:
        backend = create_backend(verifier.cfg.backend)
        track = backend.process_video(args.video)
        name = Path(args.video).stem
    clip = clip_from_track(track, name, verifier.cfg.source)
    if clip is None:
        raise SystemExit("No person detected in enough frames.")
    return clip, track


def markdown(rep: dict) -> str:
    v = rep["verification"]
    lines = [f"# Gait verification report: `{rep['input']}`\n"]
    lines.append(f"**Claimed identity:** {v['claimed_identity']}  ")
    lines.append(f"**Verdict:** **{v['verdict']}**  ")
    if "probability_authentic" in v:
        lines.append(
            f"**P(authentic), calibrated:** {v['probability_authentic']:.3f} "
            f"(raw {v['score_raw']:.3f}, threshold {v['threshold']:.3f})  "
        )
        lines.append(
            f"**Best-matching enrolled gait:** {v['best_match']} "
            f"({v['best_match_score']:.3f})  "
        )
        if v.get("likely_body_source"):
            lines.append(f"**Likely body source:** {v['likely_body_source']}  ")
    lines.append(
        f"**Clip:** {v['frames']} frames, {v['duration_s']:.1f} s, "
        f"{v.get('n_strides', 0)} strides detected\n"
    )
    if "message" in v:
        lines.append(f"> {v['message']}\n")
        return "\n".join(lines)

    lines.append("## Interpretable gait agreement with the claimed identity\n")
    lines.append(
        "Agreement = exp(-z²/2), z = (query - claimed mean) / within-person SD. "
        "ICC is the parameter's test-retest reliability on the enrolment data; "
        "treat parameters with ICC < 0.3 as weak evidence.\n"
    )
    lines.append("| Parameter | Query | Claimed mean | z | Agreement | ICC |")
    lines.append("|---|---|---|---|---|---|")
    subs = sorted(rep["sub_scores"].items(), key=lambda kv: -kv[1]["icc"])
    for p, s in subs:
        flag = "" if s["icc"] >= 0.3 else " (weak)"
        lines.append(
            f"| {READABLE.get(p, p)} | {s['query']:.3f} | {s['enrolled_mean']:.3f} "
            f"| {s['z']:+.2f} | {s['agreement']:.2f} | {s['icc']:.2f}{flag} |"
        )
    e = rep["explanation"]
    lines.append("\n## Joints driving the decision (gradient x input, max = 1)\n")
    for j, w in sorted(e["joint_attribution"].items(), key=lambda kv: -kv[1]):
        lines.append(f"- {j}: {w:.2f}")
    lines.append("\n## Feature-family share of attribution\n")
    for f, w in sorted(e["family_attribution"].items(), key=lambda kv: -kv[1]):
        lines.append(f"- {f}: {100 * w:.1f}%")
    lines.append(
        "\n## Where in the walk the mismatch lies\n\n"
        "Score change if that window of the query is replaced by the claimed "
        "identity's enrolled gait (largest = most responsible):\n"
    )
    top = sorted(e["temporal_windows"], key=lambda w: -w["delta_score_if_matched"])[:3]
    for w in top:
        lines.append(
            f"- {w['t_start_s']:.2f}-{w['t_end_s']:.2f} s: "
            f"{w['delta_score_if_matched']:+.3f}"
        )
    lines.append("\n## Counterfactual: replace one feature family with the claim's\n")
    for f, d in sorted(
        e["counterfactual_family_swap"].items(), key=lambda kv: -abs(kv[1])
    ):
        lines.append(f"- {f}: score {d:+.3f}")
    lines.append(
        "\n---\nScope: this verifies consistency of the walking pattern with an "
        "enrolled identity. It is not a general detector of AI-generated video, "
        "and it cannot detect manipulations that also re-synthesise body motion.\n"
    )
    return "\n".join(lines)


def figure(rep: dict, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from figstyle import C_AUTH, C_FAKE, C_NEUTRAL, apply_style, despine

    apply_style()
    e = rep["explanation"]
    fig, (a1, a2) = plt.subplots(
        1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1.1, 1]}
    )
    w = e["temporal_windows"]
    mid = [(x["t_start_s"] + x["t_end_s"]) / 2 for x in w]
    d = [x["delta_score_if_matched"] for x in w]
    a1.plot(mid, d, color=C_AUTH, lw=1.6)
    a1.axhline(0, color=C_NEUTRAL, lw=0.8)
    a1.set_xlabel("time in clip (s)")
    a1.set_ylabel("Δ score if window matched claim")
    a1.set_title("Where the gait departs from the claim")
    despine(a1)
    j = sorted(e["joint_attribution"].items(), key=lambda kv: kv[1])
    a2.barh([k for k, _ in j], [v for _, v in j], color=C_FAKE, height=0.6)
    a2.set_xlabel("relative attribution")
    a2.set_title("Joints driving the decision")
    a2.grid(axis="y", visible=False)
    despine(a2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Forensic gait verification report")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--video", default="")
    ap.add_argument("--pose", default="", help="pose cache .npz instead of video")
    ap.add_argument("--claimed", required=True)
    ap.add_argument("--out", default="outputs/future_work/reports/report")
    args = ap.parse_args()
    if not (args.video or args.pose):
        raise SystemExit("--video or --pose required")

    verifier = GaitVerifier.load(args.checkpoint)
    clip, _ = load_clip(args, verifier)
    ver = verifier.verify(clip, args.claimed)
    rep = {"input": args.video or args.pose, "verification": ver}
    if ver["verdict"] in ("AUTHENTIC", "IDENTITY_MISMATCH"):
        rep["sub_scores"] = verifier.sub_scores(ver["gait_parameters"], args.claimed)
        rep["explanation"] = verifier.explain(clip, args.claimed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".json"), "w") as fh:
        json.dump(rep, fh, indent=1, default=float)
    md = markdown(rep)
    out.with_suffix(".md").write_text(md, encoding="utf8")
    if "explanation" in rep:
        figure(rep, str(out.with_suffix(".png")))
    print(md)


if __name__ == "__main__":
    main()
