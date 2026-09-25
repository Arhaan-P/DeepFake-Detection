"""
Gait-cycle and rhythm analysis report (plan P2, Section 6, RQ3/RQ5).
====================================================================
Model-free: which interpretable gait parameters are stable within a person
and different between people? For every clip it measures the rhythm and
spatial parameters of utils/gait_cycle.py, then per parameter reports

  ICC(1,1)        one-way random-effects intraclass correlation across
                  subjects (test-retest reliability of the parameter)
  fisher_ratio    between-subject variance / within-subject variance
  within_sd       pooled within-subject standard deviation
  between_sd      standard deviation of subject means

overall and separately for frontal (F) and side (S) walks. It also runs a
leave-one-clip-out nearest-neighbour identification and a verification AUC
using ONLY the standardised clip-level parameters -- a floor for how much
identity the interpretable parameters carry without any learned model.

Usage:
    python scripts/future_work/rhythm_analysis.py
    python scripts/future_work/rhythm_analysis.py --backend mediapipe_heavy_video

Author: DeepFake Detection Project
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from utils.gait_cycle import GaitCycleAnalysis, analyse_gait
from utils.gait_features import leg_length
from utils.verification_harness import load_clips
from utils.verification_metrics import eer, roc_auc

PARAMS = list(GaitCycleAnalysis.RHYTHM_NAMES) + list(GaitCycleAnalysis.STRIDE_NAMES)


def icc_1_1(values: np.ndarray, groups: np.ndarray) -> float:
    """ICC(1,1) for unbalanced one-way random-effects designs."""
    ids = np.unique(groups)
    a, n = len(ids), len(values)
    grand = values.mean()
    sizes = np.array([np.sum(groups == g) for g in ids])
    means = np.array([values[groups == g].mean() for g in ids])
    ssb = np.sum(sizes * (means - grand) ** 2)
    ssw = sum(np.sum((values[groups == g] - m) ** 2) for g, m in zip(ids, means))
    msb, msw = ssb / (a - 1), ssw / (n - a)
    k0 = (n - np.sum(sizes**2) / n) / (a - 1)
    return float((msb - msw) / (msb + (k0 - 1) * msw + 1e-12))


def param_stats(x: np.ndarray, groups: np.ndarray) -> dict:
    ids = np.unique(groups)
    means = np.array([x[groups == g].mean() for g in ids])
    within = np.sqrt(
        np.mean(
            np.concatenate([(x[groups == g] - x[groups == g].mean()) ** 2 for g in ids])
        )
    )
    between = float(np.std(means, ddof=1))
    return {
        "icc": icc_1_1(x, groups),
        "fisher_ratio": float(between**2 / (within**2 + 1e-12)),
        "within_sd": float(within),
        "between_sd": between,
        "mean": float(x.mean()),
    }


def nn_identification(feats: np.ndarray, groups: np.ndarray) -> dict:
    """Leave-one-clip-out: standardise on the other clips, then (a) rank-1
    identification against subject means and (b) verification scores."""
    correct, labels, scores = 0, [], []
    ids = np.unique(groups)
    for i in range(len(feats)):
        m = np.arange(len(feats)) != i
        mu, sd = feats[m].mean(0), feats[m].std(0) + 1e-9
        z = (feats - mu) / sd
        cents = {
            g: z[m & (groups == g)].mean(0) for g in ids if np.any(m & (groups == g))
        }
        d = {g: np.linalg.norm(z[i] - c) for g, c in cents.items()}
        correct += min(d, key=d.get) == groups[i]
        for g, dist in d.items():
            labels.append(int(g == groups[i]))
            scores.append(-dist)
    labels, scores = np.array(labels), np.array(scores)
    return {
        "rank1_accuracy": correct / len(feats),
        "chance": 1 / len(ids),
        "verification_auc": roc_auc(labels, scores),
        "verification_eer": eer(labels, scores)[0],
    }


def cycle_consistency_and_alignment(clips) -> dict:
    """Plan Section 6 'temporal consistency' + RQ4, model-free.

    Signals: the six baseline joint angles (scale-free). For each clip:
      * whole-clip representation: angles resampled to 60 steps (baseline)
      * cycle representation: mean heel-strike-to-heel-strike cycle, 60 pts
      * intra-clip consistency: mean pairwise correlation between cycles
    Then every clip pair is scored by (negative) distance under three
    comparisons -- whole-clip L2, mean-cycle L2 with best circular shift,
    and DTW on the whole-clip sequences -- and the genuine-vs-impostor AUC of
    each is reported.
    """
    from utils.gait_cycle import cycle_normalise, dtw
    from utils.gait_features import baseline_joint_angles, resample_sequence

    whole, cyc, consist, ids, views = [], [], [], [], []
    for c in clips:
        ang = baseline_joint_angles(c.pose)
        ang = (ang - ang.mean(0)) / (ang.std(0) + 1e-6)
        ga = analyse_gait(c.pose_iso, c.t, c.fps)
        ev = ga.hs_l if len(ga.hs_l) >= len(ga.hs_r) else ga.hs_r
        cy = cycle_normalise(ang, c.t, ev, 60)
        if cy is None:
            continue
        whole.append(resample_sequence(ang, 60))
        cyc.append(cy.mean(0))
        if len(cy) > 1:
            cors = [
                np.mean(
                    [np.corrcoef(cy[i][:, k], cy[j][:, k])[0, 1] for k in range(6)]
                )
                for i in range(len(cy))
                for j in range(i + 1, len(cy))
            ]
            consist.append(float(np.nanmean(cors)))
        ids.append(c.identity)
        views.append(c.view)

    def shifted_l2(a, b):
        return min(np.linalg.norm(a - np.roll(b, s, axis=0)) for s in range(0, 60, 3))

    labels, s_whole, s_cyc, s_dtw = [], [], [], []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if views[i] != views[j]:
                continue  # same-view pairs: isolate the alignment question
            labels.append(int(ids[i] == ids[j]))
            s_whole.append(-np.linalg.norm(whole[i] - whole[j]))
            s_cyc.append(-shifted_l2(cyc[i], cyc[j]))
            s_dtw.append(-dtw(whole[i], whole[j], band=15)[0])
    labels = np.array(labels)
    return {
        "n_clips_with_cycles": len(ids),
        "intra_clip_cycle_correlation_median": float(np.median(consist)),
        "same_view_pairs": int(len(labels)),
        "genuine_pairs": int(labels.sum()),
        "auc_whole_clip_l2": roc_auc(labels, np.array(s_whole)),
        "auc_mean_cycle_l2": roc_auc(labels, np.array(s_cyc)),
        "auc_dtw_whole_clip": roc_auc(labels, np.array(s_dtw)),
    }


def main():
    # Markdown output contains non-ASCII; Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Gait rhythm / parameter analysis")
    ap.add_argument("--backend", default="mediapipe_lite")
    ap.add_argument("--out", default="outputs/future_work/rhythm")
    args = ap.parse_args()

    clips = load_clips(args.backend)
    rows, groups, views, names, quality = [], [], [], [], []
    for c in clips:
        ga = analyse_gait(c.pose_iso, c.t, c.fps)
        d = ga.rhythm_dict()
        d.update(ga.stride_dict(leg_length(c.pose_iso)))
        rows.append([d[p] for p in PARAMS])
        groups.append(c.identity)
        views.append(c.view)
        names.append(c.name)
        quality.append(
            {
                "clip": c.name,
                "view": c.view,
                "duration_s": c.duration,
                "stride_period_s": ga.stride_period,
                "period_confidence": ga.period_confidence,
                "heel_strikes": int(len(ga.hs_l) + len(ga.hs_r)),
                "params": {p: float(v) for p, v in zip(PARAMS, rows[-1])},
            }
        )
    x = np.array(rows, dtype=float)
    groups, views = np.array(groups), np.array(views)

    report = {"backend": args.backend, "n_clips": len(clips), "parameters": {}}
    for subset in ("all", "F", "S"):
        m = np.ones(len(x), bool) if subset == "all" else views == subset
        # need >= 2 clips per subject for within-subject terms
        ok_ids = [g for g in np.unique(groups[m]) if np.sum(m & (groups == g)) >= 2]
        m &= np.isin(groups, ok_ids)
        report["parameters"][subset] = {
            p: param_stats(x[m, j], groups[m]) for j, p in enumerate(PARAMS)
        }
        report.setdefault("identification", {})[subset] = nn_identification(
            x[m], groups[m]
        )
    report["gait_event_quality_by_view"] = {
        v: {
            "median_period_confidence": float(
                np.median([q["period_confidence"] for q in quality if q["view"] == v])
            ),
            "median_heel_strikes_per_s": float(
                np.median(
                    [
                        q["heel_strikes"] / q["duration_s"]
                        for q in quality
                        if q["view"] == v
                    ]
                )
            ),
        }
        for v in sorted(set(views))
    }
    report["alignment"] = cycle_consistency_and_alignment(clips)
    report["per_clip"] = quality

    os.makedirs(args.out, exist_ok=True)
    tag = args.backend
    with open(Path(args.out) / f"rhythm_{tag}.json", "w") as fh:
        json.dump(report, fh, indent=1)

    lines = [
        f"# Gait parameter reliability ({args.backend}, {len(clips)} clips)\n",
        "| Parameter | ICC all | ICC side | ICC frontal | Fisher all | Fisher side "
        "| mean | within SD | between SD |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    order = sorted(PARAMS, key=lambda p: -report["parameters"]["all"][p]["icc"])
    for p in order:
        a = report["parameters"]["all"][p]
        s = report["parameters"]["S"][p]
        f = report["parameters"]["F"][p]
        lines.append(
            f"| {p} | {a['icc']:.2f} | {s['icc']:.2f} | {f['icc']:.2f} "
            f"| {a['fisher_ratio']:.2f} | {s['fisher_ratio']:.2f} | {a['mean']:.3f} "
            f"| {a['within_sd']:.3f} | {a['between_sd']:.3f} |"
        )
    lines.append("\n## Identity from parameters alone (no learned model)\n")
    lines.append("| Subset | Rank-1 ID | Chance | Verification AUC | EER |")
    lines.append("|---|---|---|---|---|")
    for subset, r in report["identification"].items():
        lines.append(
            f"| {subset} | {r['rank1_accuracy']:.3f} | {r['chance']:.3f} "
            f"| {r['verification_auc']:.3f} | {r['verification_eer']:.3f} |"
        )
    lines.append("\n## Gait-event detectability by view\n")
    for v, q in report["gait_event_quality_by_view"].items():
        lines.append(
            f"- **{v}**: median stride-period confidence "
            f"{q['median_period_confidence']:.2f}, "
            f"{q['median_heel_strikes_per_s']:.2f} heel strikes / s"
        )
    al = report["alignment"]
    lines.append(
        "\n## Temporal consistency and alignment (RQ4, model-free, joint angles)\n"
    )
    lines.append(
        f"- median correlation between gait cycles within a clip: "
        f"{al['intra_clip_cycle_correlation_median']:.2f}"
    )
    lines.append(
        f"- same-view clip pairs: {al['same_view_pairs']} "
        f"({al['genuine_pairs']} genuine)"
    )
    lines.append("\n| Comparison | Genuine-vs-impostor AUC |\n|---|---|")
    lines.append(f"| whole clip resampled to 60, L2 | {al['auc_whole_clip_l2']:.3f} |")
    lines.append(
        f"| mean gait cycle (phase-aligned), L2 | {al['auc_mean_cycle_l2']:.3f} |"
    )
    lines.append(f"| whole clip, DTW-aligned | {al['auc_dtw_whole_clip']:.3f} |")
    (Path(args.out) / f"rhythm_{tag}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
