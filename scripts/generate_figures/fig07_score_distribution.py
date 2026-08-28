"""
Figure 7 -- Distribution of the verification score P(authentic) over all 2,240
pooled LOOCV pairs, separated into genuine (matched-identity) and impostor
(mismatched-identity) pairs.

This is the figure that shows *why* the aggregate metrics look the way they do:
both classes pile up hard against their respective extremes, and essentially
all of the residual error lives in a thin band of ambiguous scores between the
equal-error threshold and the Youden threshold.

Source: outputs/evaluation/loocv/loocv_results.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt

import figstyle as fs


def main():
    fs.apply_style()
    agg, per_fold, y, s = fs.load_loocv()

    gen = s[y == 1]
    imp = s[y == 0]
    thr_j, _, _, _ = fs.youden(y, s)
    eer, thr_eer = fs.eer_point(y, s)

    fig, ax = plt.subplots(figsize=(3.45, 2.75))
    bins = np.linspace(0, 1, 41)

    ax.hist(imp, bins=bins, color=fs.C_FAKE, alpha=0.78, lw=0.4,
            edgecolor="white",
            label="Impostor pairs ($n$ = %d)" % imp.size)
    ax.hist(gen, bins=bins, color=fs.C_AUTH, alpha=0.78, lw=0.4,
            edgecolor="white",
            label="Genuine pairs ($n$ = %d)" % gen.size)

    ax.axvline(thr_eer, color=fs.C_NEUTRAL, lw=1.0, ls=(0, (1, 2)))
    ax.axvline(thr_j, color=fs.C_ACCENT, lw=1.1, ls=(0, (4, 2)))

    ymax = ax.get_ylim()[1]
    ax.text(thr_eer - 0.02, ymax * 0.97, r"EER $\tau$" + "\n%.3f" % thr_eer,
            fontsize=6.6, ha="right", va="top", color=fs.C_NEUTRAL)
    ax.text(thr_j + 0.02, ymax * 0.97, r"Youden $\tau^*$" + "\n%.3f" % thr_j,
            fontsize=6.6, ha="left", va="top", color=fs.C_ACCENT)

    ax.set_xlabel(r"Verification score $P(\mathrm{authentic})$")
    ax.set_ylabel("Verification pairs")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper center")
    fs.despine(ax)

    fig.tight_layout()
    fs.save(fig, "fig7_score_distribution.png")

    print("    genuine  mean %.4f  median %.4f" % (gen.mean(), np.median(gen)))
    print("    impostor mean %.4f  median %.4f" % (imp.mean(), np.median(imp)))
    band = ((s > min(thr_eer, thr_j)) & (s < max(thr_eer, thr_j))).sum()
    print("    pairs between the two thresholds: %d (%.1f%%)"
          % (band, 100 * band / s.size))


if __name__ == "__main__":
    main()
