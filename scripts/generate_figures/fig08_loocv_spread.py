"""
Figure 8 -- Fold-to-fold dispersion of the 13-fold LOOCV evaluation.

The point of this plate is to show that the "+/- std" in the aggregate row is
not concealing a bimodal result or a single catastrophic fold. Panel (a) gives
the full distribution of each metric across folds; panel (b) names the subjects
so the two tail folds are identifiable rather than anonymous.

Source: outputs/evaluation/loocv/loocv_results.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt

import figstyle as fs

METRICS = [("roc_auc", "ROC-AUC"), ("accuracy", "Accuracy"), ("f1", "F1"),
           ("precision", "Precision"), ("recall", "Recall"), ("eer", "EER")]


def main():
    fs.apply_style()
    agg, per_fold, y, s = fs.load_loocv()

    names = [f["test_person"] for f in per_fold]
    data = {k: np.array([f[k] for f in per_fold]) * 100 for k, _ in METRICS}

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.16, 3.15), gridspec_kw={"width_ratios": [1.0, 1.25]})

    # ---------------- (a) distribution per metric ----------------
    # EER lives on a 0-20% scale while the other five sit near 90%; plotted on
    # one axis the five would collapse into a band and the dispersion this
    # panel exists to show would be invisible. EER therefore gets its own
    # right-hand scale, separated by a rule.
    hi = METRICS[:-1]                       # higher-is-better metrics
    lo = METRICS[-1:]                       # EER
    ax1b = ax1.twinx()

    def draw(axis, entries, positions, colour):
        vals = [data[k] for k, _ in entries]
        vp = axis.violinplot(vals, positions=positions, showextrema=False,
                             widths=0.82)
        for body in vp["bodies"]:
            body.set_facecolor(colour)
            body.set_alpha(0.42)
            body.set_edgecolor("none")
        axis.boxplot(vals, positions=positions, widths=0.20,
                     patch_artist=True, showfliers=False,
                     medianprops=dict(color=fs.C_ACCENT, lw=1.3),
                     boxprops=dict(facecolor="white",
                                   edgecolor=fs.C_NEUTRAL, lw=0.7),
                     whiskerprops=dict(color=fs.C_NEUTRAL, lw=0.7),
                     capprops=dict(color=fs.C_NEUTRAL, lw=0.7))
        rng = np.random.default_rng(0)
        for p, v in zip(positions, vals):
            axis.scatter(p + rng.uniform(-0.07, 0.07, v.size), v, s=8,
                         color=colour, alpha=0.85, zorder=4, linewidth=0)
        return vals

    pos_hi = list(range(1, len(hi) + 1))
    pos_lo = [len(hi) + 1]
    draw(ax1, hi, pos_hi, fs.C_AUTH)
    draw(ax1b, lo, pos_lo, fs.C_FAKE)

    for p, (k, _) in zip(pos_hi + pos_lo, hi + lo):
        v = data[k]
        axis = ax1 if (k, _) in hi else ax1b
        top = 100.6 if axis is ax1 else 20.6
        axis.text(p, top, "%.1f\n$\\pm$%.1f" % (v.mean(), v.std()),
                  ha="center", va="bottom", fontsize=6.3,
                  color=fs.C_AUTH if axis is ax1 else fs.C_FAKE)

    ax1.axvline(len(hi) + 0.5, color=fs.C_GRID, lw=0.9, zorder=0)
    ax1.set_xticks(pos_hi + pos_lo, labels=[lab for _, lab in METRICS],
                   rotation=30, ha="right")
    ax1.set_xlim(0.4, len(METRICS) + 0.6)
    ax1.set_ylim(70, 108)
    ax1.set_yticks([70, 80, 90, 100])
    ax1.set_ylabel("Higher is better (%)", color=fs.C_AUTH)
    ax1.tick_params(axis="y", colors=fs.C_AUTH)
    ax1b.set_ylim(0, 28)
    ax1b.set_yticks([0, 5, 10, 15, 20])
    ax1b.set_ylabel("EER, lower is better (%)", color=fs.C_FAKE)
    ax1b.tick_params(axis="y", colors=fs.C_FAKE)
    ax1b.grid(False)
    ax1.set_title("(a) Distribution across the 13 folds", loc="left", pad=11)
    for side in ("top", "right"):
        ax1.spines[side].set_visible(False)
    for side in ("top", "left"):
        ax1b.spines[side].set_visible(False)
    ax1b.spines["right"].set_color(fs.C_FAKE)
    ax1.spines["left"].set_color(fs.C_AUTH)

    # ---------------- (b) per-subject ROC-AUC and accuracy ----------------
    order = np.argsort(data["roc_auc"])
    ypos = np.arange(len(order))
    auc_sorted = data["roc_auc"][order]
    acc_sorted = data["accuracy"][order]
    names_sorted = [names[i] for i in order]

    ax2.barh(ypos + 0.19, auc_sorted, height=0.36, color=fs.C_AUTH,
             label="ROC-AUC", edgecolor="white")
    ax2.barh(ypos - 0.19, acc_sorted, height=0.36, color=fs.C_FOLD,
             label="Accuracy", edgecolor="white")

    mean_auc = data["roc_auc"].mean()
    ax2.axvline(mean_auc, color=fs.C_ACCENT, lw=1.0, ls=(0, (4, 2)), zorder=5)
    ax2.text(mean_auc, -1.15, "mean AUC %.2f%%" % mean_auc, fontsize=6.4,
             ha="center", va="center", color=fs.C_ACCENT)

    for i, (a, c) in enumerate(zip(auc_sorted, acc_sorted)):
        ax2.text(a + 0.7, i + 0.19, "%.1f" % a, va="center", fontsize=6.2,
                 color=fs.C_NEUTRAL)
        ax2.text(c + 0.7, i - 0.19, "%.1f" % c, va="center", fontsize=6.2,
                 color=fs.C_NEUTRAL)

    ax2.set_yticks(ypos, labels=names_sorted)
    ax2.set_xlim(60, 104)
    ax2.set_ylim(-1.9, len(order) - 0.35)
    ax2.set_xlabel("Value (%)")
    ax2.set_title("(b) Per held-out subject", loc="left", pad=11)
    # Legend above the axes: every bar starts at the left spine, so there is no
    # interior region it could occupy without covering data.
    ax2.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2,
               frameon=False, handlelength=1.4, columnspacing=1.2)
    ax2.grid(axis="y", visible=False)
    fs.despine(ax2)

    fig.tight_layout(w_pad=1.4)
    fs.save(fig, "fig8_loocv_spread.png")

    for k, lab in METRICS:
        v = data[k]
        print("    %-10s mean %6.2f  std %5.2f  min %6.2f (%s)  max %6.2f (%s)"
              % (lab, v.mean(), v.std(), v.min(), names[int(np.argmin(v))],
                 v.max(), names[int(np.argmax(v))]))


if __name__ == "__main__":
    main()
