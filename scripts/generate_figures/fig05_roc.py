"""
Figure 5 -- ROC analysis of the 13-fold LOOCV evaluation.

(a) Pooled ROC over all 2,240 verification pairs, with the Youden-J operating
    point and the equal-error operating point marked.
(b) The 13 individual subject-held-out ROC curves overlaid, so the reader can
    see the fold-to-fold spread that the aggregate +/- std summarises.

Source: outputs/evaluation/loocv/loocv_results.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

import figstyle as fs


def main():
    fs.apply_style()
    agg, per_fold, y, s = fs.load_loocv()
    folds = fs.fold_slices(per_fold, y, s)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.05))

    # ---------------- (a) pooled ROC ----------------
    fpr, tpr, _ = roc_curve(y, s)
    pooled_auc = roc_auc_score(y, s)
    thr_j, tpr_j, fpr_j, jstat = fs.youden(y, s)
    eer, thr_eer = fs.eer_point(y, s)

    ax1.plot([0, 1], [0, 1], ls=(0, (4, 3)), lw=0.9, color=fs.C_NEUTRAL,
             label="Chance (AUC = 0.500)")
    ax1.plot(fpr, tpr, color=fs.C_AUTH, lw=1.6,
             label="Pooled ROC (AUC = %.4f)" % pooled_auc)
    ax1.fill_between(fpr, tpr, alpha=0.10, color=fs.C_AUTH, lw=0)

    # Equal-error line and its intersection.
    ax1.plot([0, 1], [1, 0], ls=(0, (1, 2)), lw=0.8, color=fs.C_NEUTRAL,
             alpha=0.8)
    ax1.scatter([eer], [1 - eer], s=26, zorder=5, color=fs.C_FAKE,
                edgecolor="white", linewidth=0.7,
                label="EER = %.2f%%" % (100 * eer))
    ax1.scatter([fpr_j], [tpr_j], s=30, zorder=5, marker="D",
                color=fs.C_ACCENT, edgecolor="white", linewidth=0.7,
                label=r"Youden $J$: $\tau^*$ = %.4f" % thr_j)

    ax1.annotate("TPR %.2f%%\nFPR %.2f%%\n$J$ = %.4f"
                 % (100 * tpr_j, 100 * fpr_j, jstat),
                 xy=(fpr_j, tpr_j), xytext=(fpr_j + 0.20, tpr_j - 0.26),
                 fontsize=6.8, color=fs.C_NEUTRAL, ha="left",
                 arrowprops=dict(arrowstyle="-", lw=0.6, color=fs.C_NEUTRAL))

    ax1.set_xlabel("False positive rate")
    ax1.set_ylabel("True positive rate")
    ax1.set_title("(a) Pooled over 2,240 verification pairs", loc="left")
    ax1.set_xlim(-0.01, 1.0)
    ax1.set_ylim(0.0, 1.01)
    ax1.legend(loc="lower right")
    fs.despine(ax1)

    # ---------------- (b) per-fold ROC ----------------
    aucs = np.array([roc_auc_score(yy, ss) for _, yy, ss in folds])
    best = int(np.argmax(aucs))
    worst = int(np.argmin(aucs))

    ax2.plot([0, 1], [0, 1], ls=(0, (4, 3)), lw=0.9, color=fs.C_NEUTRAL)
    for k, (name, yy, ss) in enumerate(folds):
        f_, t_, _ = roc_curve(yy, ss)
        if k in (best, worst):
            continue
        ax2.plot(f_, t_, color=fs.C_FOLD, lw=0.8, alpha=0.85)

    for k, colour, style in ((best, fs.C_AUTH, "-"), (worst, fs.C_FAKE, "-")):
        name, yy, ss = folds[k]
        f_, t_, _ = roc_curve(yy, ss)
        ax2.plot(f_, t_, color=colour, lw=1.6, ls=style,
                 label="%s (AUC = %.3f)" % (name, aucs[k]))

    ax2.plot([], [], color=fs.C_FOLD, lw=0.8,
             label="Remaining 11 folds")
    ax2.set_xlabel("False positive rate")
    ax2.set_ylabel("True positive rate")
    ax2.set_title("(b) Per-subject folds: %.4f $\\pm$ %.4f AUC"
                  % (aucs.mean(), aucs.std()), loc="left")
    ax2.set_xlim(-0.01, 1.0)
    ax2.set_ylim(0.0, 1.01)
    ax2.legend(loc="lower right")
    fs.despine(ax2)

    fig.tight_layout(w_pad=1.6)
    fs.save(fig, "fig5_roc.png")

    print("    pooled AUC      %.6f" % pooled_auc)
    print("    per-fold AUC    %.6f +/- %.6f" % (aucs.mean(), aucs.std()))
    print("    Youden tau*     %.4f (TPR %.4f, FPR %.4f, J %.4f)"
          % (thr_j, tpr_j, fpr_j, jstat))
    print("    pooled EER      %.6f at tau %.4f" % (eer, thr_eer))


if __name__ == "__main__":
    main()
