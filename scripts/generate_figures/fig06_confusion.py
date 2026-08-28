"""
Figure 6 -- Pooled confusion matrices over all 2,240 LOOCV verification pairs.

Two operating points are shown side by side because they answer different
questions, and because the two are frequently conflated in the project's own
prose:

(a) tau = 0.50, the argmax decision the network makes natively. This is the
    operating point at which the per-fold accuracy / F1 / precision / recall
    in loocv_results.json were computed.
(b) tau* = Youden-J optimal, recomputed here from the pooled ROC. This is the
    threshold the deployed inference script applies.

Source: outputs/evaluation/loocv/loocv_results.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import figstyle as fs
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

CLASSES = ["Authentic", "Deepfake"]


def panel(ax, y, s, tau, title):
    """Render one confusion matrix. Rows = actual, columns = predicted."""
    pred = (s >= tau).astype(int)
    # sklearn orders labels ascending (0 = deepfake, 1 = authentic); flip so the
    # authentic class reads first, matching how the results tables are written.
    cm = confusion_matrix(y, pred, labels=[1, 0])
    total = cm.sum()

    cmap = LinearSegmentedColormap.from_list(
        "auth", ["#ffffff", "#cfe2f0", "#6aa5cc", fs.C_AUTH]
    )
    ax.imshow(cm / total, cmap=cmap, vmin=0, vmax=0.5)

    tags = [["TP", "FN"], ["FP", "TN"]]
    for i in range(2):
        for j in range(2):
            n = cm[i, j]
            frac = n / total
            colour = "white" if frac > 0.28 else fs.C_NEUTRAL
            ax.text(
                j,
                i - 0.13,
                "%d" % n,
                ha="center",
                va="center",
                fontsize=15,
                color=colour,
            )
            ax.text(
                j,
                i + 0.17,
                "%s  %.1f%%" % (tags[i][j], 100 * frac),
                ha="center",
                va="center",
                fontsize=7.5,
                color=colour,
            )

    ax.set_xticks([0, 1], labels=["Predicted\nauthentic", "Predicted\ndeepfake"])
    ax.set_yticks([0, 1], labels=["Actual\nauthentic", "Actual\ndeepfake"])
    ax.set_title(title, loc="left")
    ax.grid(False)
    ax.tick_params(length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    m = dict(
        acc=accuracy_score(y, pred),
        f1=f1_score(y, pred),
        prec=precision_score(y, pred),
        rec=recall_score(y, pred),
    )
    ax.set_xlabel(
        "Acc %.2f%%   F1 %.2f%%   P %.2f%%   R %.2f%%"
        % (100 * m["acc"], 100 * m["f1"], 100 * m["prec"], 100 * m["rec"]),
        fontsize=7.4,
        labelpad=7,
    )
    return cm, m


def main():
    fs.apply_style()
    agg, per_fold, y, s = fs.load_loocv()
    thr_j, _, _, _ = fs.youden(y, s)

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.15))
    cm_a, m_a = panel(axes[0], y, s, 0.50, r"(a) Native decision, $\tau = 0.50$")
    cm_b, m_b = panel(
        axes[1], y, s, thr_j, r"(b) Youden-optimal, $\tau^* = %.4f$" % thr_j
    )

    fig.tight_layout(w_pad=2.4)
    fs.save(fig, "fig6_confusion.png")

    for tag, cm, m in (("tau=0.50", cm_a, m_a), ("tau*=%.4f" % thr_j, cm_b, m_b)):
        print(
            "    %-14s TP %4d  FN %4d  FP %4d  TN %4d | acc %.4f f1 %.4f"
            % (tag, cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1], m["acc"], m["f1"])
        )


if __name__ == "__main__":
    main()
