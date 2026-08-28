"""
Figure 9 -- LOOCV ablation: does the temporal encoder stack earn its place?

Panel (a) reports ROC-AUC and accuracy per variant over the full
leave-one-subject-out protocol: 13 folds x 3 seeds = 39 observations each,
every variant measured on identical folds and seeds.

Panel (b) is the substantive content. Because all variants share folds and
seeds, each can be compared to the deployed model PAIRWISE, which removes fold
difficulty and initialization luck from the comparison. It plots the paired
change in ROC-AUC against the deployed baseline with 95% confidence intervals.
Every interval lies wholly below zero: adding the encoder stack costs accuracy,
and the cost is not explained by the parameters added.

The "Raw (deployed)" variant is an identity encoder -- it reproduces
models/full_pipeline.py exactly (133,058 parameters) and serves as the control.
"Hybrid only (no raw)" removes the raw per-timestep comparison entirely and
collapses to chance, showing which half of the architecture carries the signal.

Note: this module deliberately contains no backslash escapes. The shell used to
author it collapses them inside heredocs, which previously turned a mathtext
"times" macro into a literal tab. Line breaks in labels use chr(10).

Source: outputs/ablation/ablation_loocv_results.json
        (produced by scripts/evaluation/ablation_loocv.py)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt

import figstyle as fs

NL = chr(10)
TIMES = chr(215)          # multiplication sign, avoids a mathtext macro

BASELINE = "Raw (deployed)"
ORDER = [BASELINE, "Raw + CNN", "Raw + Transformer", "Raw + Hybrid",
         "Raw + BiLSTM", "Hybrid only (no raw)"]

SHORT = {
    "Raw (deployed)": ("Raw", "(deployed)"),
    "Raw + CNN": ("Raw", "+ CNN"),
    "Raw + Transformer": ("Raw", "+ Transf."),
    "Raw + Hybrid": ("Raw", "+ Hybrid"),
    "Raw + BiLSTM": ("Raw", "+ BiLSTM"),
    "Hybrid only (no raw)": ("Hybrid only", "(no raw)"),
}


def stacked(v):
    return NL.join(SHORT[v])


def flat(v):
    return " ".join(SHORT[v])


def main():
    fs.apply_style()
    meta, per_variant, legacy = fs.load_ablation_loocv()

    present = [v for v in ORDER if v in per_variant]
    base_auc, base_keys = fs.variant_metric(per_variant[BASELINE], "roc_auc")

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.16, 3.25), gridspec_kw={"width_ratios": [1.12, 1.0]})

    # ---------------- (a) AUC and accuracy per variant ----------------
    x = np.arange(len(present))
    width = 0.38

    aucs, auc_sd, accs, acc_sd, params = [], [], [], [], []
    for v in present:
        a, _ = fs.variant_metric(per_variant[v], "roc_auc")
        c, _ = fs.variant_metric(per_variant[v], "accuracy")
        aucs.append(100 * a.mean()); auc_sd.append(100 * a.std())
        accs.append(100 * c.mean()); acc_sd.append(100 * c.std())
        params.append(per_variant[v]["params"])

    # Control in the strong colour, everything else muted: the plate reads
    # "baseline vs. the things that did not beat it".
    cols_auc = [fs.C_AUTH if v == BASELINE else "#9ecae1" for v in present]
    cols_acc = [fs.C_ACCENT if v == BASELINE else "#f0cba0" for v in present]

    ax1.bar(x - width / 2, aucs, width, yerr=auc_sd, color=cols_auc,
            edgecolor="white", label="ROC-AUC",
            error_kw=dict(lw=0.7, capsize=2, ecolor=fs.C_NEUTRAL))
    ax1.bar(x + width / 2, accs, width, yerr=acc_sd, color=cols_acc,
            edgecolor="white", label="Accuracy",
            error_kw=dict(lw=0.7, capsize=2, ecolor=fs.C_NEUTRAL))

    # Value labels clear the top of the error bar, not the top of the bar.
    for xi, a, sd in zip(x, aucs, auc_sd):
        ax1.text(xi - width / 2, a + sd + 1.6, "%.1f" % a, ha="center",
                 va="bottom", fontsize=5.9, color=fs.C_NEUTRAL)

    ax1.axhline(50, color=fs.C_FAKE, lw=0.8, ls=":", zorder=1)
    # Parked to the right of the last bar group -- the only clear space on
    # this axis; at the left it sat on top of the baseline bars.
    ax1.set_xlim(-0.55, len(present) + 0.05)
    ax1.text(len(present) - 0.02, 51.5, "chance", fontsize=5.8,
             color=fs.C_FAKE, ha="right", va="bottom")

    ax1.set_xticks(x, labels=[stacked(v) for v in present], fontsize=6.0)
    ax1.set_ylabel("Value (%)")
    ax1.set_ylim(0, 112)
    ax1.set_yticks([0, 20, 40, 60, 80, 100])
    ax1.set_title("(a) LOOCV performance (13 folds " + TIMES + " 3 seeds)",
                  loc="left")
    ax1.legend(loc="lower left", ncol=2, columnspacing=0.9, fontsize=6.6)
    fs.despine(ax1)

    # ---------------- (b) paired change vs. the deployed model ----------------
    others = [v for v in present if v != BASELINE]
    deltas, half_ci, labels = [], [], []
    for v in others:
        a, keys = fs.variant_metric(per_variant[v], "roc_auc")
        assert keys == base_keys, "variants not measured on identical folds/seeds"
        d = 100 * (a - base_auc)
        # 95% CI on the paired mean difference (normal approx, n = 39).
        half_ci.append(1.96 * d.std(ddof=1) / np.sqrt(d.size))
        deltas.append(d.mean())
        labels.append(v)

    ypos = np.arange(len(labels))[::-1]

    ax2.barh(ypos, deltas, height=0.5, color=fs.C_FAKE, edgecolor="white",
             xerr=[half_ci, half_ci],
             error_kw=dict(lw=0.7, capsize=2, ecolor=fs.C_NEUTRAL))
    ax2.axvline(0, color=fs.C_NEUTRAL, lw=0.9, zorder=5)

    # Annotations sit to the RIGHT of zero, which is empty space: putting them
    # at the bar tip collided with the y-axis labels.
    for yi, d, v in zip(ypos, deltas, labels):
        extra = per_variant[v]["params"] - per_variant[BASELINE]["params"]
        ax2.text(2.0, yi, "%+.1f pts, %+dk par." % (d, round(extra / 1000)),
                 va="center", ha="left", fontsize=6.1, color=fs.C_NEUTRAL)

    lo = min(d - h for d, h in zip(deltas, half_ci))
    ax2.set_yticks(ypos, labels=[flat(v) for v in labels], fontsize=6.2)
    ax2.set_xlabel("Change in ROC-AUC vs. deployed (points)")
    ax2.set_xlim(lo * 1.12, 30)
    ax2.set_xticks([-40, -30, -20, -10, 0])
    ax2.set_title("(b) Paired change vs. deployed (95% CI)", loc="left")
    ax2.grid(axis="y", visible=False)
    fs.despine(ax2)

    fig.tight_layout(w_pad=1.6)
    fs.save(fig, "fig9_ablation.png")

    # ---- console summary, so the plate's claims are checkable ----
    print("    folds=%s seeds=%s  (n=%d obs per variant)"
          % (meta["folds_completed"], meta["seeds"], base_auc.size))
    for v, a, s, p in zip(present, aucs, auc_sd, params):
        print("    %-22s %7d par  AUC %.2f +/- %.2f" % (v, p, a, s))
    print("    paired deltas vs %s:" % BASELINE)
    for v, d, h in zip(labels, deltas, half_ci):
        print("      %-22s %+6.2f  95%% CI [%+.2f, %+.2f]" % (v, d, d - h, d + h))
    if BASELINE in legacy:
        lg = legacy[BASELINE]["aggregate"]
        print("    legacy-normalization arm: AUC %.2f +/- %.2f (seed 0 only)"
              % (100 * lg["roc_auc"], 100 * lg["roc_auc_std"]))


if __name__ == "__main__":
    main()
