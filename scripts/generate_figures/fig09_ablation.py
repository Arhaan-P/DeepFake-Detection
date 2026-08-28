"""
Figure 9 -- Architectural ablation over the four trained variants.

Panel (a) reports the measured validation metrics for each variant. Panel (b)
plots accuracy against parameter count and separates, for each variant, the
parameters that lie on the verification decision path from those that do not.

That separation is the substantive content of this plate. All four variants in
scripts/evaluation/ablation_study.py compute their logits through an identical
diff_conv -> diff_classifier head applied to the raw per-timestep feature
difference; the named CNN / BiLSTM / Transformer branch contributes only an
auxiliary embedding that never reaches the logits and receives no gradient from
the cross-entropy objective. The parameter counts in panel (b) are measured by
instantiating each variant, not copied from the results file, so the claim is
checkable by running this script.

Sources: outputs/ablation/ablation_results.json
         scripts/evaluation/ablation_study.py (model definitions)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt

import figstyle as fs

ORDER = ["CNN-Only", "LSTM-Only", "Transformer-Only", "Full Hybrid"]
SHOW = [("accuracy", "Accuracy"), ("f1", "F1"), ("auc", "AUC"),
        ("precision", "Precision"), ("recall", "Recall")]


def measure_active_params():
    """
    Instantiate each ablation variant and count (total, decision-path) params.

    Returns None if the model packages cannot be imported, in which case panel
    (b) falls back to the parameter totals recorded in the results JSON and
    omits the decision-path split.
    """
    try:
        sys.path.insert(0, str(fs.REPO))
        import types
        import torch  # noqa: F401

        # ablation_study imports the data loader at module scope, which pulls in
        # MediaPipe. We only need the nn.Module definitions, so stub the loader
        # rather than requiring a working pose-extraction stack just to count
        # parameters.
        for mod, attrs in (("utils.data_loader", ["create_data_loaders"]),
                           ("utils.logger", ["setup_logging", "close_logging"])):
            if mod not in sys.modules:
                stub = types.ModuleType(mod)
                for a in attrs:
                    setattr(stub, a, lambda *args, **kw: None)
                sys.modules[mod] = stub

        from scripts.evaluation.ablation_study import (
            CNNOnlyModel, LSTMOnlyModel, TransformerOnlyModel, _create_full_model)
    except Exception as exc:  # pragma: no cover - environment dependent
        print("    (could not import ablation variants: %s)" % exc)
        return None

    builders = {
        "CNN-Only": lambda: CNNOnlyModel(
            input_dim=78, hidden_dims=(64, 128), output_dim=128,
            verification_hidden=64, dropout=0.1),
        "LSTM-Only": lambda: LSTMOnlyModel(
            input_dim=78, lstm_hidden=64, lstm_layers=1,
            verification_hidden=64, dropout=0.1),
        "Transformer-Only": lambda: TransformerOnlyModel(
            input_dim=78, d_model=128, nhead=4, num_layers=2,
            verification_hidden=64, dropout=0.1),
        "Full Hybrid": lambda: _create_full_model(0.1),
    }

    out = {}
    for name, build in builders.items():
        m = build()
        total = sum(p.numel() for p in m.parameters() if p.requires_grad)
        active = (sum(p.numel() for p in m.diff_conv.parameters())
                  + sum(p.numel() for p in m.diff_classifier.parameters()))
        out[name] = (total, active)
    return out


def main():
    fs.apply_style()
    ab = fs.load_ablation()
    measured = measure_active_params()

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.16, 3.0), gridspec_kw={"width_ratios": [1.35, 1.0]})

    # ---------------- (a) metric bars ----------------
    x = np.arange(len(SHOW))
    width = 0.20
    shades = ["#a6cee3", "#6aa5cc", fs.C_AUTH, "#0d4a73"]

    for i, variant in enumerate(ORDER):
        vals = [ab[variant][k] for k, _ in SHOW]
        pos = x + (i - 1.5) * width
        ax1.bar(pos, vals, width * 0.92, label=variant, color=shades[i],
                edgecolor="white")
        for px, v in zip(pos, vals):
            ax1.text(px, v + 0.35, "%.1f" % v, ha="center", va="bottom",
                     fontsize=5.4, rotation=90, color=fs.C_NEUTRAL)

    allv = np.array([[ab[v][k] for k, _ in SHOW] for v in ORDER])
    ax1.set_xticks(x, labels=[lab for _, lab in SHOW])
    ax1.set_ylabel("Value (%)")
    ax1.set_ylim(80, 103)
    ax1.set_yticks([80, 85, 90, 95, 100])
    ax1.set_title("(a) Validation metrics by variant", loc="left")
    ax1.legend(loc="upper center", ncol=2, columnspacing=0.9)
    fs.despine(ax1)

    # ---------------- (b) capacity vs. accuracy ----------------
    totals = np.array([measured[v][0] if measured else ab[v]["params"]
                       for v in ORDER], dtype=float)
    accs = np.array([ab[v]["accuracy"] for v in ORDER])
    ypos = np.arange(len(ORDER))

    if measured:
        active = np.array([measured[v][1] for v in ORDER], dtype=float)
        inert = totals - active
        ax2.barh(ypos, active / 1e3, height=0.55, color=fs.C_AUTH,
                 edgecolor="white", label="On decision path")
        ax2.barh(ypos, inert / 1e3, height=0.55, left=active / 1e3,
                 color="#dbe6ee", edgecolor="white",
                 label="Not on decision path")
        for i, (a, t) in enumerate(zip(active, totals)):
            ax2.text(t / 1e3 + 12, i, "%.1f%% acc" % accs[i], va="center",
                     fontsize=6.6, color=fs.C_NEUTRAL)
        ax2.text(active[0] / 1e3 / 2, -0.92,
                 "%.1fk shared by all four" % (active[0] / 1e3),
                 ha="center", va="center", fontsize=6.4, color=fs.C_AUTH)
    else:
        ax2.barh(ypos, totals / 1e3, height=0.55, color=fs.C_AUTH,
                 edgecolor="white", label="Total parameters")
        for i, t in enumerate(totals):
            ax2.text(t / 1e3 + 12, i, "%.1f%% acc" % accs[i], va="center",
                     fontsize=6.6, color=fs.C_NEUTRAL)

    ax2.set_yticks(ypos, labels=ORDER)
    ax2.set_xlabel("Trainable parameters (thousands)")
    ax2.set_xlim(0, max(totals) / 1e3 * 1.32)
    ax2.set_ylim(-1.3, len(ORDER) - 0.4)
    ax2.set_title("(b) Where the capacity sits", loc="left")
    ax2.legend(loc="lower right")
    ax2.grid(axis="y", visible=False)
    fs.despine(ax2)

    fig.tight_layout(w_pad=1.4)
    fs.save(fig, "fig9_ablation.png")

    spread = allv[:, 0].max() - allv[:, 0].min()
    print("    accuracy spread across variants: %.2f points" % spread)
    if measured:
        for v in ORDER:
            t, a = measured[v]
            print("    %-17s total %7d  decision-path %6d (%.1f%%)"
                  % (v, t, a, 100 * a / t))


if __name__ == "__main__":
    main()
