"""
Figure 10 -- Per-joint attribution over the 12 gait keypoints.

Panel (a) overlays the normalised attribution score on a skeleton whose
geometry is the mean of all 13 enrolment signatures (13 identities x 60
timesteps of hip-centred coordinates), so the layout is data-derived rather
than drawn by hand. Panel (b) gives the same 12 values as a ranked bar chart,
which is what the reader needs in order to quote a number.

Attribution is gradient-times-input on the 78-dimensional feature vector,
backpropagated from the verification logit, averaged over 26 samples
(2 per identity).

Sources: outputs/gradcam/aggregate/gradcam_results.json
         data/gait_features/enrolled_identities.pkl (skeleton geometry only)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import figstyle as fs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize


def main():
    fs.apply_style()
    g = fs.load_gradcam()
    imp = g["joint_importance"]
    n_samples = g.get("n_samples", None)

    vals = np.array([imp[j] for j in fs.JOINTS])
    xy, provenance = fs.mean_skeleton()

    norm = Normalize(vmin=vals.min() * 0.92, vmax=vals.max())
    cmap = cm.get_cmap(fs.SEQ_CMAP)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.16, 3.60), gridspec_kw={"width_ratios": [0.92, 1.08]}
    )

    # ---------------- (a) skeleton overlay ----------------
    for a, b in fs.BONES:
        ax1.plot(
            [xy[a, 0], xy[b, 0]],
            [xy[a, 1], xy[b, 1]],
            color="#b8b8b8",
            lw=1.8,
            zorder=1,
            solid_capstyle="round",
        )

    # Glow halo scaled by attribution, then the marker itself.
    ax1.scatter(
        xy[:, 0],
        xy[:, 1],
        s=110 + 1000 * (vals - vals.min()),
        c=[cmap(norm(v)) for v in vals],
        alpha=0.20,
        linewidth=0,
        zorder=2,
    )
    ax1.scatter(
        xy[:, 0],
        xy[:, 1],
        s=120,
        c=vals,
        cmap=fs.SEQ_CMAP,
        norm=norm,
        edgecolor=fs.C_NEUTRAL,
        linewidth=0.7,
        zorder=3,
    )

    order = np.argsort(-vals)
    rank = {int(k): r + 1 for r, k in enumerate(order)}
    for i in order[:3]:
        ax1.text(
            xy[i, 0],
            xy[i, 1],
            str(rank[int(i)]),
            ha="center",
            va="center",
            fontsize=5.8,
            color="white",
            zorder=4,
            weight="bold",
        )

    # The distal cluster (ankle / heel / foot) spans barely 0.03 in y, so
    # labels anchored at their joint would collide. Anchor each side's labels
    # at a fixed x and push them apart vertically, keeping a leader line back
    # to the true joint position.
    span_y = xy[:, 1].max() - xy[:, 1].min()
    min_gap = span_y * 0.105
    label_x = {1: xy[:, 0].max() + 0.085, -1: xy[:, 0].min() - 0.085}

    for side in (1, -1):
        idx = [i for i in range(len(fs.JOINTS)) if (1 if xy[i, 0] >= 0 else -1) == side]
        idx.sort(key=lambda i: xy[i, 1])
        ys = [float(xy[i, 1]) for i in idx]
        # single upward pass, then re-centre on the original span
        for k in range(1, len(ys)):
            if ys[k] - ys[k - 1] < min_gap:
                ys[k] = ys[k - 1] + min_gap
        shift = xy[idx, 1].mean() - np.mean(ys)
        ys = [v + shift for v in ys]

        for i, ly in zip(idx, ys):
            ax1.annotate(
                "%s  %.2f" % (fs.JOINT_LABEL[fs.JOINTS[i]], vals[i]),
                xy=(xy[i, 0], xy[i, 1]),
                xytext=(label_x[side], ly),
                ha="left" if side > 0 else "right",
                va="center",
                fontsize=6.2,
                color=fs.C_NEUTRAL,
                arrowprops=dict(
                    arrowstyle="-",
                    lw=0.45,
                    color="#c4c4c4",
                    shrinkA=0,
                    shrinkB=4,
                    connectionstyle="arc3,rad=0.0",
                ),
            )

    ax1.set_xlim(xy[:, 0].min() - 0.36, xy[:, 0].max() + 0.36)
    ax1.set_ylim(xy[:, 1].min() - 0.06, xy[:, 1].max() + 0.06)
    ax1.set_aspect("equal")
    ax1.axis("off")
    ax1.set_title("(a) Attribution on the mean enrolled skeleton", loc="left", pad=8)
    ax1.text(
        0.5,
        -0.01,
        "viewer-facing: subject's left appears at right",
        transform=ax1.transAxes,
        ha="center",
        va="top",
        fontsize=6,
        color="#8c8c8c",
    )

    cb = fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=fs.SEQ_CMAP),
        ax=ax1,
        orientation="horizontal",
        fraction=0.045,
        pad=0.045,
        aspect=28,
    )
    cb.set_label("Normalised attribution", fontsize=6.8)
    cb.ax.tick_params(labelsize=6.2, length=2)
    cb.outline.set_linewidth(0.5)

    # ---------------- (b) ranked bars ----------------
    ypos = np.arange(len(order))
    sorted_vals = vals[order][::-1]
    sorted_names = [fs.JOINT_LABEL[fs.JOINTS[i]] for i in order][::-1]

    ax2.barh(
        ypos,
        sorted_vals,
        height=0.68,
        color=[cmap(norm(v)) for v in sorted_vals],
        edgecolor=fs.C_NEUTRAL,
        linewidth=0.4,
    )
    for i, v in enumerate(sorted_vals):
        ax2.text(
            v + 0.012, i, "%.3f" % v, va="center", fontsize=6.4, color=fs.C_NEUTRAL
        )

    ax2.set_yticks(ypos, labels=sorted_names)
    ax2.set_xlim(0, 1.13)
    ax2.set_xlabel("Normalised attribution")
    title = "(b) Ranked joint attribution"
    if n_samples:
        title += " ($n$ = %d samples)" % n_samples
    ax2.set_title(title, loc="left")
    ax2.grid(axis="y", visible=False)
    fs.despine(ax2)

    fig.tight_layout(w_pad=0.6)
    fs.save(fig, "fig10_joint_importance.png")

    print("    skeleton geometry from: %s" % provenance)
    for r, k in enumerate(order, start=1):
        print("    %2d. %-12s %.4f" % (r, fs.JOINTS[k], vals[k]))


if __name__ == "__main__":
    main()
