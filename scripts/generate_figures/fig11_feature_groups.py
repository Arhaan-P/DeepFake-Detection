"""
Figure 11 -- Where the discriminative signal lives inside the 78-dimensional
feature vector.

Panel (a) gives each feature group's share of total attribution alongside its
share of the input dimensionality, which is the comparison that matters: joint
angles occupy 6 of 78 dimensions (7.7%) yet carry 14.9% of the attribution, so
they are the only group that is attended to disproportionately to its size.
Panel (b) breaks the angle group into its six constituent flexion angles.

Source: outputs/gradcam/aggregate/gradcam_results.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt

import figstyle as fs

# (json key, display name, dimensions, index range) -- ranges per the feature
# layout in utils/pose_extraction.py
GROUPS = [
    ("coords", "Normalised\ncoordinates", 36, "[0:36)"),
    ("velocities", "Frame-to-frame\nvelocities", 36, "[42:78)"),
    ("angles", "Joint\nangles", 6, "[36:42)"),
]

ANGLE_LABEL = {
    "L_Knee_Angle": "L Knee", "R_Knee_Angle": "R Knee",
    "L_Hip_Angle": "L Hip", "R_Hip_Angle": "R Hip",
    "L_Ankle_Angle": "L Ankle", "R_Ankle_Angle": "R Ankle",
}


def main():
    fs.apply_style()
    g = fs.load_gradcam()
    grp = g["group_importance"]
    ang = g["angle_importance"]
    total_dim = sum(d for _, _, d, _ in GROUPS)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.16, 2.85), gridspec_kw={"width_ratios": [1.15, 1.0]})

    # ---------------- (a) attribution share vs. dimensional share ----------
    names = [n for _, n, _, _ in GROUPS]
    attrib = np.array([grp[k] for k, _, _, _ in GROUPS]) * 100
    dims = np.array([d for _, _, d, _ in GROUPS], dtype=float)
    dimshare = dims / total_dim * 100

    x = np.arange(len(GROUPS))
    w = 0.36
    ax1.bar(x - w / 2, attrib, w, color=fs.C_AUTH, edgecolor="white",
            label="Share of attribution")
    ax1.bar(x + w / 2, dimshare, w, color="#c8d8e4", edgecolor="white",
            label="Share of input dimensions")

    for xi, (a, d, (_, _, nd, rng)) in enumerate(zip(attrib, dimshare, GROUPS)):
        ax1.text(xi - w / 2, a + 0.9, "%.1f%%" % a, ha="center", va="bottom",
                 fontsize=7, color=fs.C_NEUTRAL)
        ax1.text(xi + w / 2, d + 0.9, "%.1f%%" % d, ha="center", va="bottom",
                 fontsize=7, color=fs.C_NEUTRAL)
        ratio = a / d
        ax1.text(xi, -7.5, "%d dims %s\n%.2f$\\times$ per dim"
                 % (nd, rng, ratio), ha="center", va="top", fontsize=6.1,
                 color=fs.C_ACCENT if ratio > 1 else "#8c8c8c")

    ax1.set_xticks(x, labels=names)
    ax1.set_ylabel("Share (%)")
    ax1.set_ylim(0, 70)
    ax1.set_title("(a) Attribution share vs. dimensional share", loc="left")
    ax1.legend(loc="upper right")
    ax1.tick_params(axis="x", pad=26)
    fs.despine(ax1)

    # ---------------- (b) per-angle attribution ----------------
    keys = sorted(ang, key=lambda k: ang[k])
    vals = np.array([ang[k] for k in keys])
    ypos = np.arange(len(keys))

    ax2.barh(ypos, vals, height=0.66, color=fs.C_AUTH, edgecolor="white")
    for i, v in enumerate(vals):
        ax2.text(v + 0.014, i, "%.3f" % v, va="center", fontsize=6.6,
                 color=fs.C_NEUTRAL)

    ax2.set_yticks(ypos, labels=[ANGLE_LABEL.get(k, k) for k in keys])
    ax2.set_xlim(0, 1.16)
    ax2.set_xlabel("Normalised attribution")
    ax2.set_title("(b) Within the joint-angle group", loc="left")
    ax2.grid(axis="y", visible=False)
    fs.despine(ax2)

    fig.tight_layout(w_pad=1.3)
    fs.save(fig, "fig11_feature_groups.png")

    for (k, n, d, rng), a, ds in zip(GROUPS, attrib, dimshare):
        print("    %-12s attribution %5.2f%%  dims %2d (%5.2f%%)  ratio %.2f"
              % (k, a, d, ds, a / ds))
    for k in sorted(ang, key=lambda k: -ang[k]):
        print("    angle %-14s %.4f" % (k, ang[k]))


if __name__ == "__main__":
    main()
