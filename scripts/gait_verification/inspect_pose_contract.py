"""Save a skeleton-only mapping diagnostic; does not export RGB participant images."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gait_verification.pose import load_pose
from gait_verification.skeleton_contracts import coco_arrays, COCO17, GAITPT_ORDER


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seq = load_pose(args.cache)
    xy, _, _ = coco_arrays(seq)
    points = xy[len(xy) // 2]
    fig, axes = plt.subplots(1, 2, figsize=(10, 9))
    for ax in axes:
        for a, b in [
            (5, 6),
            (5, 7),
            (7, 9),
            (6, 8),
            (8, 10),
            (5, 11),
            (6, 12),
            (11, 12),
            (11, 13),
            (13, 15),
            (12, 14),
            (14, 16),
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),
        ]:
            ax.plot(points[[a, b], 0], points[[a, b], 1], color="gray")
        ax.scatter(points[:, 0], points[:, 1])
        ax.invert_yaxis()
        ax.set_aspect("equal")
    for i, name in enumerate(COCO17):
        axes[0].annotate(f"{i} {name}", points[i], fontsize=7)
    for target, source in enumerate(GAITPT_ORDER):
        axes[1].annotate(
            f"{target}←{source}",
            points[source],
            xytext=(3, target % 2 * 10),
            textcoords="offset points",
            fontsize=8,
        )
    axes[0].set_title("Native-to-COCO anatomical mapping")
    axes[1].set_title("GaitPT order (nose repeated at 0 and 3)")
    fig.tight_layout()
    with open(args.output, "xb") as f:
        fig.savefig(f, format="png")
    plt.close(fig)


if __name__ == "__main__":
    main()
