"""
Shared plotting style and data-loading helpers for all paper figures.
============================================================================
Every number rendered by the scripts in this directory is read from a JSON
artefact produced by the evaluation pipeline, or from the enrolled-identity
pickle. Nothing is hard-coded from prose. If a figure cannot find its source
artefact it fails loudly rather than falling back to a literal.

Source artefacts
----------------
  outputs/evaluation/loocv/loocv_results.json     LOOCV per-fold + pooled scores
  outputs/ablation/ablation_results.json          4-variant ablation
  outputs/gradcam/aggregate/gradcam_results.json  attribution analysis
  outputs/model_config.json                       as-trained model config
  data/gait_features/enrolled_identities.pkl      mean skeleton geometry
"""

from pathlib import Path
import json
import pickle

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[2]
FIGDIR = REPO / "figures"
FIGDIR.mkdir(exist_ok=True)

LOOCV_JSON = REPO / "outputs" / "evaluation" / "loocv" / "loocv_results.json"
ABLATION_JSON = REPO / "outputs" / "ablation" / "ablation_results.json"
ABLATION_LOOCV_JSON = REPO / "outputs" / "ablation" / "ablation_loocv_results.json"
GRADCAM_JSON = REPO / "outputs" / "gradcam" / "aggregate" / "gradcam_results.json"
CONFIG_JSON = REPO / "outputs" / "model_config.json"
ENROLLED_PKL = REPO / "data" / "gait_features" / "enrolled_identities.pkl"

DPI = 300

# ---------------------------------------------------------------------------
# Style -- a single palette and typography shared by every figure so the plates
# read as one system when set two-column in IEEEtran.
# ---------------------------------------------------------------------------

C_AUTH = "#1b6ca8"      # authentic / positive class
C_FAKE = "#c0504d"      # deepfake / negative class
C_ACCENT = "#e08214"    # operating points, thresholds, highlights
C_NEUTRAL = "#4d4d4d"
C_GRID = "#d9d9d9"
C_FOLD = "#9ecae1"      # individual LOOCV folds
SEQ_CMAP = "YlOrRd"     # attribution heatmaps


def apply_style():
    """Install the shared rcParams. Call once at the top of every script."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "axes.linewidth": 0.7,
        "axes.edgecolor": C_NEUTRAL,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": C_GRID,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.fontsize": 7.5,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": C_GRID,
        "legend.borderpad": 0.4,
        "lines.linewidth": 1.3,
        "patch.linewidth": 0.6,
    })


def despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def save(fig, name):
    """Write a figure to figures/<name> at publication resolution."""
    out = FIGDIR / name
    fig.savefig(out)
    plt.close(fig)
    print("  wrote " + str(out.relative_to(REPO)))
    return out


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _require(path):
    if not path.exists():
        raise FileNotFoundError(
            "Required results artefact missing: " + str(path) + "\n"
            "Figures are generated from real evaluation output only; "
            "re-run the evaluation pipeline before generating figures."
        )
    return path


SUBJECT_MAP_JSON = REPO / "data" / "subject_map.json"

# Subjects are anonymised for publication: real enrolment names are mapped to
# "Subject N" in the order they appear in the source JSON (already
# alphabetical by original name), so the mapping is stable across regens.
# The mapping itself is not committed to git -- it is the one artefact that
# could de-anonymise the paper's "Subject N" labels, so it lives only in the
# local, gitignored data/ tree (see data/subject_map.json).
def _load_subject_anon():
    if not SUBJECT_MAP_JSON.exists():
        raise FileNotFoundError(
            "Subject anonymisation map missing: " + str(SUBJECT_MAP_JSON) + "\n"
            "This file is intentionally gitignored (it maps real enrolment "
            "names to their published 'Subject N' labels). Recreate it "
            "locally from your enrolment records before generating figures."
        )
    with open(SUBJECT_MAP_JSON) as fh:
        return json.load(fh)


_SUBJECT_ANON = None  # lazily loaded -- only figures that render subject labels need it


def load_loocv():
    """Return (aggregate, per_fold, pooled_labels, pooled_scores)."""
    global _SUBJECT_ANON
    if _SUBJECT_ANON is None:
        _SUBJECT_ANON = _load_subject_anon()
    with open(_require(LOOCV_JSON)) as fh:
        d = json.load(fh)
    y = np.asarray(d["pooled_labels"], dtype=float).ravel()
    s = np.asarray(d["pooled_scores"], dtype=float).ravel()
    for f in d["per_fold"]:
        f["test_person"] = _SUBJECT_ANON.get(f["test_person"], f["test_person"])
    return d["aggregate"], d["per_fold"], y, s


def fold_slices(per_fold, y, s):
    """
    Recover each fold's (labels, scores) from the pooled arrays.

    The pooled arrays are written fold-by-fold in the same order as `per_fold`,
    so consecutive slices of length n_samples reconstruct each fold exactly.
    Verified: slicing this way reproduces every reported per-fold ROC-AUC to
    within 1e-6, which is asserted below so the assumption cannot rot silently.
    """
    from sklearn.metrics import roc_auc_score

    out, i = [], 0
    for f in per_fold:
        n = f["n_samples"]
        yy, ss = y[i:i + n], s[i:i + n]
        i += n
        got = roc_auc_score(yy, ss)
        assert abs(got - f["roc_auc"]) < 1e-6, (
            "fold " + str(f["test_person"]) + ": sliced AUC differs from "
            "reported value; pooled array ordering has changed"
        )
        out.append((f["test_person"], yy, ss))
    assert i == len(y), "consumed " + str(i) + " of " + str(len(y)) + " samples"
    return out


def load_ablation():
    with open(_require(ABLATION_JSON)) as fh:
        return json.load(fh)


def load_ablation_loocv():
    """
    Return (meta, per_variant) for the definitive LOOCV ablation.

    per_variant maps variant name -> dict with, for the norm="train" arm,
    arrays of per-(fold, seed) metrics. Every variant is measured on the same
    folds and the same seeds, so metrics can be compared pairwise.
    """
    with open(_require(ABLATION_LOOCV_JSON)) as fh:
        d = json.load(fh)

    out = {}
    for key, r in d["results"].items():
        if r["norm"] != "train":
            continue
        v = r["variant"]
        rec = out.setdefault(v, {"params": r["aggregate"]["params"],
                                 "obs": {}})
        for fold, fm in r["per_fold"].items():
            rec["obs"][(fold, r["seed"])] = fm

    # legacy-normalization arm, kept separate: it quantifies the protocol
    # choice in scripts/evaluation/evaluate.py, not a model difference.
    legacy = {}
    for key, r in d["results"].items():
        if r["norm"] == "legacy":
            legacy[r["variant"]] = r
    return d["_meta"], out, legacy


def variant_metric(rec, metric):
    """Ordered array of one metric across all (fold, seed) observations."""
    keys = sorted(rec["obs"])
    return np.array([rec["obs"][k][metric] for k in keys], dtype=float), keys


def load_gradcam():
    with open(_require(GRADCAM_JSON)) as fh:
        return json.load(fh)


def load_config():
    with open(_require(CONFIG_JSON)) as fh:
        return json.load(fh)


def youden(y, s):
    """Youden-J optimal threshold and its operating point."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(y, s)
    k = int(np.argmax(tpr - fpr))
    return float(thr[k]), float(tpr[k]), float(fpr[k]), float(tpr[k] - fpr[k])


def eer_point(y, s):
    """Equal error rate and the threshold at which it is attained."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(y, s)
    fnr = 1.0 - tpr
    k = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[k] + fnr[k]) / 2.0), float(thr[k])


# ---------------------------------------------------------------------------
# Skeleton geometry -- the 12 gait keypoints, in selection order
# ---------------------------------------------------------------------------

JOINTS = ["L_Shoulder", "R_Shoulder", "L_Hip", "R_Hip", "L_Knee", "R_Knee",
          "L_Ankle", "R_Ankle", "L_Heel", "R_Heel", "L_Foot", "R_Foot"]

JOINT_LABEL = {
    "L_Shoulder": "L Shoulder", "R_Shoulder": "R Shoulder",
    "L_Hip": "L Hip", "R_Hip": "R Hip",
    "L_Knee": "L Knee", "R_Knee": "R Knee",
    "L_Ankle": "L Ankle", "R_Ankle": "R Ankle",
    "L_Heel": "L Heel", "R_Heel": "R Heel",
    "L_Foot": "L Foot", "R_Foot": "R Foot",
}

# Kinematic chain over indices into JOINTS.
BONES = [(0, 1), (0, 2), (1, 3), (2, 3),
         (2, 4), (4, 6), (6, 8), (8, 10),
         (3, 5), (5, 7), (7, 9), (9, 11)]


def mean_skeleton():
    """
    Mean hip-centred 2D skeleton, averaged over all 13 enrolled identities and
    all 60 timesteps of their enrolment signatures. Returns (12, 2) in plot
    coordinates (y flipped, since MediaPipe y grows downward).

    Falls back to a canonical anatomical layout if the enrolment pickle is not
    present, so the attribution figure can still be produced from a clean
    checkout that ships results but not the multi-hundred-MB feature caches.
    """
    if ENROLLED_PKL.exists():
        with open(ENROLLED_PKL, "rb") as fh:
            enrolled = pickle.load(fh)
        stack = np.stack([
            np.asarray(v["avg_normalized_coords"], dtype=float)
            for v in enrolled.values()
        ])                                    # (n_id, 60, 12, 3)
        xy = stack.mean(axis=(0, 1))[:, :2]   # (12, 2)
        xy = np.column_stack([xy[:, 0], -xy[:, 1]])
        # Widen horizontally: a frontal-view temporal average collapses lateral
        # spread, which would make the bones overlap. Vertical (proximo-distal)
        # structure -- the axis that actually carries joint identity -- is left
        # untouched.
        xy[:, 0] *= 3.0
        return xy, "enrolment signatures (13 identities x 60 frames)"

    canonical = np.array([
        [-0.11, 0.46], [0.11, 0.46], [-0.07, 0.00], [0.07, 0.00],
        [-0.08, -0.42], [0.08, -0.42], [-0.08, -0.82], [0.08, -0.82],
        [-0.10, -0.88], [0.10, -0.88], [-0.04, -0.95], [0.04, -0.95],
    ])
    return canonical, "canonical anatomical layout (enrolment cache absent)"
