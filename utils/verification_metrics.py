"""
Verification Metrics, Calibration and Statistics (plan Section 9)
================================================================
Numpy-only (scipy used for exact p-values when installed). Everything the
roadmap's evaluation checklist asks for:

  * threshold-free separability   ROC-AUC, EER, DET points
  * operating points              Youden-optimal tau, FAR/FRR at tau,
                                  TPR at fixed FPR (1%, 5%, 10%)
  * calibration                   ECE, MCE, Brier score, reliability bins,
                                  Platt scaling cross-fitted over subjects so
                                  a held-out subject never calibrates itself
  * uncertainty                   subject-level bootstrap confidence intervals
  * comparison                    paired t-test and Wilcoxon signed-rank

Author: DeepFake Detection Project
"""

import math
from collections.abc import Sequence
from typing import Dict, Optional

import numpy as np

# ============================================================
# ROC / EER
# ============================================================


def roc_curve(labels: np.ndarray, scores: np.ndarray):
    """(fpr, tpr, thresholds), thresholds descending, first = +inf."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    s, y = scores[order], labels[order]
    distinct = np.where(np.diff(s))[0]
    idx = np.r_[distinct, len(s) - 1]
    tps = np.cumsum(y)[idx]
    fps = (idx + 1) - tps
    p, n = max(y.sum(), 1), max(len(y) - y.sum(), 1)
    tpr = np.r_[0.0, tps / p]
    fpr = np.r_[0.0, fps / n]
    thr = np.r_[np.inf, s[idx]]
    return fpr, tpr, thr


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney AUC with tie correction."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sorted_v = allv[order]
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def eer(labels: np.ndarray, scores: np.ndarray):
    """(EER, threshold). Interpolates the crossing of FAR and FRR."""
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1 - tpr
    k = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[k] + fnr[k]) / 2), float(thr[k])


def youden_threshold(labels, scores):
    fpr, tpr, thr = roc_curve(labels, scores)
    k = int(np.argmax(tpr - fpr))
    return float(thr[k]), float(tpr[k]), float(fpr[k])


def tpr_at_fpr(labels, scores, target: float) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    ok = fpr <= target
    return float(tpr[ok].max()) if ok.any() else 0.0


def rates_at(labels, scores, tau: float) -> Dict[str, float]:
    labels = np.asarray(labels).astype(int)
    pred = (np.asarray(scores) >= tau).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {
        "threshold": float(tau),
        "far": fp / max(fp + tn, 1),
        "frr": fn / max(fn + tp, 1),
        "accuracy": (tp + tn) / max(len(labels), 1),
        "balanced_accuracy": 0.5 * (rec + tn / max(tn + fp, 1)),
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / max(prec + rec, 1e-12),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def summarize_scores(labels, scores) -> Dict[str, float]:
    """Threshold-free metrics + operating points for one score set."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) < 2:
        return {"n": len(labels)}
    e, e_thr = eer(labels, scores)
    tau, _, _ = youden_threshold(labels, scores)
    out = {
        "n": len(labels),
        "n_genuine": int(labels.sum()),
        "n_impostor": int((1 - labels).sum()),
        "roc_auc": roc_auc(labels, scores),
        "eer": e,
        "eer_threshold": e_thr,
        "tpr_at_fpr_1": tpr_at_fpr(labels, scores, 0.01),
        "tpr_at_fpr_5": tpr_at_fpr(labels, scores, 0.05),
        "tpr_at_fpr_10": tpr_at_fpr(labels, scores, 0.10),
        "youden_threshold": tau,
    }
    out["at_0.5"] = rates_at(labels, scores, 0.5)
    out["at_youden"] = rates_at(labels, scores, tau)
    return out


# ============================================================
# Calibration
# ============================================================


def reliability(labels, probs, n_bins: int = 10):
    """Equal-width reliability bins -> list of dicts (conf, acc, count)."""
    labels = np.asarray(labels).astype(float)
    probs = np.clip(np.asarray(probs, dtype=float), 0, 1)
    edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    for k in range(n_bins):
        m = (probs >= edges[k]) & (
            (probs < edges[k + 1]) if k < n_bins - 1 else (probs <= 1)
        )
        if m.any():
            bins.append(
                {
                    "lo": float(edges[k]),
                    "hi": float(edges[k + 1]),
                    "mean_prob": float(probs[m].mean()),
                    "frac_positive": float(labels[m].mean()),
                    "count": int(m.sum()),
                }
            )
    return bins


def calibration_summary(labels, probs, n_bins: int = 10) -> Dict[str, float]:
    labels = np.asarray(labels).astype(float)
    probs = np.clip(np.asarray(probs, dtype=float), 0, 1)
    bins = reliability(labels, probs, n_bins)
    n = len(labels)
    gaps = [abs(b["mean_prob"] - b["frac_positive"]) for b in bins]
    return {
        "ece": float(sum(g * b["count"] / n for g, b in zip(gaps, bins))),
        "mce": float(max(gaps)) if gaps else float("nan"),
        "brier": float(np.mean((probs - labels) ** 2)),
        "nll": float(
            -np.mean(
                labels * np.log(np.clip(probs, 1e-7, 1))
                + (1 - labels) * np.log(np.clip(1 - probs, 1e-7, 1))
            )
        ),
        "bins": bins,
    }


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_platt(labels, probs, iters: int = 100, l2: float = 1e-3):
    """Logistic regression on logit(prob) by Newton's method -> (a, b)."""
    x = _logit(probs)
    y = np.asarray(labels, dtype=float)
    a, b = 1.0, 0.0
    for _ in range(iters):
        z = np.clip(a * x + b, -50, 50)
        p = 1 / (1 + np.exp(-z))
        w = p * (1 - p) + 1e-9
        g = np.array([np.sum((p - y) * x) + l2 * a, np.sum(p - y)])
        h = np.array(
            [[np.sum(w * x * x) + l2, np.sum(w * x)], [np.sum(w * x), np.sum(w)]]
        )
        step = np.linalg.solve(h, g)
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-8:
            break
    return float(a), float(b)


def apply_platt(probs, ab):
    a, b = ab
    return 1 / (1 + np.exp(-np.clip(a * _logit(probs) + b, -50, 50)))


def crossfit_platt(labels, probs, groups) -> np.ndarray:
    """Calibrate each group's scores with a Platt map fitted on every OTHER
    group (leave-one-subject-out calibration: no subject calibrates itself)."""
    labels, probs, groups = map(np.asarray, (labels, probs, groups))
    out = np.empty(len(probs))
    for g in np.unique(groups):
        m = groups == g
        ab = fit_platt(labels[~m], probs[~m])
        out[m] = apply_platt(probs[m], ab)
    return out


# ============================================================
# Uncertainty and paired comparisons
# ============================================================


def bootstrap_ci(
    labels,
    scores,
    groups,
    metric=roc_auc,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
):
    """Subject-level (cluster) bootstrap CI: resample whole subjects, since
    trials of one subject are not independent."""
    labels, scores, groups = map(np.asarray, (labels, scores, groups))
    uniq = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[g] for g in pick])
        if len(np.unique(labels[idx])) < 2:
            continue
        vals.append(metric(labels[idx], scores[idx]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _t_sf(t: float, df: int) -> float:
    try:
        from scipy import stats

        return float(stats.t.sf(t, df))
    except ImportError:  # normal approximation
        return 0.5 * math.erfc(t / math.sqrt(2))


def paired_ttest(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2 or np.std(d, ddof=1) == 0:
        return {"n": n, "mean_diff": float(d.mean()) if n else float("nan"), "p": 1.0}
    t = d.mean() / (np.std(d, ddof=1) / math.sqrt(n))
    return {
        "n": n,
        "mean_diff": float(d.mean()),
        "t": float(t),
        "p": float(2 * _t_sf(abs(t), n - 1)),
    }


def wilcoxon(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Wilcoxon signed-rank test (exact via scipy if available, otherwise the
    normal approximation with tie correction)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d) & (d != 0)]
    n = len(d)
    if n == 0:
        return {"n": 0, "p": 1.0}
    try:
        from scipy import stats

        res = stats.wilcoxon(d)
        return {"n": n, "statistic": float(res.statistic), "p": float(res.pvalue)}
    except ImportError:
        pass
    ranks = np.argsort(np.argsort(np.abs(d))) + 1.0
    w = ranks[d > 0].sum()
    mu = n * (n + 1) / 4
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (w - mu) / sigma
    return {"n": n, "statistic": float(w), "p": float(math.erfc(abs(z) / math.sqrt(2)))}


def per_group_metric(
    labels, scores, groups, metric=roc_auc, names: Optional[Sequence] = None
) -> Dict[str, float]:
    labels, scores, groups = map(np.asarray, (labels, scores, groups))
    out = {}
    for g in names if names is not None else np.unique(groups):
        m = groups == g
        if m.any() and len(np.unique(labels[m])) > 1:
            out[str(g)] = float(metric(labels[m], scores[m]))
    return out
