"""
Compare experiment results against the frozen baseline (plan Sections 9, 14).
============================================================================
Reads every outputs/future_work/results/*.json and produces:

  report/summary.md     one table per experiment group, each row compared
                        with E0 on IDENTICAL trials:
                          dAUC            mean paired difference of per-fold
                                          AUC (13 folds x seeds)
                          p (Wilcoxon)    paired signed-rank over those folds
                          95% CI          subject-level bootstrap of the
                                          difference in pooled AUC
  report/summary.json   the same numbers, machine-readable
  report/robustness.md  E8: AUC/EER under each perturbation / degraded video
  figures/*.png         forest plot of dAUC, robustness plot, calibration

Experiments whose trial set differs from E0 (cross-view protocols) are
reported without a paired test.

Usage:
    python scripts/future_work/compare_results.py

Author: DeepFake Detection Project
"""

import argparse
import json
import sys
from glob import glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generate_figures"))

import numpy as np

from utils.verification_metrics import (  # noqa: E402
    calibration_summary,
    crossfit_platt,
    paired_ttest,
    roc_auc,
    wilcoxon,
)

GROUPS = [
    ("E0", "P0 - frozen baseline and protocol checks"),
    ("E1", "E1 - pose backend (P1)"),
    ("E2", "E2 - RGB-only 3D pose"),
    ("E3", "E3 - gait cycle and rhythm (P2)"),
    ("E4", "E4 - biomechanical feature families (P3)"),
    ("E7", "E7 - verifier architecture"),
    ("E8", "E8 - cross-view and robustness (P5)"),
]
REFERENCE = "E0_baseline"


def load_results(results_dir):
    out = {}
    for f in sorted(glob(str(Path(results_dir) / "*.json"))):
        with open(f) as fh:
            r = json.load(fh)
        out[r["config"]["name"]] = r
    return out


def fold_auc_map(r):
    return {(row["seed"], row["subject"]): row["roc_auc"] for row in r["fold_table"]}


def calibration_from_trials(r):
    """Recompute calibration from stored trial scores (mean over seeds), so
    the report does not depend on the calibration code version that was
    current when an experiment ran."""
    y = np.array(r["trials"]["label"])
    groups = np.array([q.rsplit("_", 1)[0] for q in r["trials"]["query"]])
    raw, platt, brier = [], [], []
    for v in r["trials"]["scores"].values():
        s = np.array(v["clean"], dtype=float)
        if r["config"]["model"] == "dtw":  # distances: squash to (0, 1) first
            s = 1 / (1 + np.exp(-(s - np.median(s)) / (np.std(s) + 1e-9)))
        raw.append(calibration_summary(y, s)["ece"])
        cal = calibration_summary(y, crossfit_platt(y, s, groups))
        platt.append(cal["ece"])
        brier.append(cal["brier"])
    return float(np.mean(raw)), float(np.mean(platt)), float(np.mean(brier))


def seed_mean_scores(r, cond="clean"):
    s = np.array(
        [np.array(v[cond], dtype=float) for v in r["trials"]["scores"].values()]
    )
    return np.nanmean(s, axis=0)


def same_trials(a, b):
    return (
        a["trials"]["query"] == b["trials"]["query"]
        and a["trials"]["claim"] == b["trials"]["claim"]
    )


def paired_vs_reference(r, ref, n_boot=2000, seed=0):
    fa, fr = fold_auc_map(r), fold_auc_map(ref)
    keys = [k for k in fa if k in fr]
    if len(keys) < 5:  # e.g. single-seed DTW: pair per subject on seed means
        subj = sorted({k[1] for k in fa} & {k[1] for k in fr})
        a = [np.mean([v for (s, u), v in fa.items() if u == x]) for x in subj]
        b = [np.mean([v for (s, u), v in fr.items() if u == x]) for x in subj]
    else:
        a = [fa[k] for k in keys]
        b = [fr[k] for k in keys]
    out = {
        "n_pairs": len(a),
        "d_fold_auc": float(np.mean(np.array(a) - np.array(b))),
        "wilcoxon_p": wilcoxon(a, b)["p"],
        "ttest_p": paired_ttest(a, b)["p"],
    }
    # subject-level bootstrap of the pooled-AUC difference on identical trials
    y = np.array(ref["trials"]["label"])
    sa, sr = seed_mean_scores(r), seed_mean_scores(ref)
    groups = np.array([q.rsplit("_", 1)[0] for q in ref["trials"]["query"]])
    uniq = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[g] for g in pick])
        diffs.append(roc_auc(y[idx], sa[idx]) - roc_auc(y[idx], sr[idx]))
    out["d_pooled_auc"] = float(roc_auc(y, sa) - roc_auc(y, sr))
    out["d_pooled_auc_ci"] = [
        float(np.percentile(diffs, 2.5)),
        float(np.percentile(diffs, 97.5)),
    ]
    return out


def pct(x, nd=2):
    return "–" if x is None or not np.isfinite(x) else f"{100 * x:.{nd}f}"


def main():
    # Markdown output contains non-ASCII; Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Compare future-work results")
    ap.add_argument("--results", default="outputs/future_work/results")
    ap.add_argument("--out", default="outputs/future_work/report")
    ap.add_argument("--figures", default="outputs/future_work/figures")
    args = ap.parse_args()

    res = load_results(args.results)
    if REFERENCE not in res:
        raise SystemExit(f"{REFERENCE} result missing in {args.results}")
    ref = res[REFERENCE]
    rows = {}
    for name, r in res.items():
        a = r["aggregate"]
        row = {
            "name": name,
            "description": r["config"].get("description", ""),
            "params": r["params"],
            "seeds": len(r["config"]["seeds"]),
            "pooled_auc": a["pooled_roc_auc_mean"],
            "pooled_auc_sd": a["pooled_roc_auc_std"],
            "fold_auc": a["fold_auc_mean"],
            "fold_auc_sd": a["fold_auc_std"],
            "pooled_eer": a["pooled_eer_mean"],
            "tpr_at_fpr_5": a["pooled_tpr_at_fpr_5_mean"],
            **dict(
                zip(("ece_raw", "ece_platt", "brier_platt"), calibration_from_trials(r))
            ),
            "runtime_min": r["runtime_s"] / 60,
            "by_view": {
                v: np.mean(
                    [r["per_seed"][s]["by_view"][v]["roc_auc"] for s in r["per_seed"]]
                )
                for v in r["per_seed"][next(iter(r["per_seed"]))]["by_view"]
            },
        }
        if name != REFERENCE and same_trials(r, ref):
            row.update(paired_vs_reference(r, ref))
        rows[name] = row

    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / "summary.json", "w") as fh:
        json.dump(rows, fh, indent=1)

    lines = [
        "# Future-work experiment results\n",
        "All experiments share the protocol in `utils/verification_harness.py` "
        "(13-fold leave-one-subject-out, training-subject normalisation, strict "
        "enrolment, exhaustive claims). AUC/EER in %. dAUC = paired mean "
        "difference of per-fold AUC vs `E0_baseline` (positive = better); "
        "CI = subject-bootstrap 95% interval of the pooled-AUC difference; "
        "p = Wilcoxon signed-rank over folds x seeds.\n",
    ]
    for prefix, title in GROUPS:
        names = [n for n in rows if n.startswith(prefix)]
        if not names:
            continue
        lines += [
            f"\n## {title}\n",
            "| Experiment | Params | Pooled AUC | Fold AUC | EER | TPR@5%FPR "
            "| AUC side | AUC frontal | ECE→Platt | dAUC | 95% CI | p |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for n in sorted(names, key=lambda x: (x != REFERENCE, x)):
            r = rows[n]
            ci = r.get("d_pooled_auc_ci")
            d_auc, ci_txt, p_txt = "", "", ""
            if "d_fold_auc" in r:
                d_auc = f"{100 * r['d_fold_auc']:+.2f}"
                ci_txt = f"[{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}]"
                p_txt = f"{r['wilcoxon_p']:.3g}" + (
                    " *" if r["wilcoxon_p"] < 0.05 else ""
                )
            lines.append(
                f"| {n} | {r['params']:,} | {pct(r['pooled_auc'])} ± {pct(r['pooled_auc_sd'])} "
                f"| {pct(r['fold_auc'])} ± {pct(r['fold_auc_sd'])} | {pct(r['pooled_eer'])} "
                f"| {pct(r['tpr_at_fpr_5'])} | {pct(r['by_view'].get('S'))} "
                f"| {pct(r['by_view'].get('F'))} "
                f"| {pct(r['ece_raw'], 1)}→{pct(r['ece_platt'], 1)} "
                f"| {d_auc} | {ci_txt} | {p_txt} |"
            )
        lines.append("")
        for n in sorted(names):
            if rows[n]["description"]:
                lines.append(f"- `{n}`: {rows[n]['description']}")
    (Path(args.out) / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf8")

    robustness_report(res, args.out)
    make_figures(res, rows, args.figures)
    print((Path(args.out) / "summary.md").read_text(encoding="utf8"))


def robustness_report(res, out_dir):
    lines = [
        "# Robustness (E8): clean enrolment, degraded query\n",
        "AUC / EER in %, mean over seeds. Keypoint perturbations are applied to "
        "the extracted pose stream; `video:` rows re-ran pose estimation on "
        "degraded copies of the videos.\n",
    ]
    for name in sorted(n for n, r in res.items() if "robustness_auc" in r["aggregate"]):
        a = res[name]["aggregate"]
        lines += [
            f"\n## {name}\n",
            "| Condition | AUC | ΔAUC vs clean | EER |",
            "|---|---|---|---|",
            f"| clean | {pct(a['pooled_roc_auc_mean'])} | – | {pct(a['pooled_eer_mean'])} |",
        ]
        for cond, auc in a["robustness_auc"].items():
            lines.append(
                f"| {cond} | {pct(auc)} | {100 * (auc - a['pooled_roc_auc_mean']):+.2f} "
                f"| {pct(a['robustness_eer'][cond])} |"
            )
    (Path(out_dir) / "robustness.md").write_text(
        "\n".join(lines) + "\n", encoding="utf8"
    )


def make_figures(res, rows, fig_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from figstyle import C_AUTH, C_FAKE, C_NEUTRAL, apply_style, despine

    apply_style()
    Path(fig_dir).mkdir(parents=True, exist_ok=True)

    # 1. forest plot of paired pooled-AUC differences vs E0
    paired = [r for r in rows.values() if "d_pooled_auc" in r]
    paired.sort(key=lambda r: (r["name"][:2], r["d_pooled_auc"]))
    if paired:
        fig, ax = plt.subplots(figsize=(6.2, 0.22 * len(paired) + 0.9))
        for i, r in enumerate(paired):
            lo, hi = r["d_pooled_auc_ci"]
            c = C_AUTH if lo > 0 else (C_FAKE if hi < 0 else C_NEUTRAL)
            ax.plot(
                [100 * lo, 100 * hi], [i, i], color=c, lw=1.6, solid_capstyle="round"
            )
            ax.plot(
                100 * r["d_pooled_auc"], i, "o", ms=4.5, color=c, mec="white", mew=0.8
            )
        ax.axvline(0, color=C_NEUTRAL, lw=0.8)
        ax.set_yticks(range(len(paired)))
        ax.set_yticklabels([r["name"] for r in paired])
        ax.set_xlabel(
            "Δ pooled ROC-AUC vs E0_baseline (percentage points, 95% subject bootstrap)"
        )
        ax.set_title(
            "Each experiment changes one variable; blue = CI above 0, red = below 0"
        )
        ax.grid(axis="y", visible=False)
        despine(ax)
        fig.savefig(Path(fig_dir) / "fw_delta_auc_forest.png")
        plt.close(fig)

    # 2. robustness of E0 under each condition
    r0 = res.get(REFERENCE)
    if r0 and "robustness_auc" in r0["aggregate"]:
        a = r0["aggregate"]
        conds = list(a["robustness_auc"])
        vals = [
            100 * (a["robustness_auc"][c] - a["pooled_roc_auc_mean"]) for c in conds
        ]
        order = np.argsort(vals)
        fig, ax = plt.subplots(figsize=(6.2, 0.22 * len(conds) + 0.9))
        for i, k in enumerate(order):
            c = C_FAKE if vals[k] < -1 else C_NEUTRAL
            ax.barh(i, vals[k], color=c, height=0.62)
        ax.set_yticks(range(len(conds)))
        ax.set_yticklabels([conds[k] for k in order])
        ax.axvline(0, color=C_NEUTRAL, lw=0.8)
        ax.set_xlabel(
            f"Δ ROC-AUC vs clean query (clean = {100 * a['pooled_roc_auc_mean']:.1f}%), percentage points"
        )
        ax.set_title("E0 robustness: clean enrolment, degraded query")
        ax.grid(axis="y", visible=False)
        despine(ax)
        fig.savefig(Path(fig_dir) / "fw_robustness_e0.png")
        plt.close(fig)

        # 3. reliability diagram, raw vs cross-fitted Platt (seed 0)
        ps = r0["per_seed"][next(iter(r0["per_seed"]))]
        bins = ps["reliability_raw"]
        fig, ax = plt.subplots(figsize=(3.3, 3.1))
        ax.plot(
            [0, 1],
            [0, 1],
            color=C_NEUTRAL,
            lw=0.8,
            ls="--",
            label="perfect calibration",
        )
        ax.plot(
            [b["mean_prob"] for b in bins],
            [b["frac_positive"] for b in bins],
            "o-",
            color=C_AUTH,
            ms=4,
            label=f"raw P(authentic), ECE {100 * ps['calibration_raw']['ece']:.1f}%",
        )
        ax.set_xlabel("predicted P(authentic)")
        ax.set_ylabel("observed fraction genuine")
        ax.set_title("E0 calibration (seed 0)")
        ax.legend(loc="upper left")
        despine(ax)
        fig.savefig(Path(fig_dir) / "fw_calibration_e0.png")
        plt.close(fig)

    # 4. per-view AUC for the main groups
    names = [n for n in rows if set(rows[n]["by_view"]) >= {"F", "S"}]
    names.sort()
    if names:
        fig, ax = plt.subplots(figsize=(6.2, 0.22 * len(names) + 0.9))
        for i, n in enumerate(names):
            f, s = rows[n]["by_view"]["F"], rows[n]["by_view"]["S"]
            ax.plot([100 * f, 100 * s], [i, i], color="#bdbdbd", lw=1.2, zorder=1)
            ax.plot(
                100 * s,
                i,
                "o",
                color=C_AUTH,
                ms=4.5,
                zorder=2,
                label="side walks" if i == 0 else None,
            )
            ax.plot(
                100 * f,
                i,
                "s",
                color=C_FAKE,
                ms=4.2,
                zorder=2,
                label="frontal walks" if i == 0 else None,
            )
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel("pooled ROC-AUC by query view (%), mean over seeds")
        ax.legend(loc="lower left")
        ax.grid(axis="y", visible=False)
        despine(ax)
        fig.savefig(Path(fig_dir) / "fw_auc_by_view.png")
        plt.close(fig)
    print(f"figures -> {fig_dir}")


if __name__ == "__main__":
    main()
