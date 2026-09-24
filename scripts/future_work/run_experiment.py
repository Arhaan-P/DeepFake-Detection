"""
Run experiments from the master matrix (plan Section 14).
=========================================================
Every experiment is a small JSON config (see configs/future_work/
experiment_matrix.json) evaluated by the SAME controlled protocol in
utils/verification_harness.py. Results land in
outputs/future_work/results/<name>.json.

Usage:
    # everything in the matrix, 4 experiments at a time
    python scripts/future_work/run_experiment.py --all --parallel 4

    # selected experiments (glob patterns allowed)
    python scripts/future_work/run_experiment.py --only "E0*" "E3*"

    # a single ad-hoc experiment
    python scripts/future_work/run_experiment.py --name my_test \\
        --set backend=rtmpose_balanced feature_set=baseline78

    # fast plumbing check (4 subjects, 2 epochs, 1 seed)
    python scripts/future_work/run_experiment.py --only E0_baseline --quick

Author: DeepFake Detection Project
"""

import argparse
import fnmatch
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.verification_harness import (
    ExperimentConfig,
    run_experiment,
    save_result,
)

MATRIX = "configs/future_work/experiment_matrix.json"
OUT_DIR = "outputs/future_work/results"


def load_matrix(path: str):
    with open(path) as fh:
        m = json.load(fh)
    defaults = m.get("defaults", {})
    exps = []
    for e in m["experiments"]:
        cfg = dict(defaults)
        cfg.update(e)
        exps.append(cfg)
    return exps


def _parse_value(v: str):
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def _run(cfg_dict, out_dir, quick):
    cfg = ExperimentConfig.from_dict(cfg_dict)
    if quick:
        cfg.epochs, cfg.seeds, cfg.n_aug = 2, [0], 3
        cfg.subjects = cfg.subjects or ["A2", "Aarav", "Ananya", "Arhaan"]
    lines = []

    def log(msg):
        lines.append(msg)
        print(msg, flush=True)

    res = run_experiment(cfg, log=log)
    path = os.path.join(out_dir, f"{cfg.name}.json")
    save_result(res, path)
    a = res["aggregate"]
    return (
        f"{cfg.name:<32s} AUC {a['pooled_roc_auc_mean'] * 100:6.2f} "
        f"(fold {a['fold_auc_mean'] * 100:5.2f} +/- {a['fold_auc_std'] * 100:4.2f})  "
        f"EER {a['pooled_eer_mean'] * 100:5.2f}  params {res['params']}  "
        f"{res['runtime_s'] / 60:.1f} min"
    )


def main():
    ap = argparse.ArgumentParser(description="Future-work experiment runner")
    ap.add_argument("--matrix", default=MATRIX)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--name", default="")
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides")
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--skip_done", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    overrides = {}
    for kv in args.set:
        k, _, v = kv.partition("=")
        overrides[k] = _parse_value(v)

    if args.name:
        exps = [dict(name=args.name, **overrides)]
    else:
        exps = load_matrix(args.matrix)
        if not args.all:
            exps = [
                e for e in exps if any(fnmatch.fnmatch(e["name"], p) for p in args.only)
            ]
        for e in exps:
            e.update(overrides)
    if args.skip_done:
        exps = [
            e
            for e in exps
            if not os.path.exists(os.path.join(args.out_dir, f"{e['name']}.json"))
        ]
    if not exps:
        print("Nothing to run.")
        return
    out_dir = args.out_dir + ("_quick" if args.quick else "")
    print(f"Running {len(exps)} experiment(s): {[e['name'] for e in exps]}")

    t0 = time.time()
    summary = []
    if args.parallel <= 1:
        for e in exps:
            summary.append(_run(e, out_dir, args.quick))
    else:
        with ProcessPoolExecutor(max_workers=args.parallel) as pool:
            futs = {pool.submit(_run, e, out_dir, args.quick): e for e in exps}
            for f in as_completed(futs):
                try:
                    summary.append(f.result())
                except Exception as exc:  # report and continue
                    summary.append(f"{futs[f]['name']:<32s} FAILED: {exc!r}")
                print(summary[-1], flush=True)
    print("\n" + "=" * 100)
    for s in sorted(summary):
        print(s)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
