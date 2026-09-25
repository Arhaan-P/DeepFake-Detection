"""
Train a deployable verifier bundle (plan P8) from an evaluated configuration.
============================================================================
Trains the verifier of one experiment configuration on ALL enrolled subjects
and packages it with its normalisation, enrolment signatures and gait-
parameter statistics (utils/gait_verifier.py). The decision threshold and the
Platt calibration are taken from that configuration's LOSO result file, i.e.
from held-out-subject scores -- the deployed model is never tuned on its own
training subjects.

Usage:
    python scripts/future_work/train_final.py --experiment E0_baseline
    python scripts/future_work/train_final.py --experiment E0_baseline --operating_point youden

    # keep some clips out of training AND enrolment, for honest demos
    python scripts/future_work/train_final.py --experiment E0_baseline --holdout_clips Som_S1 Teja_S1 --out outputs/future_work/checkpoints/E0_demo.pt

Author: DeepFake Detection Project
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from utils.gait_verifier import GaitVerifier, build_param_stats
from utils.verification_harness import (
    DescriptorBank,
    ExperimentConfig,
    Signatures,
    load_clips,
    train_fold,
)
from utils.verification_metrics import eer, fit_platt, youden_threshold


def main():
    ap = argparse.ArgumentParser(description="Train a deployable gait verifier")
    ap.add_argument("--experiment", default="E0_baseline")
    ap.add_argument("--matrix", default="configs/future_work/experiment_matrix.json")
    ap.add_argument("--results", default="outputs/future_work/results")
    ap.add_argument("--operating_point", choices=["eer", "youden"], default="eer")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument(
        "--holdout_clips",
        nargs="*",
        default=[],
        help="clip names excluded from training and enrolment (demo queries)",
    )
    args = ap.parse_args()

    with open(args.matrix) as fh:
        m = json.load(fh)
    cfg_d = dict(m["defaults"])
    cfg_d.update(next(e for e in m["experiments"] if e["name"] == args.experiment))
    cfg = ExperimentConfig.from_dict(cfg_d)
    if cfg.model == "dtw":
        raise SystemExit("DTW has no trainable model to deploy.")
    torch.set_num_threads(max(cfg.threads, 4))

    res_path = Path(args.results) / f"{args.experiment}.json"
    if not res_path.exists():
        raise SystemExit(
            f"{res_path} missing: evaluate the configuration first "
            f"(run_experiment.py --only {args.experiment}); its held-out scores "
            f"set the threshold and calibration."
        )
    with open(res_path) as fh:
        res = json.load(fh)
    y = np.array(res["trials"]["label"])
    s = np.nanmean(
        [np.array(v["clean"], dtype=float) for v in res["trials"]["scores"].values()],
        axis=0,
    )
    thr = eer(y, s)[1] if args.operating_point == "eer" else youden_threshold(y, s)[0]
    platt = fit_platt(y, s)

    clips = load_clips(cfg.backend, cfg.source, cfg.cache_root)
    clips = [c for c in clips if c.name not in set(args.holdout_clips)]
    bank = DescriptorBank(clips, cfg, "data/descriptor_cache")
    flat = bank.data.reshape(-1, bank.data.shape[-1])
    mu, sd = flat.mean(0), flat.std(0)
    sd[sd < 1e-6] = 1.0

    def norm(a):
        return ((a - mu) / sd).astype(np.float32)

    originals = norm(bank.data[:, 0])
    sigs = Signatures(clips, originals, cfg.protocol)
    ident = [c.identity for c in clips]
    n_copies = bank.data.shape[1]
    train_x = norm(bank.data).reshape(-1, *bank.data.shape[2:])
    train_meta = [(ident[i], i) for i in range(len(clips)) for _ in range(n_copies)]
    print(
        f"Training {cfg.model} on {len(set(ident))} identities, {len(train_x)} samples"
    )
    model, loss = train_fold(cfg, train_x, train_meta, sigs, ident, args.seed)

    signatures = {g: sigs.get(g) for g in sorted(set(ident))}
    bundle = {
        "config": cfg.__dict__,
        "input_dim": int(train_x.shape[-1]),
        "model_state": model.state_dict(),
        "mu": mu,
        "sd": sd,
        "signatures": signatures,
        "threshold": float(thr),
        "platt": platt,
        "param_stats": build_param_stats(clips),
        "meta": {
            "experiment": args.experiment,
            "operating_point": args.operating_point,
            "train_loss": loss,
            "loso_pooled_auc": res["aggregate"]["pooled_roc_auc_mean"],
            "loso_pooled_eer": res["aggregate"]["pooled_eer_mean"],
            "clips": [c.name for c in clips],
            "holdout_clips": args.holdout_clips,
        },
    }
    out = args.out or f"outputs/future_work/checkpoints/{args.experiment}.pt"
    GaitVerifier.save_bundle(bundle, out)
    print(
        f"saved {out}\n  threshold ({args.operating_point}) = {thr:.4f}, "
        f"Platt a,b = {platt[0]:.3f},{platt[1]:.3f}, "
        f"LOSO AUC {100 * bundle['meta']['loso_pooled_auc']:.2f}%"
    )


if __name__ == "__main__":
    main()
