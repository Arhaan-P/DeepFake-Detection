"""
Definitive LOOCV ablation for gait-based deepfake detection.
============================================================================

WHY THIS SCRIPT EXISTS
----------------------
Two earlier attempts at this ablation were both unsound:

  v1 (original ablation_study.py)
      All four variants computed logits from an identical diff_conv head on
      the RAW 78-dim feature difference. The named CNN/LSTM/Transformer branch
      produced an embedding that never reached the logits and got no gradient.
      The four "variants" were one classifier trained four times; the 1.58-point
      spread was training noise. See NOTES.md Section 2.2.

  v2 (corrected ablation_study.py, embedding-comparison)
      Put each branch genuinely on the decision path, but by mean-pooling each
      sequence to a single embedding and comparing embeddings. That works, but
      it discards the per-timestep alignment between the observed sequence and
      the enrolled signature -- which is the strongest signal available, and
      the one the deployed model actually uses. It also asks a Siamese encoder
      to metric-learn identity from 10 training subjects. Result: every variant
      landed at 40-79% AUC, i.e. the formulation was measuring how hard
      metric learning is on a tiny cohort, not which temporal encoder helps.

v3 (this script) asks the question the paper actually needs answered:

    Does encoding the sequences BEFORE differencing them beat differencing
    the raw features, which is what the deployed model does?

Architecture for every variant:

    h_v = Encoder(V)                      # (B, T, d), time preserved
    h_c = Encoder(C)                      # shared weights (Siamese)
    F   = [h_v - h_c || |h_v - h_c| || h_v * h_c]      # (B, T, 3d)
    logits = DiffClassifier(TemporalCNN(F))

The "Raw" variant uses Encoder = identity, d = 78, which reproduces the
deployed model in models/full_pipeline.py exactly (133,058 params). It is the
control: any other variant must beat it to justify its parameters.

PROTOCOL
--------
13-fold leave-one-subject-out, matching the headline evaluation. Every video
of the held-out subject -- original and all 15 augmentations -- is withheld.
Model state is selected by lowest TRAINING loss (never validation/test), the
same rule scripts/evaluation/evaluate.py uses, so nothing is selected on test.

NORMALIZATION -- read this before comparing to the headline number
------------------------------------------------------------------
scripts/evaluation/evaluate.py builds its LOOCV test dataset WITHOUT passing
feature_stats, so GaitDataset falls back to computing z-score statistics from
the held-out subject's own data. The paper (Section IV) states that mu and
sigma are estimated on the training split alone. Those two statements are not
compatible, and the committed loocv_results.json was produced the first way.

This script runs BOTH so the difference can be measured rather than argued
about:
    norm=train   test set normalized with TRAINING statistics  (correct)
    norm=legacy  test set normalized with its own statistics   (as shipped)

All variant-vs-variant comparisons use norm=train. The legacy arm is run for
the Raw baseline only, purely to quantify what that protocol choice was worth.

OUTPUT
------
Writes outputs/ablation/ablation_loocv_results.json incrementally after every
fold, so an interrupted run still yields usable partial results.

Usage:
    python scripts/evaluation/ablation_loocv.py                 # full run
    python scripts/evaluation/ablation_loocv.py --quick         # smoke test
"""

import argparse
import json
import time
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from utils.data_loader import GaitDataset


# ============================================================
# Encoders -- every one maps (B, T, 78) -> (B, T, d), time preserved
# ============================================================

class RawEncoder(nn.Module):
    """Identity. Reproduces the deployed model's raw-difference behaviour."""
    out_dim = 78

    def forward(self, x):
        return x


class CNNEncoder(nn.Module):
    """GaitEncoder: 1D residual CNN, 78 -> 64 -> 128."""

    def __init__(self, input_dim=78, hidden_dims=(64, 128), out_dim=128, dropout=0.1):
        super().__init__()
        from models.gait_encoder import GaitEncoder
        self.enc = GaitEncoder(input_dim=input_dim, hidden_dims=hidden_dims,
                               output_dim=out_dim, dropout=dropout)
        self.out_dim = out_dim

    def forward(self, x):
        return self.enc(x)


class LSTMEncoder(nn.Module):
    """BiLSTM, per-timestep outputs (not the pooled final hidden state)."""

    def __init__(self, input_dim=78, hidden=64, layers=1, dropout=0.1):
        super().__init__()
        from models.temporal_model import BiLSTMEncoder
        self.enc = BiLSTMEncoder(input_dim=input_dim, hidden_dim=hidden,
                                 num_layers=layers, dropout=dropout)
        self.out_dim = self.enc.output_dim

    def forward(self, x):
        outputs, _ = self.enc(x)
        return outputs


class TransformerEnc(nn.Module):
    """Transformer encoder, per-timestep outputs."""

    def __init__(self, input_dim=78, d_model=128, nhead=4, layers=2, dropout=0.1):
        super().__init__()
        from models.temporal_model import TransformerEncoder
        self.enc = TransformerEncoder(input_dim=input_dim, d_model=d_model,
                                      nhead=nhead, num_layers=layers,
                                      dropout=dropout)
        self.out_dim = d_model

    def forward(self, x):
        outputs, _ = self.enc(x)
        return outputs


class HybridEncoder(nn.Module):
    """GaitEncoder -> DualPathTemporalModel, per-timestep fused sequence."""

    def __init__(self, input_dim=78, hidden_dims=(64, 128), enc_dim=128,
                 lstm_hidden=64, lstm_layers=1, d_model=128, nhead=4,
                 t_layers=2, out_dim=128, dropout=0.1):
        super().__init__()
        from models.gait_encoder import GaitEncoder
        from models.temporal_model import DualPathTemporalModel
        self.cnn = GaitEncoder(input_dim=input_dim, hidden_dims=hidden_dims,
                               output_dim=enc_dim, dropout=dropout)
        self.temporal = DualPathTemporalModel(
            input_dim=enc_dim, lstm_hidden=lstm_hidden, lstm_layers=lstm_layers,
            transformer_d_model=d_model, transformer_heads=nhead,
            transformer_layers=t_layers, output_dim=out_dim, dropout=dropout)
        self.out_dim = out_dim

    def forward(self, x):
        seq, _ = self.temporal(self.cnn(x))
        return seq


# variant name -> (encoder factory, fuse_raw)
#
# "Raw" is the control: identity encoder, 133,058 params, reproduces the
# deployed model in models/full_pipeline.py exactly.
#
# "Raw + X" arms ask the question that matters for the paper: does wiring
# encoder X in ADD anything on top of the deployed model? Each of these
# contains the raw comparison channels, so a well-optimized run should be
# able to match Raw and can only justify itself by beating it.
#
# "X only" arms ask whether an encoder can REPLACE raw differencing. The
# hybrid is carried through as the representative case, since the earlier
# embedding-comparison study already indicated the answer is no.
VARIANTS = {
    'Raw (deployed)':      (lambda dr: RawEncoder(),          False),
    'Raw + CNN':           (lambda dr: CNNEncoder(dropout=dr), True),
    'Raw + BiLSTM':        (lambda dr: LSTMEncoder(dropout=dr), True),
    'Raw + Transformer':   (lambda dr: TransformerEnc(dropout=dr), True),
    'Raw + Hybrid':        (lambda dr: HybridEncoder(dropout=dr), True),
    'Hybrid only (no raw)': (lambda dr: HybridEncoder(dropout=dr), False),
}


# ============================================================
# Verifier: encode -> per-timestep difference -> temporal CNN -> 2 logits
# ============================================================

class EncodedDiffVerifier(nn.Module):
    """
    Comparison features are formed from the raw pair, the encoded pair, or both.

    fuse_raw=True keeps the raw per-timestep comparison channels alongside the
    encoded ones. This matters: raw dimension j of V and of C are the same
    physical quantity (e.g. left-knee-x), so their difference is meaningful at
    initialization, whereas a freshly initialized encoder makes h_v - h_c a
    random projection that has to relearn that structure from scratch. Fusing
    means each variant is asking the question the paper actually needs -- does
    wiring the encoder in ADD anything on top of what the deployed model
    already does -- rather than whether an encoder can replace raw differencing
    outright (which the encoded-only arm answers separately).
    """

    def __init__(self, encoder, verification_hidden=64, dropout=0.1,
                 fuse_raw=False, raw_dim=78):
        super().__init__()
        self.encoder = encoder
        self.fuse_raw = fuse_raw
        self.is_raw_only = isinstance(encoder, RawEncoder)

        comparison_dim = 0
        if self.is_raw_only or fuse_raw:
            comparison_dim += raw_dim * 3
        if not self.is_raw_only:
            comparison_dim += encoder.out_dim * 3
        vh = verification_hidden

        self.diff_conv = nn.Sequential(
            nn.Conv1d(comparison_dim, vh, kernel_size=7, padding=3),
            nn.BatchNorm1d(vh), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(vh, vh, kernel_size=5, padding=2),
            nn.BatchNorm1d(vh), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(vh, vh // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(vh // 2), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.diff_classifier = nn.Sequential(
            nn.Linear(vh // 2, vh // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(vh // 2, 2),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.LSTM):
                continue
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _compare(a, b):
        d = a - b
        return torch.cat([d, torch.abs(d), a * b], dim=2)

    def forward(self, video, claimed):
        parts = []

        if self.is_raw_only or self.fuse_raw:
            parts.append(self._compare(video, claimed))

        if not self.is_raw_only:
            # Encode the pair in ONE batched pass, then split. The encoders use
            # BatchNorm; running video and claimed through separately would let
            # BN compute different statistics for each, normalizing away the
            # scale difference between a single noisy video and an averaged
            # enrolment signature -- which is part of the identity signal.
            n = video.size(0)
            h = self.encoder(torch.cat([video, claimed], dim=0))
            parts.append(self._compare(h[:n], h[n:]))

        f = torch.cat(parts, dim=2)
        x = self.diff_conv(f.permute(0, 2, 1)).squeeze(-1)
        return self.diff_classifier(x)


# ============================================================
# Metrics
# ============================================================

def compute_eer(labels, scores):
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    k = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[k] + fnr[k]) / 2.0), float(thr[k])


def fold_metrics(labels, scores, preds):
    m = {
        'accuracy': float(accuracy_score(labels, preds)),
        'f1': float(f1_score(labels, preds, zero_division=0)),
        'precision': float(precision_score(labels, preds, zero_division=0)),
        'recall': float(recall_score(labels, preds, zero_division=0)),
    }
    if len(np.unique(labels)) > 1:
        m['roc_auc'] = float(roc_auc_score(labels, scores))
        m['eer'], _ = compute_eer(labels, scores)
    else:
        m['roc_auc'] = float('nan')
        m['eer'] = float('nan')
    return m


# ============================================================
# Train / evaluate one (variant, seed, fold)
# ============================================================

def run_one(spec, train_loader, test_loader, device, epochs, lr,
            dropout, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    encoder_fn, fuse_raw = spec
    model = EncodedDiffVerifier(encoder_fn(dropout), dropout=dropout,
                                fuse_raw=fuse_raw).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_loss, best_state = float('inf'), None
    model.train()
    for _ in range(epochs):
        running = 0.0
        for batch in train_loader:
            v = batch['video_features'].to(device, non_blocking=True)
            c = batch['claimed_features'].to(device, non_blocking=True)
            y = batch['label'].to(device).squeeze(-1).long()

            optimizer.zero_grad()
            logits = model(v, c)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running += loss.item()

        avg = running / max(len(train_loader), 1)
        # Selection on TRAINING loss only -- never on the held-out subject.
        if avg < best_loss:
            best_loss = avg
            best_state = deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    labels, scores = [], []
    with torch.no_grad():
        for batch in test_loader:
            v = batch['video_features'].to(device, non_blocking=True)
            c = batch['claimed_features'].to(device, non_blocking=True)
            y = batch['label'].squeeze(-1).long()
            probs = F.softmax(model(v, c), dim=1)
            labels.extend(y.numpy().tolist())
            scores.extend(probs[:, 1].cpu().numpy().tolist())

    labels = np.array(labels)
    scores = np.array(scores)
    preds = (scores >= 0.5).astype(int)
    m = fold_metrics(labels, scores, preds)
    m['params'] = int(n_params)
    m['train_loss'] = float(best_loss)
    return m, labels, scores


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description='Definitive LOOCV ablation')
    ap.add_argument('--features_file', default='data/gait_features/gait_features.pkl')
    ap.add_argument('--enrolled_file', default='data/gait_features/enrolled_identities.pkl')
    ap.add_argument('--output', default='outputs/ablation/ablation_loocv_results.json')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--quick', action='store_true',
                    help='2 folds, 2 epochs, 1 seed -- smoke test only')
    args = ap.parse_args()

    if args.quick:
        args.epochs, args.seeds = 2, [0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('=' * 78, flush=True)
    print('  DEFINITIVE LOOCV ABLATION  (encoder-then-difference)', flush=True)
    print('=' * 78, flush=True)
    print(f'  Started      : {datetime.now():%Y-%m-%d %H:%M:%S}', flush=True)
    print(f'  Torch        : {torch.__version__}', flush=True)
    print(f'  Device       : {device}', flush=True)
    if device.type == 'cuda':
        print(f'  GPU          : {torch.cuda.get_device_name(0)}', flush=True)
        print(f'  VRAM         : '
              f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB',
              flush=True)
        torch.backends.cudnn.benchmark = True
    else:
        print('  WARNING: CUDA not available -- this will be slow.', flush=True)
    print(f'  Epochs/run   : {args.epochs}', flush=True)
    print(f'  Seeds        : {args.seeds}', flush=True)
    print('=' * 78, flush=True)

    # Discover subjects from the enrolment file.
    import pickle
    with open(args.enrolled_file, 'rb') as fh:
        persons = sorted(pickle.load(fh).keys())
    if args.quick:
        persons = persons[:2]
    print(f'\n  Subjects ({len(persons)}): {persons}\n', flush=True)

    variant_names = list(VARIANTS.keys())
    # Work items per fold: (variant, seed, norm_mode)
    work = [(v, s, 'train') for s in args.seeds for v in variant_names]
    work += [('Raw (deployed)', s, 'legacy') for s in args.seeds[:1]]

    total_runs = len(persons) * len(work)
    print(f'  Variants     : {len(variant_names)}', flush=True)
    print(f'  Runs/fold    : {len(work)}   (incl. 1 legacy-normalization arm)',
          flush=True)
    print(f'  TOTAL RUNS   : {total_runs}\n', flush=True)

    results = {}   # key -> {'per_fold': {...}, 'pooled': {...}}
    def key_of(variant, seed, norm):
        return f'{variant} | seed={seed} | norm={norm}'

    for k in [key_of(*w) for w in work]:
        results[k] = {'per_fold': {}, '_pooled_labels': [], '_pooled_scores': []}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Archive any existing results before the first fold overwrites them.
    # Without this, starting a second run silently destroys a completed one:
    # the per-fold checkpoint rewrites this path from fold 1 onward, so a
    # finished 13-fold result becomes a 1-fold result the moment a new run
    # checkpoints. Learned the hard way on 2026-08-28.
    if out_path.exists():
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup = out_path.with_suffix(f'.{stamp}.bak.json')
        out_path.replace(backup)
        print(f'  NOTE: archived previous results -> {backup.name}', flush=True)

    done = 0
    t_start = time.time()

    for fold_idx, test_person in enumerate(persons, 1):
        train_persons = [p for p in persons if p != test_person]
        fold_t0 = time.time()

        print('-' * 78, flush=True)
        print(f'  FOLD {fold_idx}/{len(persons)}  --  held out: {test_person}',
              flush=True)
        print('-' * 78, flush=True)

        # Build datasets ONCE per fold and reuse for every variant/seed.
        # (Each construction reads ~234 MB of pickles; doing it per-run would
        # dominate total runtime.)
        t0 = time.time()
        train_ds = GaitDataset(features_file=args.features_file,
                               enrolled_identities_file=args.enrolled_file,
                               person_list=train_persons, mode='verification')
        stats = train_ds.get_feature_stats()
        test_ds_train_norm = GaitDataset(
            features_file=args.features_file,
            enrolled_identities_file=args.enrolled_file,
            person_list=[test_person], mode='verification',
            feature_stats=stats)
        test_ds_legacy = GaitDataset(
            features_file=args.features_file,
            enrolled_identities_file=args.enrolled_file,
            person_list=[test_person], mode='verification')
        print(f'    data ready in {time.time() - t0:.1f}s '
              f'(train {len(train_ds)}, test {len(test_ds_train_norm)})',
              flush=True)

        if len(train_ds) == 0 or len(test_ds_train_norm) == 0:
            print('    skipping fold -- insufficient data', flush=True)
            continue

        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0, pin_memory=True)
        loaders = {
            'train': DataLoader(test_ds_train_norm, batch_size=args.batch_size,
                                shuffle=False, num_workers=0, pin_memory=True),
            'legacy': DataLoader(test_ds_legacy, batch_size=args.batch_size,
                                 shuffle=False, num_workers=0, pin_memory=True),
        }

        for variant, seed, norm in work:
            k = key_of(variant, seed, norm)
            t0 = time.time()
            m, labels, scores = run_one(
                VARIANTS[variant], train_loader, loaders[norm], device,
                args.epochs, args.lr, args.dropout, seed)
            dt = time.time() - t0
            done += 1

            results[k]['per_fold'][test_person] = m
            results[k]['_pooled_labels'].extend(labels.tolist())
            results[k]['_pooled_scores'].extend(scores.tolist())

            rate = (time.time() - t_start) / done
            eta = timedelta(seconds=int(rate * (total_runs - done)))
            print(f'    [{done:3d}/{total_runs}] {variant:<24s} '
                  f'seed={seed} norm={norm:<6s} | '
                  f'AUC={m["roc_auc"]:.4f} acc={m["accuracy"]:.4f} '
                  f'F1={m["f1"]:.4f} EER={m["eer"]:.4f} | '
                  f'{dt:5.1f}s | ETA {eta}', flush=True)

        # ---- incremental save after every fold ----
        snapshot = _summarize(results, args, persons, fold_idx)
        with open(out_path, 'w') as fh:
            json.dump(snapshot, fh, indent=2)
        print(f'    fold done in {time.time() - fold_t0:.1f}s '
              f'-> checkpointed to {out_path}', flush=True)

    # ---- final summary ----
    snapshot = _summarize(results, args, persons, len(persons))
    with open(out_path, 'w') as fh:
        json.dump(snapshot, fh, indent=2)

    print('\n' + '=' * 78, flush=True)
    print('  FINAL RESULTS  (mean +/- std across folds; pooled over all scores)',
          flush=True)
    print('=' * 78, flush=True)
    hdr = (f'  {"Variant":<24s} {"seed":>4s} {"norm":>6s} {"params":>9s} '
           f'{"AUC":>16s} {"Acc":>16s} {"EER":>16s} {"pooled AUC":>11s}')
    print(hdr, flush=True)
    print('  ' + '-' * (len(hdr) - 2), flush=True)
    for k, agg in snapshot['results'].items():
        a = agg['aggregate']
        print(f'  {agg["variant"]:<24s} {agg["seed"]:>4d} {agg["norm"]:>6s} '
              f'{a["params"]:>9,d} '
              f'{a["roc_auc"]*100:>7.2f} +/-{a["roc_auc_std"]*100:5.2f} '
              f'{a["accuracy"]*100:>7.2f} +/-{a["accuracy_std"]*100:5.2f} '
              f'{a["eer"]*100:>7.2f} +/-{a["eer_std"]*100:5.2f} '
              f'{agg["pooled"]["roc_auc"]*100:>10.2f}', flush=True)

    print('\n' + '=' * 78, flush=True)
    print(f'  Completed    : {datetime.now():%Y-%m-%d %H:%M:%S}', flush=True)
    print(f'  Total time   : {timedelta(seconds=int(time.time() - t_start))}',
          flush=True)
    print(f'  Results      : {out_path}', flush=True)
    print('=' * 78, flush=True)


def _summarize(results, args, persons, folds_done):
    """Build the JSON snapshot, including aggregates over completed folds."""
    out = {
        '_meta': {
            'schema': 'loocv-ablation-v3-encoded-difference',
            'description': (
                'Encoder-then-difference LOOCV ablation. Each variant encodes '
                'the observed and claimed sequences with shared weights, keeps '
                'the time axis, then classifies the per-timestep difference. '
                'The "Raw (deployed model)" variant uses an identity encoder '
                'and reproduces models/full_pipeline.py exactly; it is the '
                'control every other variant must beat.'),
            'normalization_arms': {
                'train': 'test set z-scored with TRAINING statistics (correct)',
                'legacy': ('test set z-scored with its own statistics -- the '
                           'behaviour of scripts/evaluation/evaluate.py, which '
                           'produced the committed loocv_results.json'),
            },
            'model_selection': 'lowest training loss; never selected on test',
            'epochs': args.epochs,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'dropout': args.dropout,
            'seeds': args.seeds,
            'subjects': persons,
            'folds_completed': folds_done,
            'folds_total': len(persons),
            'generated': datetime.now().isoformat(),
        },
        'results': {},
    }

    for k, r in results.items():
        pf = r['per_fold']
        if not pf:
            continue
        variant, seed, norm = [p.split('=')[-1].strip() for p in k.split('|')]
        variant = k.split('|')[0].strip()

        agg = {}
        for metric in ('accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'eer'):
            vals = np.array([v[metric] for v in pf.values()], dtype=float)
            vals = vals[~np.isnan(vals)]
            agg[metric] = float(vals.mean()) if vals.size else float('nan')
            agg[metric + '_std'] = float(vals.std()) if vals.size else float('nan')
        agg['params'] = int(next(iter(pf.values()))['params'])

        y = np.array(r['_pooled_labels'])
        s = np.array(r['_pooled_scores'])
        pooled = {}
        if y.size and len(np.unique(y)) > 1:
            pooled['roc_auc'] = float(roc_auc_score(y, s))
            pooled['eer'], _ = compute_eer(y, s)
            pooled['accuracy'] = float(accuracy_score(y, (s >= 0.5).astype(int)))
            pooled['n'] = int(y.size)
        else:
            pooled = {'roc_auc': float('nan'), 'eer': float('nan'),
                      'accuracy': float('nan'), 'n': int(y.size)}

        out['results'][k] = {
            'variant': variant, 'seed': int(seed), 'norm': norm,
            'aggregate': agg, 'pooled': pooled, 'per_fold': pf,
        }

    return out


if __name__ == '__main__':
    main()
