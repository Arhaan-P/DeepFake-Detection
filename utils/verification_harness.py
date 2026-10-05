"""
Controlled Verification Harness (future work, plan Sections 2, 9, 14)
====================================================================
One protocol for every experiment in the master matrix (E0-E4, E7-E8), so
that a change in the result can only come from the one variable the
experiment changes: pose backend, feature set, sequence mode, or model.

Protocol (fixed across experiments)
-----------------------------------
* Subject-disjoint leave-one-subject-out folds (13 for GaitDeepfake-13).
* z-score statistics estimated on the training subjects only.
* Strict enrolment: a query clip is never part of the signature it is
  compared against. (`enrol='legacy'` reproduces the original behaviour,
  where the held-out subject's signature averages all of their videos,
  including the query, so the size of that effect can be measured.)
* Equal-size signatures: every evaluation signature, genuine or impostor,
  averages the same number of clips, and training signatures average a
  random 1..k clips. Without this, leave-one-out genuine signatures are
  noisier than full impostor signatures and a verifier can "detect" genuine
  claims by counting averaged clips (observed: see Signatures).
* Training pairs: 1 genuine + 1 impostor claim per training sample, the
  impostor drawn from TRAINING identities only (the held-out subject's
  signature is never shown to the model during training).
* Exhaustive, deterministic test trials: every original clip of the held-out
  subject is scored against every enrolled identity (1 genuine + N-1
  impostor claims). Identical trials across experiments make paired
  comparisons exact.
* Model state selected by lowest training loss, as in the baseline.
* Multiple seeds; per-fold mean +/- std and pooled statistics.

Protocols
---------
loso            signatures built from all enrolment clips (both views)
loso_same_view  signatures from clips of the query's own view only
cross_view_F2S  enrol on frontal clips, query with side clips   (RQ6)
cross_view_S2F  enrol on side clips, query with frontal clips   (RQ6)
cross:<field>:<A>:<B>
                enrol on clips whose metadata <field> == A, query with clips
                where it == B (e.g. cross:camera:phoneA:phoneB,
                cross:lighting:indoor:outdoor). Metadata comes from the
                recording manifest (`metadata_csv`, see
                DOCUMENTATION/DATA_COLLECTION_PROTOCOL.md); `view` is always
                available from the file name.

Author: DeepFake Detection Project
"""

import hashlib
import json
import os
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from glob import glob
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from utils.gait_cycle import dtw
from utils.gait_descriptors import Clip, build_descriptor, clip_from_track
from utils.gait_features import FEATURE_SETS, family_slices
from utils.keypoint_augment import augment_clip, perturb_clip
from utils.pose_backends import PoseTrack
from utils.verification_metrics import (
    apply_platt,
    calibration_summary,
    crossfit_platt,
    eer,
    fit_platt,
    rates_at,
    roc_auc,
    summarize_scores,
)


@dataclass
class ExperimentConfig:
    name: str
    backend: str = "mediapipe_lite"
    source: str = "image"  # image | world
    feature_set: str = "baseline78"
    seq_mode: str = "clip"  # clip | cycle
    model: str = "tcn"
    protocol: str = "loso"
    enrol: str = "strict"  # strict | legacy
    augment: str = "rhythm_safe"
    n_aug: int = 15  # augmented copies per training clip (16x incl. original)
    epochs: int = 30
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.1
    hidden: int = 64
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    seq_len: int = 60
    perturbations: List[str] = field(default_factory=list)
    # pose caches of DEGRADED copies of the same videos (compression, low
    # light...); used for queries only, enrolment and training stay clean
    test_backends: List[str] = field(default_factory=list)
    cache_root: str = "data/pose_cache"
    metadata_csv: str = ""  # recording manifest with per-clip conditions
    subjects: Optional[List[str]] = None
    threads: int = 2
    device: str = "cpu"  # cpu | cuda | auto (cuda when available)
    description: str = ""

    @classmethod
    def from_dict(cls, d: Dict) -> "ExperimentConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    @property
    def families(self) -> List[str]:
        return FEATURE_SETS[self.feature_set]


# ============================================================
# Data
# ============================================================


def load_clips(
    backend: str, source: str = "image", cache_root: str = "data/pose_cache"
) -> List[Clip]:
    files = sorted(glob(os.path.join(cache_root, backend, "*.npz")))
    if not files:
        raise FileNotFoundError(
            f"No pose cache for '{backend}' in {cache_root}. Run "
            f"scripts/future_work/extract_pose_cache.py --backend {backend}"
        )
    clips = []
    for f in files:
        c = clip_from_track(PoseTrack.load(f), os.path.basename(f)[:-4], source)
        if c is not None:
            clips.append(c)
    return clips


class DescriptorBank:
    """Descriptors of every clip x (original + n_aug augmented copies).

    Augmentation is seeded per clip name and copy index, independent of fold
    and model seed, so experiments that share backend/features/augmentation
    see identical training data. Cached on disk.
    """

    def __init__(self, clips: List[Clip], cfg: ExperimentConfig, cache_dir: str):
        self.clips = clips
        self.cfg = cfg
        key = json.dumps(
            [
                cfg.backend,
                cfg.source,
                cfg.families,
                cfg.seq_mode,
                cfg.augment,
                cfg.n_aug,
                cfg.seq_len,
                [c.name for c in clips],
                "v1",
            ]
        )
        digest = hashlib.sha1(key.encode()).hexdigest()[:16]
        path = os.path.join(cache_dir, f"{digest}.npy")
        if os.path.exists(path):
            self.data = np.load(path)
        else:
            self.data = np.stack([self._build(c) for c in clips])
            os.makedirs(cache_dir, exist_ok=True)
            np.save(path, self.data)
        # (n_clips, 1 + n_aug, T, D)

    def _build(self, clip: Clip) -> np.ndarray:
        cfg = self.cfg
        out = [build_descriptor(clip, cfg.families, cfg.seq_mode, cfg.seq_len)]
        seed = int(hashlib.md5(clip.name.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        for _ in range(cfg.n_aug):
            aug = augment_clip(clip, rng, cfg.augment)
            out.append(build_descriptor(aug, cfg.families, cfg.seq_mode, cfg.seq_len))
        return np.stack(out)

    def describe(self, clip: Clip) -> np.ndarray:
        cfg = self.cfg
        return build_descriptor(clip, cfg.families, cfg.seq_mode, cfg.seq_len)


def attach_metadata(clips: List[Clip], metadata_csv: str) -> None:
    """Merge recording-manifest columns into clip.meta, keyed by clip name."""
    import csv

    with open(metadata_csv, newline="") as fh:
        rows = {_file_stem(r["file"]): r for r in csv.DictReader(fh)}
    for c in clips:
        c.meta.update(rows.get(c.name, {}))


def _file_stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _view_filter(protocol: str, role: str, clip) -> bool:
    """Is a clip usable as enrolment ('enrol') or query ('query')?"""
    if protocol == "cross_view_F2S":
        protocol = "cross:view:F:S"
    elif protocol == "cross_view_S2F":
        protocol = "cross:view:S:F"
    if protocol.startswith("cross:"):
        _, fld, a, b = protocol.split(":")
        value = clip.view if fld == "view" else str(clip.meta.get(fld, ""))
        return value == (a if role == "enrol" else b)
    return True


class Signatures:
    """Enrolment signatures: means of normalised original descriptors.

    Equal-size rule. A signature averaged over fewer clips is noisier, so if
    genuine claims used leave-one-out means (n - 1 clips) while impostor
    claims used full means (n clips), a verifier could separate them by
    counting clips instead of comparing gaits. Every evaluation signature --
    genuine or impostor -- therefore averages exactly `k_eval` clips, and
    training signatures average a random 1..k_eval clips, so the number of
    averaged clips carries no label information.
    """

    def __init__(self, clips, originals: np.ndarray, protocol: str):
        self.clips, self.x, self.protocol = clips, originals, protocol
        self.by_id: Dict[str, List[int]] = {}
        for i, c in enumerate(clips):
            if _view_filter(protocol, "enrol", c):
                self.by_id.setdefault(c.identity, []).append(i)
        if protocol == "loso_same_view":
            counts = [
                sum(clips[i].view == v for i in idx)
                for idx in self.by_id.values()
                for v in {clips[i].view for i in idx}
            ]
        else:
            counts = [len(idx) for idx in self.by_id.values()]
        # a genuine test query is excluded from its own signature unless the
        # protocol enrols and queries disjoint views
        disjoint = protocol.startswith("cross")
        self.k_eval = max(1, min(counts) - (0 if disjoint else 1))

    def candidates(
        self, identity: str, exclude: Optional[int] = None, view: Optional[str] = None
    ) -> List[int]:
        idx = self.by_id.get(identity, [])
        if self.protocol == "loso_same_view" and view is not None:
            same = [i for i in idx if self.clips[i].view == view]
            idx = same or idx
        if exclude is not None:
            kept = [i for i in idx if i != exclude]
            idx = kept or idx  # only one enrolment clip: cannot exclude
        return idx

    def get(
        self,
        identity: str,
        exclude: Optional[int] = None,
        view: Optional[str] = None,
        k: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Mean of `k` of the identity's clips (all clips if k is None)."""
        idx = self.candidates(identity, exclude, view)
        if k is not None and len(idx) > k:
            rng = rng or np.random.default_rng(0)
            idx = sorted(rng.choice(idx, size=k, replace=False).tolist())
        return self.x[idx].mean(axis=0)

    def get_eval(self, identity: str, query: int, strict: bool = True) -> np.ndarray:
        """Deterministic evaluation signature for (query clip, claim)."""
        if not strict:  # legacy: all clips, query included (original pipeline)
            return self.get(identity, view=self.clips[query].view)
        key = f"{self.clips[query].name}|{identity}".encode()
        rng = np.random.default_rng(int(hashlib.md5(key).hexdigest()[:8], 16))
        return self.get(
            identity, exclude=query, view=self.clips[query].view, k=self.k_eval, rng=rng
        )


# ============================================================
# Training / scoring
# ============================================================


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but CUDA is not available")
    return torch.device(name)


def environment(cfg) -> Dict:
    """Where an experiment ran: CPU and GPU arithmetic are not bit-identical,
    so every result records its device."""
    dev = resolve_device(cfg.device)
    env = {"device": dev.type, "torch": torch.__version__, "threads": cfg.threads}
    if dev.type == "cuda":
        env["gpu"] = torch.cuda.get_device_name(0)
    return env


def _dtw_score(v: np.ndarray, c: np.ndarray) -> float:
    d, _ = dtw(v, c, band=10)
    return -d


def _make_model(cfg: ExperimentConfig, dim: int):
    from models.verifier_variants import build_verifier

    return build_verifier(
        cfg.model,
        dim,
        family_slices(cfg.families),
        dropout=cfg.dropout,
        hidden=cfg.hidden,
    )


def train_fold(cfg, train_x, train_meta, sigs, train_ids, seed, device="cpu"):
    """Train one verifier. train_x: (N, T, D) normalised descriptors of all
    training copies; train_meta: list of (identity, clip_index)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = _make_model(cfg, train_x.shape[-1]).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    crit = nn.CrossEntropyLoss()

    ids = sorted(set(train_ids))
    # candidate enrolment clips: positives exclude the sample's own clip
    pos_cands = {
        ci: sigs.candidates(ident, exclude=ci, view=sigs.clips[ci].view)
        for ident, ci in set(train_meta)
    }
    neg_cands = {
        (i, v): sigs.candidates(i, view=v)
        for i in ids
        for v in {sigs.clips[ci].view for _, ci in train_meta}
    }
    others = {ident: [i for i in ids if i != ident] for ident in ids}

    def subset_mean(cands):
        k = int(rng.integers(1, sigs.k_eval + 1))
        if len(cands) > k:
            cands = rng.choice(cands, size=k, replace=False)
        return sigs.x[cands].mean(axis=0)

    x = torch.from_numpy(train_x).float().to(device)
    best_loss, best_state = float("inf"), None
    for _ in range(cfg.epochs):
        # fresh random-size signature subsets every epoch (see Signatures)
        pos_sig = np.stack([subset_mean(pos_cands[ci]) for _, ci in train_meta])
        neg_sig = np.stack(
            [
                subset_mean(neg_cands[(rng.choice(others[ident]), sigs.clips[ci].view)])
                for ident, ci in train_meta
            ]
        )
        v = torch.cat([x, x])
        c = torch.from_numpy(np.concatenate([pos_sig, neg_sig])).float().to(device)
        y = torch.cat([torch.ones(len(x)), torch.zeros(len(x))]).long().to(device)
        perm = torch.from_numpy(rng.permutation(len(v))).to(device)
        model.train()
        total, nb = 0.0, 0
        for s in range(0, len(v), cfg.batch_size):
            b = perm[s : s + cfg.batch_size]
            if len(b) < 2:
                continue
            opt.zero_grad()
            loss = crit(model(v[b], c[b]), y[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
            nb += 1
        avg = total / max(nb, 1)
        if avg < best_loss:
            best_loss, best_state = avg, deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return model, best_loss


@torch.no_grad()
def score_pairs(
    model, v: np.ndarray, c: np.ndarray, batch: int = 256, device="cpu"
) -> np.ndarray:
    out = []
    for s in range(0, len(v), batch):
        vb = torch.from_numpy(v[s : s + batch]).float().to(device)
        cb = torch.from_numpy(c[s : s + batch]).float().to(device)
        out.append(F.softmax(model(vb, cb), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


# ============================================================
# Main LOSO loop
# ============================================================


def run_experiment(
    cfg: ExperimentConfig,
    descriptor_cache: str = "data/descriptor_cache",
    log=print,
) -> Dict:
    torch.set_num_threads(cfg.threads)
    device = resolve_device(cfg.device)
    t_start = time.time()
    clips = load_clips(cfg.backend, cfg.source, cfg.cache_root)
    if cfg.metadata_csv:
        attach_metadata(clips, cfg.metadata_csv)
    if cfg.subjects:
        clips = [c for c in clips if c.identity in cfg.subjects]
    bank = DescriptorBank(clips, cfg, descriptor_cache)
    subjects = sorted(set(c.identity for c in clips))
    ident = np.array([c.identity for c in clips])
    n_copies = bank.data.shape[1]
    log(
        f"[{cfg.name}] {len(clips)} clips, {len(subjects)} subjects, "
        f"descriptor {bank.data.shape[2:]} x {n_copies} copies"
    )

    # Perturbed test descriptors (robustness): clean enrolment, degraded query.
    perturbed = {}
    for spec in cfg.perturbations:
        perturbed[spec] = np.stack(
            [bank.describe(perturb_clip(c, spec, seed=k)) for k, c in enumerate(clips)]
        )
    for tb in cfg.test_backends:
        degraded = {c.name: c for c in load_clips(tb, cfg.source, cfg.cache_root)}
        missing = [c.name for c in clips if c.name not in degraded]
        if missing:
            log(f"  WARNING {tb}: {len(missing)} clips missing, using clean ones")
        perturbed[f"video:{tb}"] = np.stack(
            [
                (
                    bank.describe(degraded[c.name])
                    if c.name in degraded
                    else bank.data[k, 0]
                )
                for k, c in enumerate(clips)
            ]
        )

    # trial table (identical for every experiment on the same clip set)
    trials = []
    for qi, c in enumerate(clips):
        if not _view_filter(cfg.protocol, "query", c):
            continue
        for claim in subjects:
            trials.append((qi, claim))
    t_q = np.array([q for q, _ in trials])
    t_claim = np.array([cl for _, cl in trials])
    t_label = (ident[t_q] == t_claim).astype(int)
    conditions = ["clean"] + list(perturbed)
    scores = {
        s: {cond: np.full(len(trials), np.nan) for cond in conditions}
        for s in cfg.seeds
    }
    fold_info = {}
    params = None

    for fi, held in enumerate(subjects, 1):
        tf = time.time()
        tr_clip = np.where(ident != held)[0]
        te_trials = np.where(ident[t_q] == held)[0]
        if len(te_trials) == 0:
            continue
        # normalisation from training subjects (all copies, all timesteps)
        tr = bank.data[tr_clip].reshape(-1, *bank.data.shape[2:])
        mu = tr.reshape(-1, tr.shape[-1]).mean(0)
        sd = tr.reshape(-1, tr.shape[-1]).std(0)
        sd[sd < 1e-6] = 1.0

        def norm(a):
            return ((a - mu) / sd).astype(np.float32)

        originals = norm(bank.data[:, 0])
        sigs = Signatures(clips, originals, cfg.protocol)

        train_x = norm(bank.data[tr_clip]).reshape(-1, *bank.data.shape[2:])
        train_meta = [(ident[ci], ci) for ci in tr_clip for _ in range(n_copies)]

        # claimed signatures for this fold's trials
        strict = cfg.enrol == "strict"
        claim_sig = np.stack(
            [sigs.get_eval(t_claim[k], t_q[k], strict) for k in te_trials]
        )
        query = {"clean": originals[t_q[te_trials]]}
        for spec, arr in perturbed.items():
            query[spec] = norm(arr[t_q[te_trials]])

        losses = []
        for seed in cfg.seeds:
            if cfg.model == "dtw":
                for cond in conditions:
                    scores[seed][cond][te_trials] = [
                        _dtw_score(v, c) for v, c in zip(query[cond], claim_sig)
                    ]
                params = 0
                losses.append(float("nan"))
                continue
            model, loss = train_fold(
                cfg, train_x, train_meta, sigs, ident[tr_clip].tolist(), seed, device
            )
            losses.append(loss)
            if params is None:
                from models.verifier_variants import count_parameters

                params = count_parameters(model)
            for cond in conditions:
                scores[seed][cond][te_trials] = score_pairs(
                    model, query[cond], claim_sig, device=device
                )
        y = t_label[te_trials]
        aucs = [roc_auc(y, scores[s]["clean"][te_trials]) for s in cfg.seeds]
        fold_info[held] = {"train_loss": losses, "seconds": time.time() - tf}
        log(
            f"  fold {fi:2d}/{len(subjects)} {held:<9s} "
            f"AUC "
            + " ".join(f"{a:.3f}" for a in aucs)
            + f"  ({time.time() - tf:.0f}s)"
        )

    return _summarise(
        cfg,
        clips,
        subjects,
        trials,
        t_label,
        t_q,
        t_claim,
        scores,
        fold_info,
        params,
        time.time() - t_start,
    )


def _summarise(
    cfg,
    clips,
    subjects,
    trials,
    labels,
    t_q,
    t_claim,
    scores,
    fold_info,
    params,
    seconds,
) -> Dict:
    ident = np.array([c.identity for c in clips])
    views = np.array([c.view for c in clips])
    groups = ident[t_q]
    out = {
        "config": asdict(cfg),
        "params": params,
        "runtime_s": seconds,
        "environment": environment(cfg),
        "n_trials": len(trials),
        "subjects": subjects,
        "folds": fold_info,
        "per_seed": {},
    }
    fold_rows = []
    for seed, conds in scores.items():
        s = conds["clean"]
        ok = np.isfinite(s)
        per_fold = {}
        for subj in subjects:
            m = ok & (groups == subj)
            if m.any() and len(np.unique(labels[m])) > 1:
                e, _ = eer(labels[m], s[m])
                per_fold[subj] = {"roc_auc": roc_auc(labels[m], s[m]), "eer": e}
                fold_rows.append((seed, subj, per_fold[subj]["roc_auc"], e))
        pooled = summarize_scores(labels[ok], s[ok])
        # probabilities: DTW scores are distances, map them with Platt first
        probs = s[ok]
        if cfg.model == "dtw":
            probs = apply_platt(
                1 / (1 + np.exp(-probs)),
                fit_platt(labels[ok], 1 / (1 + np.exp(-probs))),
            )
        cal_raw = calibration_summary(labels[ok], probs)
        cal_cf = calibration_summary(
            labels[ok], crossfit_platt(labels[ok], probs, groups[ok])
        )
        robust = {}
        for cond, sc in conds.items():
            if cond == "clean":
                continue
            m = np.isfinite(sc)
            robust[cond] = {
                "roc_auc": roc_auc(labels[m], sc[m]),
                "eer": eer(labels[m], sc[m])[0],
            }
        by_view = {}
        for v in sorted(set(views)):
            m = ok & (views[t_q] == v)
            if m.any() and len(np.unique(labels[m])) > 1:
                by_view[v] = {
                    "roc_auc": roc_auc(labels[m], s[m]),
                    "eer": eer(labels[m], s[m])[0],
                    "n": int(m.sum()),
                }
        out["per_seed"][str(seed)] = {
            "per_fold": per_fold,
            "pooled": pooled,
            "calibration_raw": {k: v for k, v in cal_raw.items() if k != "bins"},
            "calibration_crossfit_platt": {
                k: v for k, v in cal_cf.items() if k != "bins"
            },
            "reliability_raw": cal_raw["bins"],
            "by_view": by_view,
            "robustness": robust,
        }

    aucs = np.array([r[2] for r in fold_rows])
    eers = np.array([r[3] for r in fold_rows])
    seeds = list(scores)
    agg = {
        "fold_auc_mean": float(np.nanmean(aucs)),
        "fold_auc_std": float(np.nanstd(aucs)),
        "fold_eer_mean": float(np.nanmean(eers)),
        "fold_eer_std": float(np.nanstd(eers)),
        "n_fold_obs": len(aucs),
    }
    for key in ("roc_auc", "eer", "tpr_at_fpr_5"):
        vals = [out["per_seed"][str(s)]["pooled"][key] for s in seeds]
        agg[f"pooled_{key}_mean"] = float(np.mean(vals))
        agg[f"pooled_{key}_std"] = float(np.std(vals))
    for key in ("ece", "brier"):
        for which in ("calibration_raw", "calibration_crossfit_platt"):
            vals = [out["per_seed"][str(s)][which][key] for s in seeds]
            agg[f"{which}_{key}"] = float(np.mean(vals))
    if any(out["per_seed"][str(s)]["robustness"] for s in seeds):
        agg["robustness_auc"] = {
            cond: float(
                np.mean(
                    [
                        out["per_seed"][str(s)]["robustness"][cond]["roc_auc"]
                        for s in seeds
                    ]
                )
            )
            for cond in out["per_seed"][str(seeds[0])]["robustness"]
        }
        agg["robustness_eer"] = {
            cond: float(
                np.mean(
                    [out["per_seed"][str(s)]["robustness"][cond]["eer"] for s in seeds]
                )
            )
            for cond in out["per_seed"][str(seeds[0])]["robustness"]
        }
    out["aggregate"] = agg
    out["fold_table"] = [
        {"seed": int(r[0]), "subject": r[1], "roc_auc": r[2], "eer": r[3]}
        for r in fold_rows
    ]
    out["failure_analysis"] = _failures(
        labels, scores, groups, t_claim, views[t_q], clips, t_q
    )
    # compact trial-level record for paired re-analysis
    out["trials"] = {
        "query": [clips[q].name for q in t_q],
        "claim": list(map(str, t_claim)),
        "label": labels.tolist(),
        "scores": {
            str(s): {c: np.round(v, 6).tolist() for c, v in conds.items()}
            for s, conds in scores.items()
        },
    }
    return out


def _failures(labels, scores, groups, claims, qviews, clips, t_q) -> Dict:
    """Seed-averaged score -> per-subject FRR/FAR at the pooled EER threshold,
    most-confusable impostor pairs, and the worst genuine clips."""
    s = np.nanmean(np.stack([v["clean"] for v in scores.values()]), axis=0)
    _, thr = eer(labels, s)
    per_subject = {}
    for subj in np.unique(groups):
        g = (groups == subj) & (labels == 1)
        imp_as_claim = (claims == subj) & (labels == 0)
        per_subject[str(subj)] = {
            "frr": float(np.mean(s[g] < thr)) if g.any() else float("nan"),
            "far_as_claimed": (
                float(np.mean(s[imp_as_claim] >= thr))
                if imp_as_claim.any()
                else float("nan")
            ),
        }
    pair = {}
    for k in np.where(labels == 0)[0]:
        key = f"{groups[k]}->{claims[k]}"
        pair.setdefault(key, []).append(s[k])
    confusable = sorted(
        ((k, float(np.mean(v))) for k, v in pair.items()), key=lambda kv: -kv[1]
    )[:10]
    gen = np.where(labels == 1)[0]
    worst = sorted(gen, key=lambda k: s[k])[:8]
    by_view = {
        str(v): rates_at(labels[qviews == v], s[qviews == v], thr)
        for v in np.unique(qviews)
    }
    return {
        "eer_threshold": float(thr),
        "per_subject": per_subject,
        "most_confusable_impostor_pairs": [
            {"query_subject->claim": k, "mean_score": v} for k, v in confusable
        ],
        "hardest_genuine_clips": [
            {"clip": clips[t_q[k]].name, "score": float(s[k])} for k in worst
        ],
        "rates_by_view_at_eer_threshold": by_view,
    }


def save_result(result: Dict, path: str, keep_trials: bool = True) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not keep_trials:
        result = {k: v for k, v in result.items() if k != "trials"}
    with open(path, "w") as fh:
        json.dump(result, fh, indent=1, default=float)
