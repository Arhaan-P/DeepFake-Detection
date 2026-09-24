"""
Deployable Gait Verifier (future work, plan Sections 10-11 / P7-P8)
==================================================================
A trained verifier plus everything needed to use it outside the LOSO
harness, in one checkpoint:

  model state + experiment config (backend, feature set, sequence mode)
  z-score statistics, per-identity enrolment signatures
  decision threshold and Platt calibration taken from the LOSO evaluation
  of the same configuration (never from the enrolled subjects' own scores)
  per-identity gait-parameter statistics for interpretable sub-scores

Verdicts (plan Section 11, "unknown identity handling" and scope boundary):

  AUTHENTIC            gait consistent with the claimed enrolled identity
  IDENTITY_MISMATCH    gait inconsistent with the claim -> possible face-swap;
                       the best-matching enrolled identity is reported as the
                       likely body source
  UNKNOWN_IDENTITY     the claimed identity is not enrolled; the system
                       refuses rather than guessing
  INSUFFICIENT_GAIT    too few frames / strides to verify

The output is a verification of consistency with a claimed identity, not a
universal detector of AI-generated video.

Author: DeepFake Detection Project
"""

import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from utils.gait_cycle import GaitCycleAnalysis, analyse_gait
from utils.gait_descriptors import Clip, build_descriptor
from utils.gait_features import family_slices, leg_length
from utils.verification_harness import ExperimentConfig
from utils.verification_metrics import apply_platt

PARAM_NAMES = list(GaitCycleAnalysis.RHYTHM_NAMES) + list(
    GaitCycleAnalysis.STRIDE_NAMES
)
MIN_FRAMES = 45  # ~1.5 s at 30 fps
MIN_STRIDES = 1


def gait_parameters(clip: Clip) -> Dict[str, float]:
    ga = analyse_gait(clip.pose_iso, clip.t, clip.fps)
    d = ga.rhythm_dict()
    d.update(ga.stride_dict(leg_length(clip.pose_iso)))
    d["_n_strides"] = float(ga.n_cycles)
    d["_period_confidence"] = float(ga.period_confidence)
    return d


class GaitVerifier:
    def __init__(self, bundle: Dict):
        self.cfg = ExperimentConfig.from_dict(bundle["config"])
        from models.verifier_variants import build_verifier

        self.model = build_verifier(
            self.cfg.model,
            int(bundle["input_dim"]),
            family_slices(self.cfg.families),
            dropout=self.cfg.dropout,
            hidden=self.cfg.hidden,
        )
        self.model.load_state_dict(bundle["model_state"])
        self.model.eval()
        self.mu = np.asarray(bundle["mu"], dtype=np.float32)
        self.sd = np.asarray(bundle["sd"], dtype=np.float32)
        self.signatures: Dict[str, np.ndarray] = {
            k: np.asarray(v, dtype=np.float32) for k, v in bundle["signatures"].items()
        }
        self.threshold = float(bundle["threshold"])
        self.platt = tuple(bundle["platt"]) if bundle.get("platt") else None
        self.param_stats = bundle.get("param_stats", {})
        self.meta = bundle.get("meta", {})

    # ---------------- persistence ----------------

    @classmethod
    def load(cls, path: str) -> "GaitVerifier":
        return cls(torch.load(path, map_location="cpu", weights_only=False))

    @staticmethod
    def save_bundle(bundle: Dict, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(bundle, path)

    @property
    def identities(self) -> List[str]:
        return sorted(self.signatures)

    # ---------------- scoring ----------------

    def describe(self, clip: Clip) -> np.ndarray:
        d = build_descriptor(
            clip, self.cfg.families, self.cfg.seq_mode, self.cfg.seq_len
        )
        return ((d - self.mu) / self.sd).astype(np.float32)

    def _prob(self, raw: np.ndarray) -> np.ndarray:
        return apply_platt(raw, self.platt) if self.platt else raw

    @torch.no_grad()
    def raw_scores(self, query: np.ndarray, claims: List[str]) -> np.ndarray:
        v = torch.from_numpy(np.repeat(query[None], len(claims), axis=0))
        c = torch.from_numpy(np.stack([self.signatures[k] for k in claims]))
        return F.softmax(self.model(v, c), dim=1)[:, 1].numpy()

    def verify(self, clip: Clip, claimed: str) -> Dict:
        out = {
            "claimed_identity": claimed,
            "threshold": self.threshold,
            "frames": len(clip.t),
            "duration_s": clip.duration,
        }
        if claimed not in self.signatures:
            out.update(
                verdict="UNKNOWN_IDENTITY",
                message=f"'{claimed}' is not enrolled; verification refused.",
                enrolled=self.identities,
            )
            return out
        params = gait_parameters(clip)
        out["n_strides"] = int(params["_n_strides"])
        if len(clip.t) < MIN_FRAMES or params["_n_strides"] < MIN_STRIDES:
            out.update(
                verdict="INSUFFICIENT_GAIT",
                message="Too few frames or strides for a reliable gait comparison.",
            )
            return out
        q = self.describe(clip)
        names = self.identities
        raw = self.raw_scores(q, names)
        by_id = dict(zip(names, raw.tolist()))
        s = by_id[claimed]
        best = max(by_id, key=by_id.get)
        out.update(
            score_raw=s,
            probability_authentic=float(self._prob(np.array([s]))[0]),
            verdict="AUTHENTIC" if s >= self.threshold else "IDENTITY_MISMATCH",
            best_match=best,
            best_match_score=by_id[best],
            scores_all_identities=by_id,
            gait_parameters={k: v for k, v in params.items() if not k.startswith("_")},
        )
        if out["verdict"] == "IDENTITY_MISMATCH" and best != claimed:
            out["likely_body_source"] = best if by_id[best] >= self.threshold else None
        return out

    def identify(self, clip: Clip) -> Dict[str, float]:
        raw = self.raw_scores(self.describe(clip), self.identities)
        return dict(sorted(zip(self.identities, raw.tolist()), key=lambda kv: -kv[1]))

    # ---------------- explanation (P7) ----------------

    def sub_scores(self, params: Dict[str, float], claimed: str) -> Dict[str, Dict]:
        """Interpretable agreement per gait parameter: z = (query - claimed
        mean) / pooled within-subject SD, agreement = exp(-z^2 / 2). Each
        entry carries the parameter's test-retest ICC so unreliable
        parameters can be discounted by the reader."""
        st = self.param_stats.get("per_identity", {}).get(claimed)
        pooled = self.param_stats.get("within_sd", {})
        icc = self.param_stats.get("icc", {})
        out = {}
        if not st:
            return out
        for p in PARAM_NAMES:
            if p not in params or p not in st or not pooled.get(p):
                continue
            z = (params[p] - st[p]) / pooled[p]
            out[p] = {
                "query": float(params[p]),
                "enrolled_mean": float(st[p]),
                "z": float(z),
                "agreement": float(np.exp(-0.5 * z * z)),
                "icc": float(icc.get(p, float("nan"))),
            }
        return out

    def explain(self, clip: Clip, claimed: str, window: int = 10, hop: int = 5) -> Dict:
        """Joint / family attribution, temporal localisation and counterfactual
        family removal for one (clip, claim) decision."""
        q = self.describe(clip)
        c = self.signatures[claimed]
        slices = family_slices(self.cfg.families)

        # gradient x input on the query (log-odds of AUTHENTIC)
        v = torch.from_numpy(q[None]).requires_grad_(True)
        logits = self.model(v, torch.from_numpy(c[None]))
        (logits[0, 1] - logits[0, 0]).backward()
        gxi = (v.grad[0] * v[0]).detach().numpy()  # (T, D)
        mag = np.abs(gxi)

        fam = {f: float(mag[:, s].sum()) for f, s in slices.items()}
        tot = sum(fam.values()) + 1e-12
        fam = {f: x / tot for f, x in fam.items()}

        joints = np.zeros(12)
        for f in ("coords", "velocity", "coords_scaled", "acceleration", "jerk"):
            if f in slices:
                joints += mag[:, slices[f]].reshape(len(q), 12, 3).sum(axis=(0, 2))
        if "angles" in slices:
            a = mag[:, slices["angles"]].sum(0)
            for k, j in enumerate([4, 5, 2, 3, 6, 7]):
                joints[j] += a[k]
        joints = joints / (joints.max() + 1e-12)

        base = float(self.raw_scores(q, [claimed])[0])

        # temporal localisation: make one window agree with the claim
        windows = []
        grid_t = np.interp(
            np.linspace(0, len(clip.t) - 1, len(q)), np.arange(len(clip.t)), clip.t
        )
        for s in range(0, len(q) - window + 1, hop):
            qq = q.copy()
            qq[s : s + window] = c[s : s + window]
            windows.append(
                {
                    "t_start_s": float(grid_t[s] - clip.t[0]),
                    "t_end_s": float(grid_t[s + window - 1] - clip.t[0]),
                    "delta_score_if_matched": float(
                        self.raw_scores(qq, [claimed])[0] - base
                    ),
                }
            )

        # counterfactual: replace one family with the claimed identity's
        counterfactual = {}
        for f, s in slices.items():
            qq = q.copy()
            qq[:, s] = c[:, s]
            counterfactual[f] = float(self.raw_scores(qq, [claimed])[0] - base)

        from utils.pose_backends import GAIT_JOINT_NAMES

        return {
            "score_raw": base,
            "family_attribution": fam,
            "joint_attribution": dict(zip(GAIT_JOINT_NAMES, map(float, joints))),
            "temporal_windows": windows,
            "counterfactual_family_swap": counterfactual,
        }


def build_param_stats(clips: List[Clip], params: Optional[List[Dict]] = None) -> Dict:
    """Per-identity parameter means, pooled within-subject SD and ICC(1,1)."""
    params = params or [gait_parameters(c) for c in clips]
    ids = np.array([c.identity for c in clips])
    per_id, within, icc = {}, {}, {}
    for p in PARAM_NAMES:
        x = np.array([d[p] for d in params], dtype=float)
        res = []
        for g in np.unique(ids):
            xs = x[ids == g]
            per_id.setdefault(g, {})[p] = float(xs.mean())
            if len(xs) > 1:
                res.extend(xs - xs.mean())
        within[p] = float(np.sqrt(np.mean(np.square(res)))) if res else float("nan")
        # ICC(1,1), unbalanced
        a, n = len(np.unique(ids)), len(x)
        sizes = np.array([np.sum(ids == g) for g in np.unique(ids)])
        means = np.array([x[ids == g].mean() for g in np.unique(ids)])
        msb = np.sum(sizes * (means - x.mean()) ** 2) / (a - 1)
        msw = np.sum(np.square(res)) / max(n - a, 1)
        k0 = (n - np.sum(sizes**2) / n) / (a - 1)
        icc[p] = float((msb - msw) / (msb + (k0 - 1) * msw + 1e-12))
    return {"per_identity": per_id, "within_sd": within, "icc": icc}
