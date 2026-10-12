"""Claimed-subject similarity. Only the references handed in are ever scored.

Cosine similarity is an uncalibrated score, never a probability. Only the PRD
baseline (`quality_weighted_centroid`) can back an operational policy; the other
strategies are development-only candidates and carry their own version string.
"""

from dataclasses import dataclass
import numpy as np

BASELINE = "quality_weighted_centroid"
DEVELOPMENT_ONLY = ("mean_cosine", "max_cosine", "top2_mean_cosine")
STRATEGIES = (BASELINE,) + DEVELOPMENT_ONLY


@dataclass(frozen=True)
class ScoreResult:
    score: float
    strategy: str
    reference_count: int


def prototype(templates):
    """Normalized quality-weighted centroid of per-original unit embeddings."""
    z = np.stack([t.embedding.normalized() for t in templates]).astype(np.float64)
    out = np.average(z, axis=0, weights=[t.quality_weight for t in templates])
    norm = np.linalg.norm(out)
    if norm < 1e-12:  # e.g. exactly opposed references cancel
        raise ValueError("Degenerate enrollment prototype")
    return (out / norm).astype(np.float32)


def score_claim(probe, templates, strategy=BASELINE):
    if strategy not in STRATEGIES:
        raise ValueError("Unknown scoring strategy")
    if not templates:
        raise ValueError("No references to score")
    q = probe.normalized()
    refs = np.stack([t.embedding.normalized() for t in templates])
    if refs.shape[1] != q.shape[0]:
        raise ValueError("Embedding dimension mismatch")
    if strategy == BASELINE:
        value = float(np.dot(q, prototype(templates)))
    else:
        sims = np.sort(refs.astype(np.float64) @ q.astype(np.float64))[::-1]
        value = {
            "mean_cosine": sims.mean(),
            "max_cosine": sims[0],
            "top2_mean_cosine": sims[:2].mean(),
        }[strategy]
    return ScoreResult(float(np.clip(value, -1, 1)), strategy, len(templates))
