"""Diversity-aware re-ranking (maximal marginal relevance) and simple "because you liked" explanations.

MMR picks the next item by   lam * relevance - (1 - lam) * (largest similarity to items already picked).
Relevance is the ranker score rescaled to 0..1 inside the candidate pool, similarity is the cosine of the
two-tower item vectors. lam = 1 returns the ranker order unchanged.
"""
from __future__ import annotations

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def mmr(scores: np.ndarray, vectors: np.ndarray, k: int, lam: float, pool: int = 50) -> np.ndarray:
    """Return positions (into `scores`) of k items, in the order chosen."""
    if not 0.0 <= lam <= 1.0:
        raise ValueError("diversity lambda must be between 0 and 1")
    order = np.argsort(-scores, kind="stable")[: max(k, pool)]
    if lam >= 1.0 or len(order) <= 1:
        return order[:k]
    s = scores[order].astype(np.float64)
    span = s.max() - s.min()
    rel = (s - s.min()) / span if span > 0 else np.ones_like(s)
    unit = _unit(vectors[order].astype(np.float64))
    sim = np.clip(unit @ unit.T, 0.0, 1.0)         # opposite items are 'not similar', not a bonus
    chosen = [0]                                   # the ranker's best item always leads
    best_sim = sim[0].copy()
    while len(chosen) < min(k, len(order)):
        gain = lam * rel - (1.0 - lam) * best_sim
        gain[chosen] = -np.inf
        nxt = int(np.argmax(gain))
        chosen.append(nxt)
        best_sim = np.maximum(best_sim, sim[nxt])
    return order[chosen]


def because(item_vectors: np.ndarray, rec_idx: np.ndarray, hist_idx: np.ndarray) -> list[int]:
    """For each recommended item, the catalog index of the liked item it is closest to (two-tower cosine)."""
    hist_idx = np.asarray([h for h in hist_idx if h > 0], dtype=np.int64)
    if not len(hist_idx):
        return [0] * len(rec_idx)
    h = _unit(item_vectors[hist_idx])
    r = _unit(item_vectors[np.asarray(rec_idx, dtype=np.int64)])
    return [int(hist_idx[j]) for j in (r @ h.T).argmax(axis=1)]
