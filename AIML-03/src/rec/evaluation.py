"""Top-K evaluation: Recall@K and NDCG@K with binary relevance, averaged over users.

Relevant items for a user are items they rated >= 4.0 in the evaluation window that they
had not already rated in the history window. Items already rated are never recommended.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Ratings


@dataclass
class EvalSet:
    users: np.ndarray                 # user ids that have at least one relevant item
    relevant: dict                    # user -> np.ndarray of relevant item ids
    seen: dict                        # user -> np.ndarray of item ids rated in the history window


def _group(users: np.ndarray, items: np.ndarray) -> dict:
    order = np.argsort(users, kind="stable")
    u, i = users[order], items[order]
    cut = np.flatnonzero(np.diff(u)) + 1
    return {int(g[0]): v for g, v in zip(np.split(u, cut), np.split(i, cut)) if len(g)}


def build_eval_set(history: Ratings, target: Ratings) -> EvalSet:
    seen = _group(history.user, history.item)
    pos = target.positive
    tgt = _group(target.user[pos], target.item[pos])
    relevant = {}
    for u, items in tgt.items():
        s = seen.get(u)
        keep = items if s is None else items[~np.isin(items, s)]
        if len(keep):
            relevant[u] = np.unique(keep)
    users = np.array(sorted(relevant), dtype=np.int64)
    return EvalSet(users=users, relevant=relevant, seen=seen)


def recall_ndcg(recommended: np.ndarray, relevant: np.ndarray, k: int) -> tuple[float, float]:
    top = recommended[:k]
    hits = np.isin(top, relevant)
    recall = hits.sum() / len(relevant)
    discounts = 1.0 / np.log2(np.arange(2, len(top) + 2))
    dcg = float((hits * discounts).sum())
    ideal = float((1.0 / np.log2(np.arange(2, min(len(relevant), k) + 2))).sum())
    return float(recall), dcg / ideal if ideal > 0 else 0.0


def evaluate(recommend, eval_set: EvalSet, users: np.ndarray, ks=(10,), max_k: int | None = None) -> dict:
    """recommend(user_id, seen_items, n) -> array of up to n item ids, best first."""
    max_k = max_k or max(ks)
    acc = {k: [] for k in ks}
    for u in users:
        recs = np.asarray(recommend(int(u), eval_set.seen.get(int(u), np.empty(0, dtype=np.int32)), max_k))
        rel = eval_set.relevant[int(u)]
        for k in ks:
            acc[k].append(recall_ndcg(recs, rel, k))
    out = {"users": int(len(users))}
    for k in ks:
        a = np.array(acc[k])
        out[f"recall@{k}"] = float(a[:, 0].mean())
        out[f"ndcg@{k}"] = float(a[:, 1].mean())
    return out


def bootstrap_ci(values: np.ndarray, n: int = 1000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
