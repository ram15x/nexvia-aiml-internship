"""Fallbacks for users the personalised path cannot serve (no usable history)."""
from __future__ import annotations

import numpy as np

from .features import GENRE_INDEX, Catalog


def popular(catalog: Catalog, k: int, exclude: np.ndarray | None = None, genres: list[str] | None = None) -> np.ndarray:
    """Most popular catalog indices (train-window likes). With `genres`, restrict to items sharing at least one of them
    and order by popularity times the share of requested genres matched (a content-based cold start)."""
    score = catalog.train_count.astype(np.float64).copy()
    score[0] = -1
    if genres:
        cols = [GENRE_INDEX[g] for g in genres if g in GENRE_INDEX]
        if cols:
            match = catalog.genres[:, cols].sum(1) / len(cols)
            score = np.where(match > 0, score * match, -1.0)
            score[0] = -1
    if exclude is not None and len(exclude):
        score[np.asarray(exclude, dtype=np.int64)] = -1
    top = np.argpartition(-score, min(k, len(score) - 1))[:k]
    top = top[np.argsort(-score[top], kind="stable")]
    return top[score[top] >= 0]


def similar_to(catalog: Catalog, seed_idx: np.ndarray, k: int, exclude: np.ndarray | None = None) -> np.ndarray:
    """Content-based: popular items whose genre vector is close (cosine) to the seed items' mean genre vector."""
    seed_idx = np.asarray([s for s in seed_idx if s > 0], dtype=np.int64)
    if not len(seed_idx):
        return popular(catalog, k, exclude)
    prof = catalog.genres[seed_idx].mean(0)
    g = catalog.genres
    cos = (g @ prof) / (np.linalg.norm(g, axis=1) * np.linalg.norm(prof) + 1e-9)
    score = cos * np.log1p(catalog.train_count)
    score[0] = -1
    ban = np.concatenate([seed_idx, np.asarray(exclude if exclude is not None else [], dtype=np.int64)])
    score[ban.astype(np.int64)] = -1
    top = np.argpartition(-score, min(k, len(score) - 1))[:k]
    return top[np.argsort(-score[top], kind="stable")]
