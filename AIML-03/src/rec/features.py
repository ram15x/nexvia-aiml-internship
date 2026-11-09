"""Item features and user-history windows shared by training, evaluation and serving."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import Ratings

GENRES = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama",
          "Fantasy", "Film-Noir", "Horror", "IMAX", "Musical", "Mystery", "Romance", "Sci-Fi",
          "Thriller", "War", "Western", "(no genres listed)"]
GENRE_INDEX = {g: i for i, g in enumerate(GENRES)}
YEAR_MIN, YEAR_MAX = 1900, 2019
N_YEAR_BUCKETS = (YEAR_MAX - YEAR_MIN) // 5 + 2     # 5-year buckets, 0 = unknown
N_POP_BUCKETS = 24


def parse_year(title: str) -> int | None:
    m = re.search(r"\((\d{4})\)\s*$", title)
    return int(m.group(1)) if m else None


@dataclass
class Catalog:
    """Item vocabulary built from the TRAIN window only. Index 0 is padding / unknown."""
    item_ids: np.ndarray        # index -> movieId (index 0 is 0)
    index_of: dict              # movieId -> index
    genres: np.ndarray          # [n+1, 20] float32 multi-hot
    year_bucket: np.ndarray     # [n+1] int32
    pop_bucket: np.ndarray      # [n+1] int32
    train_count: np.ndarray     # [n+1] positives in train
    titles: dict

    @property
    def size(self) -> int:
        return len(self.item_ids)


def build_catalog(train: Ratings, movies_csv: Path) -> Catalog:
    movies = pd.read_csv(movies_csv).set_index("movieId")
    ids, counts = np.unique(train.item, return_counts=True)
    pos_ids, pos_counts = np.unique(train.item[train.positive], return_counts=True)
    pos = dict(zip(pos_ids.tolist(), pos_counts.tolist()))
    item_ids = np.concatenate([[0], ids]).astype(np.int64)
    index_of = {int(m): k for k, m in enumerate(item_ids) if k > 0}
    n = len(item_ids)
    genres = np.zeros((n, len(GENRES)), np.float32)
    year = np.zeros(n, np.int32)
    popc = np.zeros(n, np.int64)
    titles = {}
    for k in range(1, n):
        m = int(item_ids[k])
        row = movies.loc[m]
        titles[m] = row["title"]
        for g in str(row["genres"]).split("|"):
            if g in GENRE_INDEX:
                genres[k, GENRE_INDEX[g]] = 1.0
        y = parse_year(row["title"])
        year[k] = 0 if y is None else 1 + (min(max(y, YEAR_MIN), YEAR_MAX) - YEAR_MIN) // 5
        popc[k] = pos.get(m, 0)
    # log-spaced popularity buckets from train-window positive counts (0 positives -> bucket 0)
    pop_bucket = np.where(popc > 0, 1 + np.minimum(np.log2(popc + 1) * 1.6, N_POP_BUCKETS - 2), 0).astype(np.int32)
    return Catalog(item_ids, index_of, genres, year, pop_bucket, popc.astype(np.int64), titles)


class UserHistories:
    """Per-user positive items in time order, in one flat array, for fast history windows."""

    def __init__(self, ratings: Ratings, catalog: Catalog, max_user: int):
        pos = ratings.positive
        u, it, t = ratings.user[pos], ratings.item[pos], ratings.ts[pos]
        idx = np.fromiter((catalog.index_of.get(int(m), 0) for m in it), dtype=np.int32, count=len(it)) \
            if len(it) < 1000 else _map_items(it, catalog)
        keep = idx > 0
        u, idx, t = u[keep], idx[keep], t[keep]
        order = np.lexsort((t, u))
        self.user = u[order]
        self.item_idx = idx[order]
        self.ts = t[order]
        counts = np.bincount(self.user, minlength=max_user + 1)
        self.start = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
        self.count = counts.astype(np.int64)

    def window(self, positions: np.ndarray, starts: np.ndarray, n: int) -> np.ndarray:
        """History of length n (left padded with 0) ending just before each absolute position."""
        off = np.arange(n, 0, -1)[None, :]
        idx = positions[:, None] - off
        valid = idx >= starts[:, None]
        out = np.where(valid, self.item_idx[np.clip(idx, 0, len(self.item_idx) - 1)], 0)
        return out.astype(np.int32)

    def latest(self, users: np.ndarray, n: int) -> np.ndarray:
        end = self.start[users] + self.count[users]
        return self.window(end, self.start[users], n)


def _map_items(items: np.ndarray, catalog: Catalog) -> np.ndarray:
    lookup = np.zeros(int(max(catalog.item_ids.max(), items.max())) + 1, np.int32)
    lookup[catalog.item_ids[1:]] = np.arange(1, catalog.size, dtype=np.int32)
    return lookup[items]


def genre_profile(history_idx: np.ndarray, catalog: Catalog) -> np.ndarray:
    """Mean genre vector of the items in each history row (padding ignored)."""
    g = catalog.genres[history_idx]                       # [B, N, 20]
    w = (history_idx > 0).astype(np.float32)[..., None]
    return (g * w).sum(1) / np.maximum(w.sum(1), 1.0)
