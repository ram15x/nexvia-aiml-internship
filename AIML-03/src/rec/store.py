"""Serving-side data: per-user history and rated-item lists, and ranker statistics, all read from the Spark Parquet tables."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .features import Catalog
from .ranking import Stats


def _lookup(catalog: Catalog) -> np.ndarray:
    lk = np.zeros(int(catalog.item_ids.max()) + 1, np.int32)
    lk[catalog.item_ids[1:]] = np.arange(1, catalog.size, dtype=np.int32)
    return lk


class HistoryStore:
    """user_history (likes, time order) and user_seen (everything rated) as catalog indices."""

    def __init__(self, hist: dict, seen: dict):
        self._hist, self._seen = hist, seen

    @classmethod
    def from_spark(cls, features_dir, catalog: Catalog) -> "HistoryStore":
        d, lk = Path(features_dir), _lookup(catalog)

        def load(name, col):
            df = pd.read_parquet(d / name)
            out = {}
            for u, items in zip(df["userId"].to_numpy(), df[col]):
                arr = np.asarray(items, dtype=np.int64)
                arr = lk[arr[arr < len(lk)]]
                out[int(u)] = arr[arr > 0]
            return out
        return cls(load("user_history", "history"), load("user_seen", "seen"))

    def known(self, user: int) -> bool:
        return int(user) in self._hist or int(user) in self._seen

    def count(self, user: int) -> int:
        return len(self._hist.get(int(user), ()))

    def latest(self, user: int, n: int) -> np.ndarray:
        h = self._hist.get(int(user), np.empty(0, np.int32))[-n:]
        return np.concatenate([np.zeros(n - len(h), np.int32), h.astype(np.int32)])

    def seen(self, user: int) -> np.ndarray:
        return np.union1d(self._seen.get(int(user), np.empty(0, np.int32)), self._hist.get(int(user), np.empty(0, np.int32))).astype(np.int64)


def stats_from_spark(features_dir, catalog: Catalog, max_user: int) -> Stats:
    """Ranker statistics from the Spark user_features / item_features tables (train window only)."""
    d, lk = Path(features_dir), _lookup(catalog)
    u = pd.read_parquet(d / "user_features"); i = pd.read_parquet(d / "item_features")
    um, un, up = (np.zeros(max_user + 1, np.float32) for _ in range(3))
    uid = u["userId"].to_numpy()
    um[uid], un[uid], up[uid] = u["mean_rating"], u["n_ratings"], u["positive_rate"]
    n = catalog.size
    im, inn, ip = (np.zeros(n, np.float32) for _ in range(3))
    idx = lk[i["movieId"].to_numpy()]
    ok = idx > 0
    im[idx[ok]], inn[idx[ok]], ip[idx[ok]] = i["mean_rating"].to_numpy()[ok], i["n_ratings"].to_numpy()[ok], i["positive_rate"].to_numpy()[ok]
    return Stats(um, un, up, im, inn, ip)
