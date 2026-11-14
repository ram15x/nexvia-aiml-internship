"""Stage two: re-score retrieved candidates with a deep model.

Both rankers see only features that exist for every catalog item and every user with history,
plus the retrieval score and rank. They never rely on an id vocabulary the retriever lacks,
so a candidate cannot be "unsupported".
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from dataclasses import dataclass

import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs

from .data import Ratings
from .features import N_POP_BUCKETS, N_YEAR_BUCKETS, Catalog, genre_profile

keras = tf.keras
N_DENSE = 20 + 20 + 9   # item genres, user*item genre product, scalar features (see make_features)


@dataclass
class Stats:
    user_mean: np.ndarray   # mean rating given (train window)
    user_n: np.ndarray
    user_pos_rate: np.ndarray
    item_mean: np.ndarray   # indexed by catalog index
    item_n: np.ndarray
    item_pos_rate: np.ndarray


def build_stats(train: Ratings, catalog: Catalog, max_user: int) -> Stats:
    r = train.rating2.astype(np.float64) / 2.0
    pos = train.positive.astype(np.float64)
    un = np.bincount(train.user, minlength=max_user + 1).astype(np.float64)
    um = np.bincount(train.user, weights=r, minlength=max_user + 1) / np.maximum(un, 1)
    up = np.bincount(train.user, weights=pos, minlength=max_user + 1) / np.maximum(un, 1)
    lookup = np.zeros(int(catalog.item_ids.max()) + 1, np.int64)
    lookup[catalog.item_ids[1:]] = np.arange(1, catalog.size)
    ii = lookup[train.item]
    n = catalog.size
    inn = np.bincount(ii, minlength=n).astype(np.float64)
    im = np.bincount(ii, weights=r, minlength=n) / np.maximum(inn, 1)
    ip = np.bincount(ii, weights=pos, minlength=n) / np.maximum(inn, 1)
    return Stats(um.astype("float32"), un.astype("float32"), up.astype("float32"),
                 im.astype("float32"), inn.astype("float32"), ip.astype("float32"))


def make_features(users: np.ndarray, hist: np.ndarray, count: np.ndarray, cand: np.ndarray,
                  score: np.ndarray, stats: Stats, catalog: Catalog, extra: np.ndarray | None = None) -> dict:
    """users [B], hist [B,N] item idx, count [B], cand [B,K] item idx, score [B,K] retrieval cosine.
    extra [B,K,E] optional additional per-candidate features (e.g. the item-KNN score and ranks of a second source)."""
    B, K = cand.shape
    prof = genre_profile(hist, catalog)                               # [B,20]
    item_g = catalog.genres[cand]                                     # [B,K,20]
    cross = item_g * prof[:, None, :]                                 # [B,K,20]
    rank = np.broadcast_to(np.arange(K, dtype=np.float32), (B, K))
    pop_of_hist = catalog.pop_bucket[hist].astype(np.float32)
    hist_pop = (pop_of_hist * (hist > 0)).sum(1) / np.maximum((hist > 0).sum(1), 1)
    scalars = np.stack([
        score,
        1.0 - rank / K,
        np.broadcast_to(np.log1p(count.astype(np.float32))[:, None], (B, K)) / 8.0,
        np.broadcast_to(stats.user_mean[users][:, None], (B, K)) / 5.0,
        np.broadcast_to(stats.user_pos_rate[users][:, None], (B, K)),
        stats.item_mean[cand] / 5.0,
        stats.item_pos_rate[cand],
        np.log1p(stats.item_n[cand]) / 12.0,
        (catalog.pop_bucket[cand].astype(np.float32) - hist_pop[:, None]) / N_POP_BUCKETS,
    ], axis=-1)                                                       # [B,K,9]
    parts = [item_g, cross, scalars] + ([extra] if extra is not None else [])
    dense = np.concatenate(parts, axis=-1).astype(np.float32)
    return {"dense": dense.reshape(B * K, -1),
            "year": catalog.year_bucket[cand].reshape(-1),
            "pop": catalog.pop_bucket[cand].reshape(-1)}


class _Inputs(keras.layers.Layer):
    def __init__(self, emb=8):
        super().__init__()
        self.year = keras.layers.Embedding(N_YEAR_BUCKETS, emb)
        self.pop = keras.layers.Embedding(N_POP_BUCKETS + 1, emb)

    def call(self, f):
        return tf.concat([f["dense"], self.year(f["year"]), self.pop(f["pop"])], axis=-1)


class DCNRanker(keras.Model):
    """Deep & Cross Network v2: stacked cross layers in parallel with an MLP."""

    def __init__(self, cross_layers=3, deep=(128, 64), proj=24):
        super().__init__()
        self.inp = _Inputs()
        self.cross = [tfrs.layers.dcn.Cross(projection_dim=proj, kernel_initializer="glorot_uniform") for _ in range(cross_layers)]
        self.deep = keras.Sequential([keras.layers.Dense(u, activation="relu") for u in deep])
        self.out = keras.layers.Dense(1)

    def call(self, f):
        x = self.inp(f)
        c = x
        for layer in self.cross:
            c = layer(x, c)
        return tf.squeeze(self.out(tf.concat([c, self.deep(x)], -1)), -1)


class WideDeepRanker(keras.Model):
    """Wide & Deep: a linear 'wide' path over the raw and crossed features plus an MLP."""

    def __init__(self, deep=(128, 64)):
        super().__init__()
        self.inp = _Inputs()
        self.wide = keras.layers.Dense(1)
        self.deep = keras.Sequential([keras.layers.Dense(u, activation="relu") for u in deep] + [keras.layers.Dense(1)])

    def call(self, f):
        x = self.inp(f)
        return tf.squeeze(self.wide(f["dense"]) + self.deep(x), -1)
