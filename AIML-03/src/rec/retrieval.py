"""Two-tower retrieval model (TensorFlow Recommenders).

The user tower has NO user-id embedding. It reads the user's most recent liked items and their
genre profile, so any user with a little history gets an embedding, including users that did
not exist at training time. Users with no history go to the popularity / content fallbacks.
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs

from .features import N_POP_BUCKETS, N_YEAR_BUCKETS, Catalog

keras = tf.keras


class TwoTower(tfrs.Model):
    def __init__(self, catalog: Catalog, dim: int = 64, hist_len: int = 30, temperature: float = 0.07, l2: float = 0.0):
        super().__init__()
        self.dim, self.hist_len = dim, hist_len
        n = catalog.size
        self.genres = tf.constant(catalog.genres)
        self.year = tf.constant(catalog.year_bucket)
        self.pop = tf.constant(catalog.pop_bucket)
        self.item_emb = keras.layers.Embedding(n, dim, name="item_emb", embeddings_regularizer=(keras.regularizers.l2(l2) if l2 else None))
        self.year_emb = keras.layers.Embedding(N_YEAR_BUCKETS, 8)
        self.pop_emb = keras.layers.Embedding(N_POP_BUCKETS + 1, 8)
        self.item_mlp = keras.Sequential([keras.layers.Dense(128, activation="relu"), keras.layers.Dense(dim)], name="item_mlp")
        self.user_mlp = keras.Sequential([keras.layers.Dense(128, activation="relu"), keras.layers.Dense(dim)], name="user_mlp")
        self.task = tfrs.tasks.Retrieval(temperature=temperature, remove_accidental_hits=True)

    def item_vector(self, idx):
        x = tf.concat([self.item_emb(idx), self.year_emb(tf.gather(self.year, idx)),
                       self.pop_emb(tf.gather(self.pop, idx)), tf.gather(self.genres, idx)], axis=-1)
        return tf.math.l2_normalize(self.item_mlp(x), axis=-1)

    def user_vector(self, hist, count):
        mask = tf.cast(hist > 0, tf.float32)[..., None]
        e = self.item_emb(hist) * mask
        mean = tf.reduce_sum(e, 1) / tf.maximum(tf.reduce_sum(mask, 1), 1.0)
        last = e[:, -1, :]
        g = tf.gather(self.genres, hist) * mask
        gprof = tf.reduce_sum(g, 1) / tf.maximum(tf.reduce_sum(mask, 1), 1.0)
        c = tf.math.log1p(tf.cast(count, tf.float32))[:, None] / 8.0
        return tf.math.l2_normalize(self.user_mlp(tf.concat([mean, last, gprof, c], -1)), axis=-1)

    def compute_loss(self, f, training=False):
        q = self.user_vector(f["hist"], f["count"])
        c = self.item_vector(f["item"])
        return self.task(q, c, candidate_sampling_probability=f["q"], candidate_ids=f["item"], compute_metrics=False)

    def export_arrays(self, n_items: int) -> dict:
        items = self.item_vector(tf.range(n_items)).numpy().astype("float32")
        items[0] = 0.0  # row 0 is the padding slot, never a real item
        return {"item_vectors": items}
