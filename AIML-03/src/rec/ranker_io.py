"""Create / load rankers and export them as TensorFlow SavedModels (the format Triton's TensorFlow backend serves)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from .ranking import DCNRanker, WideDeepRanker

RANKERS = {"dcn": DCNRanker, "wide_deep": WideDeepRanker}


def make_ranker(name: str, n_dense: int):
    m = RANKERS[name]()
    m({"dense": tf.zeros([1, n_dense]), "year": tf.zeros([1], tf.int32), "pop": tf.zeros([1], tf.int32)})
    return m


def load_ranker(name: str, weights: str | Path, n_dense: int):
    m = make_ranker(name, n_dense)
    m.load_weights(str(weights)).expect_partial()
    return m


def export_saved_model(model, path: str | Path, n_dense: int) -> None:
    class Serve(tf.Module):
        def __init__(self, m):
            super().__init__(); self.m = m

        @tf.function(input_signature=[tf.TensorSpec([None, n_dense], tf.float32, name="dense"),
                                      tf.TensorSpec([None], tf.int32, name="year"),
                                      tf.TensorSpec([None], tf.int32, name="pop")])
        def score(self, dense, year, pop):
            return {"logit": self.m({"dense": dense, "year": year, "pop": pop})}

    s = Serve(model)
    tf.saved_model.save(s, str(path), signatures={"serving_default": s.score})


def predict_saved(path: str | Path, dense: np.ndarray, year: np.ndarray, pop: np.ndarray) -> np.ndarray:
    fn = tf.saved_model.load(str(path)).signatures["serving_default"]
    return fn(dense=tf.constant(dense), year=tf.constant(year), pop=tf.constant(pop))["logit"].numpy()
