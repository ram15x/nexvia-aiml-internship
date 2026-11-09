"""Item-based collaborative filtering as a second candidate source (neighbour matrix built by scripts/itemknn_baseline.py)."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class ItemKNN:
    def __init__(self, neighbours: sp.csr_matrix):
        self.n = neighbours  # [items, items], row i = top neighbours of item i with cosine similarity

    @classmethod
    def load(cls, path) -> "ItemKNN":
        return cls(sp.load_npz(str(path)).tocsr())

    def scores(self, hist: np.ndarray) -> np.ndarray:
        """hist [B,N] catalog indices (0 = padding) -> dense scores [B, n_items]: sum of similarities to the liked items."""
        b, n_items = hist.shape[0], self.n.shape[0]
        r, c = np.nonzero(hist > 0)
        p = sp.csr_matrix((np.ones(len(r), np.float32), (r, hist[r, c])), shape=(b, n_items))
        out = (p @ self.n).toarray()
        out[:, 0] = 0
        return out
