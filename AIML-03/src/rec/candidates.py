"""Two-source candidate generation shared by ranker training and serving (so the two cannot drift apart).

Sources: two-tower (vector search) and item-KNN. The candidate list interleaves the two sources' top lists
(unseen items only) until K unique items. Per-candidate extra features: item-KNN score scaled by the user's
best score, and each source's rank as 1 - rank/RANK_CAP (0 beyond the cap).
"""
from __future__ import annotations

import numpy as np

from .index import search_unseen

RANK_CAP = 500
N_EXTRA = 3


def interleave(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    chosen, seen = [], set()
    for i in range(max(len(a), len(b))):
        for lst in (a, b):
            if i < len(lst) and len(chosen) < k:
                c = int(lst[i])
                if c not in seen:
                    seen.add(c); chosen.append(c)
        if len(chosen) >= k:
            break
    return np.array(chosen, dtype=np.int64)


def build_union(q: np.ndarray, items: np.ndarray, index, flat, knn_row: np.ndarray, seen: np.ndarray, k: int):
    """q [d] user vector; items [n,d] with row 0 = padding; index/flat = FAISS indexes over items[1:];
    knn_row [n] item-KNN scores (column 0 ignored); seen = catalog indices to exclude.
    Returns candidate catalog indices [k], tt cosine [k], extra features [k, N_EXTRA], used_fallback."""
    rows, _, used = search_unseen(index, q, np.asarray(seen, dtype=np.int64) - 1, RANK_CAP, fallback=flat)
    tt_list = rows + 1                                                  # FAISS row i <-> catalog index i + 1
    kn = knn_row.astype(np.float32).copy(); kn[0] = -9
    if len(seen): kn[np.asarray(seen, dtype=np.int64)] = -9
    knn_list = np.argsort(-kn, kind="stable")[:RANK_CAP]
    cand = interleave(tt_list, knn_list, k)
    tt_rank = np.full(len(kn), RANK_CAP, np.int64); tt_rank[tt_list] = np.arange(len(tt_list))
    kn_rank = np.full(len(kn), RANK_CAP, np.int64); kn_rank[knn_list] = np.arange(len(knn_list))
    kmax = max(float(kn.max()), 1e-6)
    extra = np.stack([np.clip(kn[cand], 0, None) / kmax,
                      np.clip(1 - tt_rank[cand] / RANK_CAP, 0, None),
                      np.clip(1 - kn_rank[cand] / RANK_CAP, 0, None)], axis=-1).astype(np.float32)
    return cand, (items[cand] @ q).astype(np.float32), extra, used
