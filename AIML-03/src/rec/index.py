"""FAISS indexing of item embeddings (inner product on L2-normalised vectors = cosine)."""
from __future__ import annotations

import faiss
import numpy as np


def build_flat(vectors: np.ndarray) -> faiss.Index:
    idx = faiss.IndexFlatIP(vectors.shape[1]); idx.add(np.ascontiguousarray(vectors, dtype="float32")); return idx


def build_hnsw(vectors: np.ndarray, m: int = 32, ef_construction: int = 200, ef_search: int = 128) -> faiss.Index:
    idx = faiss.IndexHNSWFlat(vectors.shape[1], m, faiss.METRIC_INNER_PRODUCT)
    idx.hnsw.efConstruction = ef_construction
    idx.add(np.ascontiguousarray(vectors, dtype="float32"))
    idx.hnsw.efSearch = ef_search
    return idx


def _search(index: faiss.Index, q: np.ndarray, want: int):
    hnsw = getattr(index, "hnsw", None)
    base = hnsw.efSearch if hnsw is not None else None
    if hnsw is not None and base < want:        # HNSW returns at most ~efSearch hits; widen it for this call only
        hnsw.efSearch = want
    try:
        scores, rows = index.search(q, min(want, index.ntotal))
    finally:
        if hnsw is not None:
            hnsw.efSearch = base
    return scores[0], rows[0]


def search_unseen(index: faiss.Index, query: np.ndarray, seen: np.ndarray, k: int, overfetch: int = 0, fallback: faiss.Index | None = None):
    """Top-k item rows not in `seen`.

    A user's seen items are mostly their own nearest neighbours and HNSW is approximate, so one search for
    k + len(seen) can leave fewer than k unseen. Retry with a larger over-fetch. For some queries the HNSW graph
    search returns only a small fixed number of hits whatever efSearch is; if k unseen still cannot be found,
    use `fallback` (an exact index) instead. Returns (rows, scores, used_fallback).
    """
    q = np.ascontiguousarray(query[None], dtype="float32")
    extra = overfetch
    for _ in range(3):
        scores, rows = _search(index, q, k + len(seen) + extra)
        keep = (rows >= 0) & ~np.isin(rows, seen)
        if keep.sum() >= k or k + len(seen) + extra >= index.ntotal:
            break
        extra = max(256, extra * 4)
    if keep.sum() < k and fallback is not None:
        scores, rows = _search(fallback, q, k + len(seen))
        keep = (rows >= 0) & ~np.isin(rows, seen)
        return rows[keep][:k], scores[keep][:k], True
    return rows[keep][:k], scores[keep][:k], False
