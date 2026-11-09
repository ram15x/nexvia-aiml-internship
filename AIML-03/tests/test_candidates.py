import numpy as np
from rec.candidates import build_union, interleave
from rec.index import build_flat


def test_interleave_alternates_and_dedupes():
    assert list(interleave(np.array([1, 2, 3]), np.array([2, 4, 5]), 5)) == [1, 2, 4, 3, 5]
    assert list(interleave(np.array([1]), np.array([2, 3, 4]), 3)) == [1, 2, 3]


def test_build_union_excludes_seen_and_has_consistent_features():
    rng = np.random.default_rng(0); n, d = 1200, 8
    items = rng.normal(size=(n, d)).astype("float32"); items /= np.linalg.norm(items, axis=1, keepdims=True); items[0] = 0
    flat = build_flat(items[1:]); q = items[5]; knn = rng.random(n).astype("float32")
    seen = np.array([5, 6, 7])
    cand, cos, extra, used = build_union(q, items, flat, flat, knn, seen, k=40)
    assert len(cand) == 40 and len(set(cand)) == 40 and not set(cand) & set(seen) and 0 not in cand
    assert np.allclose(cos, items[cand] @ q) and extra.shape == (40, 3)
    assert extra.min() >= 0 and extra.max() <= 1.0
    assert not used
