import numpy as np
from rec.index import build_flat, build_hnsw, search_unseen


def _unit(n, d, seed=0):
    x = np.random.default_rng(seed).normal(size=(n, d)).astype("float32")
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_search_excludes_seen_and_matches_exact_order():
    v = _unit(500, 16); flat = build_flat(v)
    q = v[7]
    seen = np.array([7, 3, 11])
    rows, _, _ = search_unseen(flat, q, seen, k=10)
    assert len(rows) == 10 and not set(rows) & set(seen)
    brute = np.argsort(-(v @ q)); brute = [i for i in brute if i not in set(seen)][:10]
    assert list(rows) == brute


def test_hnsw_high_overlap_with_exact():
    v = _unit(2000, 32, 1); flat, h = build_flat(v), build_hnsw(v, ef_search=200)
    hits = 0
    for i in range(50):
        a, _, _ = search_unseen(flat, v[i], np.array([i]), 20); b, _, _ = search_unseen(h, v[i], np.array([i]), 20)
        hits += len(set(a) & set(b))
    assert hits / (50 * 20) > 0.95


def test_hnsw_returns_full_k_for_user_with_large_seen_list():
    v = _unit(3000, 32, 2); h = build_hnsw(v, ef_search=16)
    seen = np.arange(1500)
    rows, _, _ = search_unseen(h, v[2000], seen, k=200)
    assert len(rows) == 200 and not set(rows) & set(seen)


def test_user_whose_seen_items_are_their_nearest_neighbours_still_gets_k():
    v = _unit(3000, 32, 3); h = build_hnsw(v, ef_search=16)
    q = v[10]
    seen = np.argsort(-(v @ q))[:1200]            # seen == the 1200 closest items, the hard case
    rows, _, _ = search_unseen(h, q, seen, k=200)
    assert len(rows) == 200 and not set(rows) & set(seen)


def test_fallback_is_used_when_hnsw_cannot_supply_k(monkeypatch):
    import rec.index as ix
    v = _unit(1000, 16, 4); flat, h = build_flat(v), build_hnsw(v)
    real = ix._search
    def crippled(index, q, want):                  # simulate an HNSW search that only ever returns 50 hits
        s, r = real(index, q, want)
        if index is h: r = r.copy(); r[50:] = -1
        return s, r
    monkeypatch.setattr(ix, "_search", crippled)
    rows, _, used = search_unseen(h, v[0], np.array([0]), k=100, fallback=flat)
    assert used and len(rows) == 100
