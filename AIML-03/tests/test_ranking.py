import numpy as np
from rec.features import GENRES, Catalog
from rec.ranking import Stats, make_features, N_DENSE, DCNRanker, WideDeepRanker


def tiny():
    n = 6
    g = np.zeros((n, len(GENRES)), np.float32); g[1:, 0] = 1; g[3, 1] = 1
    cat = Catalog(np.arange(n, dtype=np.int64), {k: k for k in range(1, n)}, g, np.zeros(n, np.int32), np.ones(n, np.int32), np.ones(n, np.int64), {})
    st = Stats(*(np.ones(5, np.float32) for _ in range(3)), *(np.ones(n, np.float32) for _ in range(3)))
    return cat, st


def test_features_shape_and_no_nan():
    cat, st = tiny()
    f = make_features(np.array([1, 2]), np.array([[0, 1, 2], [0, 0, 3]]), np.array([2, 1]),
                      np.array([[3, 4, 5], [1, 2, 5]]), np.random.rand(2, 3).astype("float32"), st, cat)
    assert f["dense"].shape == (6, N_DENSE) and f["year"].shape == (6,)
    assert np.isfinite(f["dense"]).all()


def test_both_rankers_run_and_return_one_logit_per_row():
    cat, st = tiny()
    f = make_features(np.array([1, 2]), np.array([[0, 1, 2], [0, 0, 3]]), np.array([2, 1]),
                      np.array([[3, 4, 5], [1, 2, 5]]), np.random.rand(2, 3).astype("float32"), st, cat)
    for M in (DCNRanker, WideDeepRanker):
        out = M()({k: v for k, v in f.items()})
        assert tuple(out.shape) == (6,)


def test_extra_features_are_appended():
    cat, st = tiny()
    args = (np.array([1, 2]), np.array([[0, 1, 2], [0, 0, 3]]), np.array([2, 1]), np.array([[3, 4, 5], [1, 2, 5]]),
            np.random.rand(2, 3).astype("float32"), st, cat)
    base = make_features(*args)["dense"]
    more = make_features(*args, extra=np.ones((2, 3, 3), np.float32))["dense"]
    assert more.shape[1] == base.shape[1] + 3
