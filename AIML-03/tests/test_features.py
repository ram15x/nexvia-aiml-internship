import numpy as np
from rec.data import Ratings
from rec.features import UserHistories, Catalog, genre_profile, parse_year, GENRES


def cat(n=4):
    ids = np.arange(0, n + 1, dtype=np.int64) * 10
    g = np.zeros((n + 1, len(GENRES)), np.float32); g[1:, 0] = 1
    return Catalog(ids, {int(m): k for k, m in enumerate(ids) if k}, g, np.zeros(n + 1, np.int32), np.zeros(n + 1, np.int32), np.ones(n + 1, np.int64), {})


def R(u, i, r, t):
    return Ratings(np.array(u, np.int32), np.array(i, np.int32), np.array(r, np.uint8), np.array(t, np.int32))


def test_window_is_time_ordered_left_padded_and_never_sees_the_future():
    c = cat()
    r = R([1, 1, 1, 2], [20, 10, 30, 40], [10, 10, 10, 10], [5, 1, 9, 3])   # user 1 in time order: 10, 20, 30
    h = UserHistories(r, c, max_user=2)
    pos = h.start[1] + np.array([0, 1, 2])
    w = h.window(pos, np.full(3, h.start[1]), n=3)
    assert w.tolist() == [[0, 0, 0], [0, 0, 1], [0, 1, 2]]   # item idx 1,2,3 = movieIds 10,20,30


def test_low_ratings_are_not_history_and_latest_returns_last_n():
    c = cat()
    r = R([1, 1, 1], [10, 20, 30], [10, 2, 10], [1, 2, 3])
    h = UserHistories(r, c, max_user=1)
    assert h.latest(np.array([1]), 3).tolist() == [[0, 1, 3]]


def test_genre_profile_ignores_padding():
    c = cat()
    p = genre_profile(np.array([[0, 1, 2]]), c)
    assert abs(p[0, 0] - 1.0) < 1e-6


def test_parse_year():
    assert parse_year("Toy Story (1995)") == 1995 and parse_year("No year") is None
