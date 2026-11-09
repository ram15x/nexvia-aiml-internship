import numpy as np
from rec.evaluation import recall_ndcg, build_eval_set
from rec.data import Ratings, split_by_time


def R(u, i, r, t):
    return Ratings(np.array(u, dtype=np.int32), np.array(i, dtype=np.int32), np.array(r, dtype=np.uint8), np.array(t, dtype=np.int32))


def test_perfect_ranking_scores_one():
    rec, ndcg = recall_ndcg(np.array([5, 6, 7]), np.array([5, 6]), k=3)
    assert rec == 1.0 and abs(ndcg - 1.0) < 1e-9


def test_miss_scores_zero_and_order_matters():
    assert recall_ndcg(np.array([1, 2, 3]), np.array([9]), k=3) == (0.0, 0.0)
    early = recall_ndcg(np.array([9, 1, 2]), np.array([9]), k=3)[1]
    late = recall_ndcg(np.array([1, 2, 9]), np.array([9]), k=3)[1]
    assert early > late


def test_eval_set_drops_seen_items_and_low_ratings():
    hist = R([1, 1], [10, 11], [10, 10], [1, 2])
    tgt = R([1, 1, 1, 2], [10, 12, 13, 14], [10, 8, 4, 10], [5, 5, 5, 5])  # 10 seen; 13 rated 2.0
    es = build_eval_set(hist, tgt)
    assert list(es.relevant[1]) == [12] and list(es.relevant[2]) == [14]


def test_time_split_is_ordered_and_disjoint():
    r = R([1, 1, 1], [1, 2, 3], [8, 8, 8], [10, 20, 30])
    tr, va, te = split_by_time(r, train_before=15, test_from=25)
    assert (len(tr), len(va), len(te)) == (1, 1, 1)
