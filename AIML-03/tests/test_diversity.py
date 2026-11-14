import numpy as np
import pytest

from rec.diversity import because, mmr


def _vecs():
    # items 0 and 1 are near-duplicates, item 2 is different, item 3 is different again
    return np.array([[1.0, 0.0], [0.99, 0.1], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)


def test_lambda_one_keeps_ranker_order():
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    assert list(mmr(scores, _vecs(), 3, 1.0)) == [0, 1, 2]


def test_lower_lambda_pushes_near_duplicate_down():
    scores = np.array([0.9, 0.85, 0.7, 0.1])
    order = list(mmr(scores, _vecs(), 3, 0.5))
    assert order[0] == 0
    assert order[1] == 2          # the different item beats the near-duplicate of the leader


def test_best_item_always_first_and_no_repeats():
    scores = np.random.default_rng(0).random(30)
    vecs = np.random.default_rng(1).normal(size=(30, 8)).astype(np.float32)
    for lam in (0.0, 0.3, 0.7):
        order = mmr(scores, vecs, 10, lam)
        assert order[0] == int(np.argmax(scores))
        assert len(set(order.tolist())) == 10


def test_invalid_lambda_is_rejected():
    with pytest.raises(ValueError):
        mmr(np.ones(3), _vecs()[:3], 2, 1.5)


def test_because_picks_the_closest_liked_item():
    vecs = np.vstack([np.zeros((1, 2), np.float32), _vecs()])      # catalog index 0 is padding
    got = because(vecs, np.array([2, 4]), np.array([1, 3]))        # items 2 and 4 against liked items 1 and 3
    assert got == [1, 3]
    assert because(vecs, np.array([2]), np.array([], dtype=np.int64)) == [0]
