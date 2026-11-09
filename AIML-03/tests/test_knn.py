import numpy as np, scipy.sparse as sp
from rec.knn import ItemKNN


def test_scores_sum_neighbour_similarities_and_ignore_padding():
    n = sp.csr_matrix(np.array([[0, 0, 0, 0], [0, 0, .5, .1], [0, .5, 0, .2], [0, .1, .2, 0]], np.float32))
    s = ItemKNN(n).scores(np.array([[0, 1, 2]]))
    assert np.allclose(s[0], [0, .5, .5, .3])
