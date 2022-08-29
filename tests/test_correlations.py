import numpy as np
import pandas as pd

from correlations import check_pos_def


def test_check_pos_def_matrix_is_not_pos_def():
    rs = np.random.RandomState(0)
    input_matrix = pd.DataFrame(rs.rand(5, 5))
    _eigvals = np.linalg.eigvals(input_matrix)
    assert _eigvals[_eigvals <= 0].shape[0] > 0

    assert not check_pos_def(input_matrix)[0]


def test_check_pos_def_matrix_is_pos_def():
    rs = np.random.RandomState(0)
    data = pd.DataFrame(rs.rand(1000, 10))
    input_matrix = data.corr()
    _eigvals = np.linalg.eigvals(input_matrix)
    assert _eigvals[_eigvals <= 0].shape[0] == 0

    assert check_pos_def(input_matrix)[0]
