import numpy as np
import pandas as pd
import pytest

from correlations import (
    check_pos_def,
    correct_corr_mat,
    adjust_non_pos_def,
    compare_matrices,
)


def test_check_pos_def_matrix_is_not_pos_def():
    rs = np.random.RandomState(0)
    input_matrix = pd.DataFrame(rs.rand(5, 5))
    _eigvals = np.linalg.eigvals(input_matrix)
    assert _eigvals[_eigvals <= 0].shape[0] > 0

    assert not check_pos_def(input_matrix)


def test_check_pos_def_matrix_is_pos_def():
    rs = np.random.RandomState(0)
    data = pd.DataFrame(rs.rand(1000, 10))
    input_matrix = data.corr()
    _eigvals = np.linalg.eigvals(input_matrix)
    assert _eigvals[_eigvals <= 0].shape[0] == 0

    assert check_pos_def(input_matrix)


def test_correct_corr_mat_no_need_to_correct():
    rs = np.random.RandomState(0)
    data = pd.DataFrame(rs.rand(1000, 10))
    input_matrix = data.corr()
    _eigvals = np.linalg.eigvals(input_matrix)
    assert _eigvals[_eigvals <= 0].shape[0] == 0

    corrected_matrix = correct_corr_mat(input_matrix, 1e-10)
    assert np.array_equal(corrected_matrix, input_matrix.to_numpy())


def test_correct_corr_mat_need_to_correct():
    rs = np.random.RandomState(0)
    data = pd.DataFrame(rs.rand(9, 10))
    input_matrix = data.corr()
    _eigvals = np.linalg.eigvals(input_matrix)
    assert _eigvals[_eigvals <= 0].shape[0] > 0

    corrected_matrix = correct_corr_mat(input_matrix, 1e-10)
    assert not np.array_equal(corrected_matrix, input_matrix.to_numpy())
    assert np.allclose(corrected_matrix, input_matrix.to_numpy())


def test_adjust_non_pos_def_need_to_correct():
    rs = np.random.RandomState(0)
    data = pd.DataFrame(rs.rand(9, 10))
    input_matrix = data.corr()
    _eigvals = np.linalg.eigvals(input_matrix)
    assert _eigvals[_eigvals <= 0].shape[0] > 0

    corrected_matrix = adjust_non_pos_def(input_matrix)
    assert not corrected_matrix.equals(input_matrix)
    assert np.allclose(corrected_matrix, input_matrix.to_numpy())
    assert corrected_matrix.index.equals(input_matrix.index)
    assert corrected_matrix.columns.equals(input_matrix.columns)


def test_compare_matrices_need_to_correct():
    rs = np.random.RandomState(0)
    data = pd.DataFrame(rs.rand(9, 10))
    input_matrix = data.corr()
    _eigvals = np.linalg.eigvals(input_matrix)
    assert _eigvals[_eigvals <= 0].shape[0] > 0
    corrected_matrix = adjust_non_pos_def(input_matrix)

    diff = compare_matrices(input_matrix, corrected_matrix)
    assert diff is not None
    assert diff.min() < 0
    assert diff.max() > 0


def test_compare_matrices_need_to_correct_diff_too_big():
    rs = np.random.RandomState(0)
    data = pd.DataFrame(rs.rand(9, 10))
    input_matrix = data.corr()
    _eigvals = np.linalg.eigvals(input_matrix)
    assert _eigvals[_eigvals <= 0].shape[0] > 0
    corrected_matrix = adjust_non_pos_def(input_matrix)

    with pytest.raises(Exception) as e_info:
        compare_matrices(input_matrix, corrected_matrix, check_max=1e-300)

    assert "Difference is larger than threshold" in str(e_info.value)
