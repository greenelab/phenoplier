"""
It contains functions to support the creation of gene correlation matrices.
"""

import numpy as np
import pandas as pd
from statsmodels.stats.correlation_tools import corr_nearest
from IPython.display import display


def check_pos_def(matrix: pd.DataFrame):
    """
    Checks that a correlation matrix is positive definite.
    """
    # show nonpositive eigenvalues
    eigs = np.linalg.eigvals(matrix.to_numpy())
    neg_eigs = eigs[eigs <= 0]
    display(f"Number of negative eigenvalues: {len(neg_eigs)}")
    display(f"Negative eigenvalues:\n{neg_eigs}")

    # check what statsmodels.GLS expects
    try:
        # decomposition used by statsmodels.GLS
        np.linalg.cholesky(np.linalg.inv(matrix.to_numpy())).T
        print("Works! (statsmodels.GLS)")
    except Exception as e:
        print(f"Cholesky decomposition failed (statsmodels.GLS): {str(e)}")

    # check
    CHOL_DECOMPOSITION_WORKED = None

    try:
        np.linalg.inv(np.linalg.cholesky(matrix.to_numpy()))
        print("Works!")
        CHOL_DECOMPOSITION_WORKED = True
    except Exception as e:
        print(f"Cholesky decomposition failed: {str(e)}")
        CHOL_DECOMPOSITION_WORKED = False

    return CHOL_DECOMPOSITION_WORKED


def correct_corr_mat(corr_mat: pd.DataFrame, threshold):
    """
    If necessary, it fixes a correlation matrix using its eigenvalues. The approach uses this function:

        https://www.statsmodels.org/dev/generated/statsmodels.stats.correlation_tools.corr_nearest.html

    However, it could be slow in some cases. An alternative implementation is commented out below (read
    details below).

    It always returns a numpy array.
    """

    if check_pos_def(corr_mat):
        return corr_mat.to_numpy()

    return corr_nearest(corr_mat, threshold=threshold, n_fact=100)

    # commented out below there is a manual method that is faster and computes the
    # eigenvalues only once; it should be equivalent to the function corr_clipped from statsmodels.
    # Compared to corr_neareast, the difference with the original correlation matrix is larger with
    # the implementation below
    #
    # eigvals, eigvects = np.linalg.eigh(corr_mat)
    # eigvals = np.maximum(eigvals, threshold)
    # corr_mat_fixed = eigvects @ np.diag(eigvals) @ eigvects.T
    # return corr_mat_fixed


def adjust_non_pos_def(matrix, threshold=1e-15):
    """
    It is the same as correct_corr_mat, but it returns a dataframe with the same
    row and columns as the original matrix.
    """
    matrix_fixed = correct_corr_mat(matrix, threshold)

    return pd.DataFrame(
        matrix_fixed,
        index=matrix.index.copy(),
        columns=matrix.columns.copy(),
    )


def compare_matrices(matrix1, matrix2, check_max=1e-10):
    """
    Compares two matrices of the same dimension and returns the differences.
    It is used to compare how different is the original correlation matrix from
    the corrected one.
    """
    _diff = (matrix1 - matrix2).unstack()
    display(_diff.describe())
    display(_diff.sort_values())

    if check_max is not None:
        assert _diff.abs().max() < check_max, "Difference is larger than threshold"

    return _diff
