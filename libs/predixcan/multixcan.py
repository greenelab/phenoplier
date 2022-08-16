"""
This file contains functions to run MultiXcan (individual-level), taken
and adapted from https://github.com/hakyimlab/MetaXcan
"""
from patsy import dmatrices
import numpy as np
import pandas as pd
from numpy import dot as _dot
import statsmodels.api as sm

# functions
def _design_matrices(e_, keys):
    formula = "pheno ~ {}".format(" + ".join(keys))
    y, X = dmatrices(formula, data=e_, return_type="dataframe")
    return y, X


def _filter_eigen_values_from_max(s, ratio):
    s_max = np.max(s)
    return [i for i, x in enumerate(s) if x >= s_max * ratio]


def pc_filter(x, cond_num=30):
    return _filter_eigen_values_from_max(x, 1.0 / cond_num)


class Math:
    def standardize(x, unit_var=True):
        mean = np.mean(x)
        # follow R's convention, ddof=1
        scale = np.std(x, ddof=1)
        if scale == 0:
            return None
        x = x - mean
        if unit_var:
            x = x / scale
        return x


def _get_pc_input(e_, model_keys, unit_var=True):
    Xc = []
    _mk = []
    for key in model_keys:
        x = Math.standardize(e_[key], unit_var)
        if x is not None:
            Xc.append(x)
            _mk.append(key)
    return Xc, _mk


def _pca_data(e_, model_keys, unit_var=True):
    if e_.shape[1] == 2:
        return e_, model_keys, model_keys, 1, 1, 1, 1, 1
    # numpy.svd can't handle typical data size in UK Biobank. So we do PCA through the covariance matrix
    # That is: we compute ths SVD of a covariance matrix, and use those coefficients to get the SVD of input data
    # Shamelessly designed from https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca
    # In numpy.cov, each row is a variable and each column an observation. Exactly opposite to standard PCA notation: it is transposed, then.
    Xc_t, original_keys = _get_pc_input(e_, model_keys, unit_var)
    k = np.cov(Xc_t)
    u, s, vt = np.linalg.svd(k)
    # we want to keep only those components with significant variance, to reduce dimensionality
    selected = pc_filter(s)

    variance = s[selected]
    vt_projection = vt[selected]
    Xc_t_ = _dot(vt_projection, Xc_t)
    pca_keys = ["pc{}".format(i) for i in range(0, len(selected))]
    _data = {pca_keys[i]: x for i, x in enumerate(Xc_t_)}
    _data["pheno"] = e_.pheno
    pca_data = pd.DataFrame(_data)

    return (pca_data, pca_keys, selected, u, s, vt)

    # original return:
    # return (
    #     pca_data,
    #     pca_keys,
    #     original_keys,
    #     np.max(s),
    #     np.min(s),
    #     np.min(s[selected]),
    #     vt_projection,
    #     variance,
    # )


def run_multixcan(y, gene_pred_expr, unit_var=True):
    model_keys = gene_pred_expr.columns.tolist()

    e_ = gene_pred_expr.assign(pheno=y)

    e_, model_keys, *_tmp_rest = _pca_data(e_, model_keys, unit_var=unit_var)

    y, X = _design_matrices(e_, model_keys)

    model = sm.OLS(y, X)
    result = model.fit()
    return result, X, y


def get_y_hat(multixcan_model_result):
    return multixcan_model_result.fittedvalues


def get_ssm(multixcan_model_result, y_data):
    y_data = y_data.squeeze()
    assert len(y_data.shape) == 1
    y_hat = get_y_hat(multixcan_model_result)
    return np.power(y_hat - y_data.mean(), 2).sum()
