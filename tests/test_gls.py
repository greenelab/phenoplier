"""
This file contains unit tests for the GLSPhenoplier class.
They are not here yet, only some tests description I want to include in the
future. The GLSPhenoplier class was tested using a notebook in the gls_testing
branch, and those will be moved here in the future.

This is reported in this issue: https://github.com/greenelab/phenoplier/issues/40
"""
from scipy import stats

import conf
from gls import GLSPhenoplier


def test_one_sided_pvalue_coef_positive():
    model = GLSPhenoplier(conf.PHENOMEXCAN["SMULTIXCAN_MASHR_ZSCORES_FILE"])
    model.fit_named("LV603", "Astle_et_al_2016_Neutrophil_count")

    df = model.results.df_resid

    # get expected pvalues for the one-sided and two-sided tests
    exp_pval_twosided = stats.t.sf(model.results.tvalues.loc["lv"], df) * 2.0
    exp_pval_onesided = stats.t.sf(model.results.tvalues.loc["lv"], df)

    # get observed two-sided pvalue
    obs_pval_twosided = model.results.pvalues.loc["lv"]

    # check that pvalue is greater than zero and sufficiently small
    assert obs_pval_twosided is not None
    assert obs_pval_twosided > 0.0
    assert obs_pval_twosided < 1e-6
    assert obs_pval_twosided == exp_pval_twosided == exp_pval_onesided * 2.0

    # get observed one-sided pvalue
    obs_pval_onesided = model.results.pvalues_onesided.loc["lv"]

    # check pvalue
    assert obs_pval_onesided is not None
    assert obs_pval_onesided > 0.0
    assert obs_pval_onesided < 1e-6
    assert obs_pval_onesided == exp_pval_onesided == exp_pval_twosided / 2.0


def test_one_sided_pvalue_coef_negative():
    model = GLSPhenoplier(conf.PHENOMEXCAN["SMULTIXCAN_MASHR_ZSCORES_FILE"])
    model.fit_named("LV270", "20459-General_happiness_with_own_health")

    df = model.results.df_resid

    # get observed two-sided pvalue
    obs_pval_twosided = model.results.pvalues.loc["lv"]

    # check that pvalue is greater than zero and sufficiently small
    assert obs_pval_twosided is not None
    assert obs_pval_twosided > 0.0
    assert obs_pval_twosided < 1e-2

    # get expected and observed one-sided pvalue
    exp_pval_onesided = stats.t.sf(model.results.tvalues.loc["lv"], df)
    obs_pval_onesided = model.results.pvalues_onesided.loc["lv"]

    # check that pvalue for one-sided test is large enough, almost close to one,
    # since here the coeficient is negative
    assert obs_pval_onesided is not None
    assert obs_pval_onesided > 0.99
    assert obs_pval_onesided < 1.0
    assert obs_pval_onesided == exp_pval_onesided


def test_gls_no_correlation_structure():
    # check that, if no correlation structure is given, results should match
    # R's nmle::gls function
    pass


def test_gls_artificial_data():
    # check that, with artificial data and correlation, results should match
    # R's nmle::gls function
    pass


def test_gls_real_data_original_correlation():
    # slice gene correlation data and test with LV136
    # should be the same as with R gls function
    pass


def test_gls_real_data_modified_positive_correlation():
    # artificially positively increase correlation between genes COL4A1 and COL4A2
    # results should be less significant
    pass


def test_gls_real_data_modified_negative_correlation():
    # artificially positively increase correlation between genes COL4A1 and COL4A2
    # results should be more significant
    pass
