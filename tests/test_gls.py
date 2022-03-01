"""
This file contains unit tests for the GLSPhenoplier class.
They are not here yet, only some tests description I want to include in the
future. The GLSPhenoplier class was tested using a notebook in the gls_testing
branch, and those will be moved here in the future.

This is reported in this issue: https://github.com/greenelab/phenoplier/issues/40
"""
import numpy as np
from scipy import stats
import pandas as pd
import pytest

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


def test_fit_with_phenotype_pandas_series():
    model = GLSPhenoplier(conf.PHENOMEXCAN["SMULTIXCAN_MASHR_ZSCORES_FILE"])

    # get number of genes to simulated phenotype
    lv_weights = GLSPhenoplier._get_data(
        model.smultixcan_result_set_filepath, model_type="MASHR"
    )[2]

    np.random.seed(0)
    phenotype_data = pd.Series(
        np.abs(np.random.normal(size=lv_weights.shape[0])),
        index=lv_weights.index.copy(),
        name="Random phenotype",
    )
    model.fit_named("LV270", phenotype_data)

    assert model.phenotype_code == "Random phenotype"

    # get observed two-sided pvalue
    obs_pval_twosided = model.results.pvalues.loc["lv"]

    # check that pvalue is greater than zero and sufficiently small
    assert obs_pval_twosided is not None
    assert obs_pval_twosided > 0.0
    assert obs_pval_twosided < 1.0

    # get observed one-sided pvalue
    obs_pval_onesided = model.results.pvalues_onesided.loc["lv"]

    assert obs_pval_onesided is not None
    assert obs_pval_onesided > 0.0
    assert obs_pval_onesided < 1.0

    assert obs_pval_onesided < obs_pval_twosided


def test_fit_with_phenotype_pandas_series_genes_not_aligned():
    model = GLSPhenoplier(conf.PHENOMEXCAN["SMULTIXCAN_MASHR_ZSCORES_FILE"])

    # get number of genes to simulated phenotype
    lv_weights = GLSPhenoplier._get_data(
        model.smultixcan_result_set_filepath, model_type="MASHR"
    )[2]

    np.random.seed(0)
    phenotype_data = pd.Series(
        np.abs(np.random.normal(size=lv_weights.shape[0])),
        index=lv_weights.index.copy(),
        name="Random phenotype",
    )
    model.fit_named("LV270", phenotype_data)
    assert model.results.df_resid == phenotype_data.shape[0] - 2
    expected_results = model.results

    # run again with genes reordered in the phenotype
    # results should be the same
    model = GLSPhenoplier(conf.PHENOMEXCAN["SMULTIXCAN_MASHR_ZSCORES_FILE"])

    # np.random.shuffle(phenotype_data)
    phenotype_data = phenotype_data.sample(frac=1, random_state=0)
    model.fit_named("LV270", phenotype_data)
    assert model.results.df_resid == phenotype_data.shape[0] - 2

    assert model.phenotype_code == "Random phenotype"

    pd.testing.assert_series_equal(expected_results.pvalues, model.results.pvalues)
    pd.testing.assert_series_equal(
        expected_results.pvalues_onesided, model.results.pvalues_onesided
    )


def test_fit_with_phenotype_pandas_series_less_genes():
    model = GLSPhenoplier(conf.PHENOMEXCAN["SMULTIXCAN_MASHR_ZSCORES_FILE"])

    # get number of genes to simulated phenotype
    lv_weights = GLSPhenoplier._get_data(
        model.smultixcan_result_set_filepath, model_type="MASHR"
    )[2]

    np.random.seed(0)
    n_genes = 3000
    phenotype_data = pd.Series(
        np.abs(np.random.normal(size=n_genes)),
        index=lv_weights.sample(n=n_genes, random_state=0).index.copy(),
        name="Random phenotype",
    )
    model.fit_named("LV270", phenotype_data)
    assert model.results.df_resid == phenotype_data.shape[0] - 2
    expected_results = model.results

    # run again with genes reordered in the phenotype
    # results should be the same
    model = GLSPhenoplier(conf.PHENOMEXCAN["SMULTIXCAN_MASHR_ZSCORES_FILE"])

    # np.random.shuffle(phenotype_data)
    phenotype_data = phenotype_data.sample(frac=1, random_state=0)
    model.fit_named("LV270", phenotype_data)
    assert model.results.df_resid == phenotype_data.shape[0] - 2

    assert model.phenotype_code == "Random phenotype"

    pd.testing.assert_series_equal(expected_results.pvalues, model.results.pvalues)
    pd.testing.assert_series_equal(
        expected_results.pvalues_onesided, model.results.pvalues_onesided
    )


def test_fit_with_phenotype_pandas_series_more_genes():
    model = GLSPhenoplier(conf.PHENOMEXCAN["SMULTIXCAN_MASHR_ZSCORES_FILE"])

    # get number of genes to simulated phenotype
    lv_weights = GLSPhenoplier._get_data(
        model.smultixcan_result_set_filepath, model_type="MASHR"
    )[2]

    np.random.seed(0)
    # add a gene (I made up a name: AGENE) that does not exist in LV models
    phenotype_data = pd.Series(
        np.abs(np.random.normal(size=lv_weights.shape[0] + 1)),
        index=lv_weights.index.tolist() + ["AGENE"],
        name="Random phenotype",
    )
    with pytest.warns(UserWarning):
        model.fit_named("LV270", phenotype_data)
    assert model.results.df_resid == phenotype_data.shape[0] - 2 - 1  # AGENE
    expected_results = model.results

    # run again with genes reordered in the phenotype
    # results should be the same
    model = GLSPhenoplier(conf.PHENOMEXCAN["SMULTIXCAN_MASHR_ZSCORES_FILE"])

    # np.random.shuffle(phenotype_data)
    phenotype_data = phenotype_data.sample(frac=1, random_state=0)
    with pytest.warns(UserWarning):
        model.fit_named("LV270", phenotype_data)
    assert model.results.df_resid == phenotype_data.shape[0] - 2 - 1  # AGENE

    assert model.phenotype_code == "Random phenotype"

    pd.testing.assert_series_equal(expected_results.pvalues, model.results.pvalues)
    pd.testing.assert_series_equal(
        expected_results.pvalues_onesided, model.results.pvalues_onesided
    )


def test_gls_different_prediction_models_get_data_gene_corr():
    # model 1 (mashr)
    model1_gene_corrs = GLSPhenoplier._get_data(
        conf.PHENOMEXCAN["SMULTIXCAN_MASHR_ZSCORES_FILE"], model_type="MASHR"
    )[0]

    # model 1 (elastic net)
    model2_gene_corrs = GLSPhenoplier._get_data(
        conf.PHENOMEXCAN["SMULTIXCAN_MASHR_ZSCORES_FILE"], model_type="ELASTIC_NET"
    )[0]

    assert model1_gene_corrs.shape == model2_gene_corrs.shape
    assert not np.allclose(
        model1_gene_corrs.to_numpy(),
        model2_gene_corrs.to_numpy(),
    )


def test_gls_different_prediction_models_gls_fit_named():
    model = GLSPhenoplier(
        conf.PHENOMEXCAN["SMULTIXCAN_MASHR_ZSCORES_FILE"],
        model_type="MASHR",
    )

    # get number of genes to simulated phenotype
    lv_weights = GLSPhenoplier._get_data(
        model.smultixcan_result_set_filepath, model_type="MASHR"
    )[2]

    np.random.seed(0)
    # add a gene (I made up a name: AGENE) that does not exist in LV models
    phenotype_data = pd.Series(
        np.abs(np.random.normal(size=lv_weights.shape[0])),
        index=lv_weights.index.tolist(),
        name="Random phenotype",
    )

    # fit with mashr
    model.fit_named("LV270", phenotype_data)
    assert model.results.df_resid == phenotype_data.shape[0] - 2
    model1_results = model.results

    # fit with elastic net
    model = GLSPhenoplier(
        conf.PHENOMEXCAN["SMULTIXCAN_MASHR_ZSCORES_FILE"],
        model_type="ELASTIC_NET",
    )

    model.fit_named("LV270", phenotype_data)
    assert model.results.df_resid == phenotype_data.shape[0] - 2
    model2_results = model.results

    assert model.phenotype_code == "Random phenotype"

    assert not np.allclose(
        model1_results.pvalues.to_numpy(),
        model2_results.pvalues.to_numpy(),
    )

    assert not np.allclose(
        model1_results.pvalues_onesided.to_numpy(),
        model2_results.pvalues_onesided.to_numpy(),
    )


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
