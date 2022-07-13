"""
This file contains unit tests for the GLSPhenoplier class.
They are not here yet, only some tests description I want to include in the
future. The GLSPhenoplier class was tested using a notebook in the gls_testing
branch, and those will be moved here in the future.

This is reported in this issue: https://github.com/greenelab/phenoplier/issues/40
"""
from pathlib import Path

import numpy as np
from scipy import stats
import pandas as pd
import pytest

import conf
from gls import GLSPhenoplier


DATA_DIR = (Path(__file__).parent / "data" / "gls").resolve()
assert DATA_DIR.exists()


def test_gls_random_phenotype0_lv1():
    phenotype_code = 0
    lv_code = "LV1"

    y = pd.read_pickle(
        DATA_DIR / f"multixcan-random_phenotype{phenotype_code}-pvalues.pkl.xz"
    )

    model = GLSPhenoplier(
        use_own_implementation=True,
        gene_corrs_file_path=DATA_DIR / "corr_mat_fixed.pkl.xz",
    )
    model.fit_named(lv_code, y)

    obs_coef = model.results.params.loc["lv"]
    obs_coef_se = model.results.bse.loc["lv"]
    obs_tvalue = model.results.tvalues.loc["lv"]
    obs_pval_twosided = model.results.pvalues.loc["lv"]
    obs_pval_onesided = model.results.pvalues_onesided.loc["lv"]

    exp_coef = -0.00047704516235583206
    exp_coef_se = 0.0036681089173670593
    exp_tvalue = -0.13005207127225968
    exp_pval_twosided = 0.8965292913179314
    exp_res_df = 6440
    exp_pval_onesided = stats.t.sf(exp_tvalue, exp_res_df)

    # check
    assert obs_coef is not None
    assert isinstance(obs_coef, float)
    assert obs_coef == pytest.approx(exp_coef, rel=1e-10)
    assert obs_coef_se == pytest.approx(exp_coef_se, rel=1e-2)
    assert obs_tvalue == pytest.approx(exp_tvalue, rel=1e-2)
    assert obs_pval_twosided == pytest.approx(exp_pval_twosided, rel=1e-2)
    assert obs_pval_onesided == pytest.approx(exp_pval_onesided, rel=1e-3)


def test_gls_random_phenotype10_lv10():
    phenotype_code = 10
    lv_code = "LV10"

    y = pd.read_pickle(
        DATA_DIR / f"multixcan-random_phenotype{phenotype_code}-pvalues.pkl.xz"
    )

    model = GLSPhenoplier(
        use_own_implementation=True,
        gene_corrs_file_path=DATA_DIR / "corr_mat_fixed.pkl.xz",
    )
    model.fit_named(lv_code, y)

    obs_coef = model.results.params.loc["lv"]
    obs_coef_se = model.results.bse.loc["lv"]
    obs_tvalue = model.results.tvalues.loc["lv"]
    obs_pval_twosided = model.results.pvalues.loc["lv"]
    obs_pval_onesided = model.results.pvalues_onesided.loc["lv"]

    exp_coef = -0.009080664541144956
    exp_coef_se = 0.007642176496737248
    exp_tvalue = -1.1882301521067795
    exp_pval_twosided = 0.2347865648578187
    exp_res_df = 6440
    exp_pval_onesided = stats.t.sf(exp_tvalue, exp_res_df)

    # check
    assert obs_coef is not None
    assert isinstance(obs_coef, float)
    assert obs_coef == pytest.approx(exp_coef, rel=1e-10)
    assert obs_coef_se == pytest.approx(exp_coef_se, rel=1e-2)
    assert obs_tvalue == pytest.approx(exp_tvalue, rel=1e-2)
    assert obs_pval_twosided == pytest.approx(exp_pval_twosided, rel=5e-2)
    assert obs_pval_onesided == pytest.approx(exp_pval_onesided, rel=1e-2)


def test_gls_real_pheno_whooping_cough_lv100():
    phenotype_code = "whooping_cough"
    lv_code = "LV100"

    y = pd.read_pickle(
        DATA_DIR / f"multixcan-phenomexcan-{phenotype_code}-pvalues.pkl.xz"
    )

    model = GLSPhenoplier(
        use_own_implementation=True,
        gene_corrs_file_path=DATA_DIR / "corr_mat_fixed.pkl.xz",
    )
    model.fit_named(lv_code, y)

    obs_coef = model.results.params.loc["lv"]
    obs_coef_se = model.results.bse.loc["lv"]
    obs_tvalue = model.results.tvalues.loc["lv"]
    obs_pval_twosided = model.results.pvalues.loc["lv"]
    obs_pval_onesided = model.results.pvalues_onesided.loc["lv"]

    exp_coef = -0.01924107338805289
    exp_coef_se = 0.00417985154437976
    exp_tvalue = -4.603291093896502
    exp_pval_twosided = 4.2383656497944205e-06
    exp_res_df = 6448
    exp_pval_onesided = stats.t.sf(exp_tvalue, exp_res_df)

    # check
    assert obs_coef is not None
    assert isinstance(obs_coef, float)
    assert obs_coef == pytest.approx(exp_coef, rel=1e-10)
    assert obs_coef_se == pytest.approx(exp_coef_se, rel=5e-2)
    assert obs_tvalue == pytest.approx(exp_tvalue, rel=5e-2)
    assert obs_pval_twosided == pytest.approx(exp_pval_twosided, abs=1e-5, rel=1e-2)
    assert obs_pval_onesided == pytest.approx(exp_pval_onesided, rel=1e-5)


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
    lv_weights = GLSPhenoplier._get_lv_weights()

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

    # get data and simulate phenotype
    lv_weights = GLSPhenoplier._get_lv_weights()
    gene_corrs = GLSPhenoplier._get_gene_corrs(model.gene_corrs_file_path)

    np.random.seed(0)
    phenotype_data = pd.Series(
        np.abs(np.random.normal(size=lv_weights.shape[0])),
        index=lv_weights.index.copy(),
        name="Random phenotype",
    )

    # match all genes and align
    phenotype_data, lv_weights = GLSPhenoplier.match_and_align_genes(
        phenotype_data, lv_weights, gene_corrs
    )[0:2]

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

    # get data and simulate phenotype
    lv_weights = GLSPhenoplier._get_lv_weights()
    gene_corrs = GLSPhenoplier._get_gene_corrs(model.gene_corrs_file_path)

    np.random.seed(0)
    n_genes = 3000
    phenotype_data = pd.Series(
        np.abs(np.random.normal(size=n_genes)),
        index=lv_weights.sample(n=n_genes, random_state=0).index.copy(),
        name="Random phenotype",
    )

    # match all genes and align
    phenotype_data_aligned, lv_weights = GLSPhenoplier.match_and_align_genes(
        phenotype_data, lv_weights, gene_corrs
    )[0:2]

    model.fit_named("LV270", phenotype_data)
    assert model.results.df_resid == phenotype_data_aligned.shape[0] - 2
    expected_results = model.results

    # run again with genes reordered in the phenotype
    # results should be the same
    model = GLSPhenoplier(conf.PHENOMEXCAN["SMULTIXCAN_MASHR_ZSCORES_FILE"])

    # np.random.shuffle(phenotype_data)
    phenotype_data = phenotype_data.sample(frac=1, random_state=0)
    model.fit_named("LV270", phenotype_data)
    assert model.results.df_resid == phenotype_data_aligned.shape[0] - 2

    assert model.phenotype_code == "Random phenotype"

    pd.testing.assert_series_equal(expected_results.pvalues, model.results.pvalues)
    pd.testing.assert_series_equal(
        expected_results.pvalues_onesided, model.results.pvalues_onesided
    )


def test_fit_with_phenotype_pandas_series_more_genes():
    model = GLSPhenoplier(conf.PHENOMEXCAN["SMULTIXCAN_MASHR_ZSCORES_FILE"])

    # get data and simulate phenotype
    lv_weights = GLSPhenoplier._get_lv_weights()
    gene_corrs = GLSPhenoplier._get_gene_corrs(model.gene_corrs_file_path)

    np.random.seed(0)
    # add a gene (I made up a name: AGENE) that does not exist in LV models
    phenotype_data = pd.Series(
        np.abs(np.random.normal(size=lv_weights.shape[0] + 1)),
        index=lv_weights.index.tolist() + ["AGENE"],
        name="Random phenotype",
    )

    # match all genes and align
    phenotype_data_aligned, lv_weights = GLSPhenoplier.match_and_align_genes(
        phenotype_data, lv_weights, gene_corrs
    )[0:2]

    with pytest.warns(UserWarning):
        model.fit_named("LV270", phenotype_data)
    assert model.results.df_resid == phenotype_data_aligned.shape[0] - 2
    expected_results = model.results

    # run again with genes reordered in the phenotype
    # results should be the same
    model = GLSPhenoplier(conf.PHENOMEXCAN["SMULTIXCAN_MASHR_ZSCORES_FILE"])

    # np.random.shuffle(phenotype_data)
    phenotype_data = phenotype_data.sample(frac=1, random_state=0)
    with pytest.warns(UserWarning):
        model.fit_named("LV270", phenotype_data)
    assert model.results.df_resid == phenotype_data_aligned.shape[0] - 2

    assert model.phenotype_code == "Random phenotype"

    pd.testing.assert_series_equal(expected_results.pvalues, model.results.pvalues)
    pd.testing.assert_series_equal(
        expected_results.pvalues_onesided, model.results.pvalues_onesided
    )


def test_fit_with_phenotype_pandas_series_with_nan_extra_genes_not_in_lv_models():
    model = GLSPhenoplier(conf.PHENOMEXCAN["SMULTIXCAN_MASHR_ZSCORES_FILE"])

    # original run
    # get data and simulate phenotype
    lv_weights = GLSPhenoplier._get_lv_weights()
    gene_corrs = GLSPhenoplier._get_gene_corrs(model.gene_corrs_file_path)

    np.random.seed(0)
    phenotype_data = pd.Series(
        np.abs(np.random.normal(size=lv_weights.shape[0])),
        index=lv_weights.index.copy(),
        name="Random phenotype",
    )

    # match all genes and align
    phenotype_data_aligned, lv_weights = GLSPhenoplier.match_and_align_genes(
        phenotype_data, lv_weights, gene_corrs
    )[0:2]

    model.fit_named("LV270", phenotype_data)
    assert model.results.df_resid == phenotype_data_aligned.shape[0] - 2
    expected_results = model.results

    # run again with extra genes with missing data
    # results should be the same
    model = GLSPhenoplier(conf.PHENOMEXCAN["SMULTIXCAN_MASHR_ZSCORES_FILE"])

    # generate same phenotype plus three genes with missing data
    phenotype_data = phenotype_data.append(
        pd.Series(
            np.full(3, np.nan),
            index=["SIMGENE1", "SIMGENE2", "SIMGENE3"],
            name="Random phenotype",
        )
    )
    assert phenotype_data.shape[0] > 5
    assert phenotype_data.isna().sum() == 3

    model.fit_named("LV270", phenotype_data)
    assert model.results.df_resid == phenotype_data_aligned.shape[0] - 2

    assert model.phenotype_code == "Random phenotype"

    pd.testing.assert_series_equal(expected_results.pvalues, model.results.pvalues)
    pd.testing.assert_series_equal(
        expected_results.pvalues_onesided, model.results.pvalues_onesided
    )


def test_fit_with_phenotype_pandas_series_with_nan():
    model = GLSPhenoplier(conf.PHENOMEXCAN["SMULTIXCAN_MASHR_ZSCORES_FILE"])

    # original run
    # get data and simulate phenotype
    lv_weights = GLSPhenoplier._get_lv_weights()
    gene_corrs = GLSPhenoplier._get_gene_corrs(model.gene_corrs_file_path)

    np.random.seed(0)
    phenotype_data = pd.Series(
        np.abs(np.random.normal(size=lv_weights.shape[0])),
        index=lv_weights.index.copy(),
        name="Random phenotype",
    )

    # match all genes and align
    phenotype_data_aligned, lv_weights = GLSPhenoplier.match_and_align_genes(
        phenotype_data, lv_weights, gene_corrs
    )[0:2]

    model.fit_named("LV270", phenotype_data)
    assert model.results.df_resid == phenotype_data_aligned.shape[0] - 2
    expected_results = model.results

    # run again with _existing_ genes with missing data
    # results should be the same
    model = GLSPhenoplier(conf.PHENOMEXCAN["SMULTIXCAN_MASHR_ZSCORES_FILE"])

    # generate same phenotype plus three genes with missing data
    phenotype_data.iloc[10] = np.nan
    phenotype_data.iloc[100] = np.nan
    phenotype_data.iloc[500] = np.nan
    assert phenotype_data.isna().sum() == 3

    model.fit_named("LV270", phenotype_data)
    assert (
        model.results.df_resid == phenotype_data_aligned.shape[0] - 2 - 3
    )  # remove nan genes

    assert model.phenotype_code == "Random phenotype"

    assert not np.allclose(
        expected_results.pvalues.to_numpy(),
        model.results.pvalues.to_numpy(),
    )

    assert not np.allclose(
        expected_results.pvalues_onesided.to_numpy(),
        model.results.pvalues_onesided.to_numpy(),
    )


def test_gls_different_prediction_models_gls_fit_named():
    model = GLSPhenoplier(
        conf.PHENOMEXCAN["SMULTIXCAN_MASHR_ZSCORES_FILE"],
        gene_corrs_file_path=DATA_DIR / "sample-gene_corrs-gtex_v8-mashr.pkl",
    )

    # get data and simulate phenotype
    lv_weights = GLSPhenoplier._get_lv_weights()
    gene_corrs = GLSPhenoplier._get_gene_corrs(model.gene_corrs_file_path)

    np.random.seed(0)
    phenotype_data = pd.Series(
        np.abs(np.random.normal(size=lv_weights.shape[0])),
        index=lv_weights.index.tolist(),
        name="Random phenotype",
    )

    # match all genes and align
    phenotype_data, lv_weights = GLSPhenoplier.match_and_align_genes(
        phenotype_data, lv_weights, gene_corrs
    )[0:2]

    # fit with mashr
    model.fit_named("LV270", phenotype_data)
    assert model.results.df_resid == phenotype_data.shape[0] - 2
    model1_results = model.results

    # fit with elastic net
    model = GLSPhenoplier(
        conf.PHENOMEXCAN["SMULTIXCAN_MASHR_ZSCORES_FILE"],
        gene_corrs_file_path=DATA_DIR / "sample-gene_corrs-1000g-en.pkl",
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


def test_gls_debug_use_ols():
    model = GLSPhenoplier(
        conf.PHENOMEXCAN["SMULTIXCAN_MASHR_ZSCORES_FILE"],
        gene_corrs_file_path=DATA_DIR / "sample-gene_corrs-gtex_v8-mashr.pkl",
    )

    # get data and simulate phenotype
    lv_weights = GLSPhenoplier._get_lv_weights()
    gene_corrs = GLSPhenoplier._get_gene_corrs(model.gene_corrs_file_path)

    np.random.seed(0)
    phenotype_data = pd.Series(
        np.abs(np.random.normal(size=lv_weights.shape[0])),
        index=lv_weights.index.tolist(),
        name="Random phenotype",
    )

    # match all genes and align
    phenotype_data, lv_weights = GLSPhenoplier.match_and_align_genes(
        phenotype_data, lv_weights, gene_corrs
    )[0:2]

    # fit with using GLS model
    model.fit_named("LV270", phenotype_data)
    assert model.results.df_resid == phenotype_data.shape[0] - 2
    model1_results = model.results

    # fit with OLS
    model = GLSPhenoplier(
        conf.PHENOMEXCAN["SMULTIXCAN_MASHR_ZSCORES_FILE"],
        debug_use_ols=True,
        # gene_corrs_file_path=DATA_DIR / "sample-gene_corrs-1000g-en.pkl",
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
