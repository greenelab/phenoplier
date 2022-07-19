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


def test_gls_coef_positive_full_matrix_random_phenotype():
    phenotype_code = 6
    lv_code = "LV45"

    y = pd.read_pickle(
        DATA_DIR / f"multixcan-random_phenotype{phenotype_code}-pvalues.pkl.xz"
    )

    model = GLSPhenoplier(
        use_own_implementation=True,
        gene_corrs_file_path=DATA_DIR / "corr_mat.pkl.xz",
        debug_use_sub_gene_corr=False,
    )
    model.fit_named(lv_code, y)

    obs_coef = model.results.params.loc["lv"]
    obs_coef_se = model.results.bse.loc["lv"]
    obs_tvalue = model.results.tvalues.loc["lv"]
    obs_pval_twosided = model.results.pvalues.loc["lv"]
    obs_pval_onesided = model.results.pvalues_onesided.loc["lv"]

    exp_coef = 0.003810795472142111
    exp_coef_se = 0.012271545160174927
    exp_tvalue = 0.310539171913685
    exp_pval_twosided = 0.7561610253800751
    exp_pval_onesided = 0.37808051269003756

    # check
    assert obs_coef is not None
    assert isinstance(obs_coef, float)
    assert obs_coef == pytest.approx(exp_coef, rel=1e-5)
    assert obs_coef_se == pytest.approx(exp_coef_se, rel=1e-5)
    assert obs_tvalue == pytest.approx(exp_tvalue, rel=1e-5)
    assert obs_pval_twosided == pytest.approx(exp_pval_twosided, rel=1e-5)
    assert obs_pval_onesided == pytest.approx(exp_pval_onesided, rel=1e-5)


def test_gls_coef_negative_full_matrix_random_phenotype():
    phenotype_code = 0
    lv_code = "LV800"

    y = pd.read_pickle(
        DATA_DIR / f"multixcan-random_phenotype{phenotype_code}-pvalues.pkl.xz"
    )

    model = GLSPhenoplier(
        use_own_implementation=True,
        gene_corrs_file_path=DATA_DIR / "corr_mat.pkl.xz",
        debug_use_sub_gene_corr=False,
    )
    model.fit_named(lv_code, y)

    obs_coef = model.results.params.loc["lv"]
    obs_coef_se = model.results.bse.loc["lv"]
    obs_tvalue = model.results.tvalues.loc["lv"]
    obs_pval_twosided = model.results.pvalues.loc["lv"]
    obs_pval_onesided = model.results.pvalues_onesided.loc["lv"]

    exp_coef = -0.0008674165392909389
    exp_coef_se = 0.0003192417956200798
    exp_tvalue = -2.717114585845851
    exp_pval_twosided = 0.0066029845360324755
    exp_pval_onesided = 0.9966985077319838

    # check
    assert obs_coef is not None
    assert isinstance(obs_coef, float)
    assert obs_coef == pytest.approx(exp_coef, rel=1e-5)
    assert obs_coef_se == pytest.approx(exp_coef_se, rel=1e-5)
    assert obs_tvalue == pytest.approx(exp_tvalue, rel=1e-5)
    assert obs_pval_twosided == pytest.approx(exp_pval_twosided, rel=1e-4)
    assert obs_pval_onesided == pytest.approx(exp_pval_onesided, rel=1e-5)


def test_gls_full_matrix_same_model_different_lvs():
    # run on same phenotype, but different lvs, using the same model
    # this mimics the use of GLSPhenoplier by gls_cli.py (console)
    phenotype_code = 6

    y = pd.read_pickle(
        DATA_DIR / f"multixcan-random_phenotype{phenotype_code}-pvalues.pkl.xz"
    )

    model = GLSPhenoplier(
        use_own_implementation=True,
        gene_corrs_file_path=DATA_DIR / "corr_mat.pkl.xz",
        debug_use_sub_gene_corr=False,
    )

    # first LV
    lv_code = "LV45"
    model.fit_named(lv_code, y)

    obs_coef = model.results.params.loc["lv"]
    obs_coef_se = model.results.bse.loc["lv"]
    obs_tvalue = model.results.tvalues.loc["lv"]
    obs_pval_twosided = model.results.pvalues.loc["lv"]
    obs_pval_onesided = model.results.pvalues_onesided.loc["lv"]

    exp_coef = 0.003810795472142111
    exp_coef_se = 0.012271545160174927
    exp_tvalue = 0.310539171913685
    exp_pval_twosided = 0.7561610253800751
    exp_pval_onesided = 0.37808051269003756

    # check
    assert obs_coef is not None
    assert isinstance(obs_coef, float)
    assert obs_coef == pytest.approx(exp_coef, rel=1e-5)
    assert obs_coef_se == pytest.approx(exp_coef_se, rel=1e-5)
    assert obs_tvalue == pytest.approx(exp_tvalue, rel=1e-5)
    assert obs_pval_twosided == pytest.approx(exp_pval_twosided, rel=1e-5)
    assert obs_pval_onesided == pytest.approx(exp_pval_onesided, rel=1e-5)

    # second LV
    lv_code = "LV455"
    model.fit_named(lv_code, y)

    obs_coef = model.results.params.loc["lv"]
    obs_coef_se = model.results.bse.loc["lv"]
    obs_tvalue = model.results.tvalues.loc["lv"]
    obs_pval_twosided = model.results.pvalues.loc["lv"]
    obs_pval_onesided = model.results.pvalues_onesided.loc["lv"]

    exp_coef = 8.650711873537164e-05
    exp_coef_se = 0.00018035107257825515
    exp_tvalue = 0.47965957451035285
    exp_pval_twosided = 0.6314857662460573
    exp_pval_onesided = 0.3157428831230287

    # check
    assert obs_coef is not None
    assert isinstance(obs_coef, float)
    assert obs_coef == pytest.approx(exp_coef, rel=1e-5)
    assert obs_coef_se == pytest.approx(exp_coef_se, rel=1e-5)
    assert obs_tvalue == pytest.approx(exp_tvalue, rel=1e-5)
    assert obs_pval_twosided == pytest.approx(exp_pval_twosided, rel=1e-5)
    assert obs_pval_onesided == pytest.approx(exp_pval_onesided, rel=1e-5)


def test_gls_coef_negative_sub_matrix_random_phenotype():
    phenotype_code = 6
    lv_code = "LV45"

    y = pd.read_pickle(
        DATA_DIR / f"multixcan-random_phenotype{phenotype_code}-pvalues.pkl.xz"
    )

    model = GLSPhenoplier(
        use_own_implementation=True,
        gene_corrs_file_path=DATA_DIR / "corr_mat.pkl.xz",
        debug_use_sub_gene_corr=True,
    )
    model.fit_named(lv_code, y)

    obs_coef = model.results.params.loc["lv"]
    obs_coef_se = model.results.bse.loc["lv"]
    obs_tvalue = model.results.tvalues.loc["lv"]
    obs_pval_twosided = model.results.pvalues.loc["lv"]
    obs_pval_onesided = model.results.pvalues_onesided.loc["lv"]

    exp_coef = -0.012985100862501646
    exp_coef_se = 0.011620815913625014
    exp_tvalue = -1.117400099873973
    exp_pval_twosided = 0.2638649762970155
    exp_pval_onesided = 0.8680675118514922

    # check
    assert obs_coef is not None
    assert isinstance(obs_coef, float)
    assert obs_coef == pytest.approx(exp_coef, rel=1e-10)
    assert obs_coef_se == pytest.approx(exp_coef_se, rel=1e-10)
    assert obs_tvalue == pytest.approx(exp_tvalue, rel=1e-10)
    assert obs_pval_twosided == pytest.approx(exp_pval_twosided, rel=1e-10)
    assert obs_pval_onesided == pytest.approx(exp_pval_onesided, rel=1e-10)


def test_gls_coef_positive_and_very_small_sub_matrix_random_phenotype():
    phenotype_code = 10
    lv_code = "LV100"

    y = pd.read_pickle(
        DATA_DIR / f"multixcan-random_phenotype{phenotype_code}-pvalues.pkl.xz"
    )

    model = GLSPhenoplier(
        use_own_implementation=True,
        gene_corrs_file_path=DATA_DIR / "corr_mat.pkl.xz",
        debug_use_sub_gene_corr=True,
    )
    model.fit_named(lv_code, y)

    obs_coef = model.results.params.loc["lv"]
    obs_coef_se = model.results.bse.loc["lv"]
    obs_tvalue = model.results.tvalues.loc["lv"]
    obs_pval_twosided = model.results.pvalues.loc["lv"]
    obs_pval_onesided = model.results.pvalues_onesided.loc["lv"]

    exp_coef = 6.008877267380782e-08
    exp_coef_se = 2.7345042158271415e-05
    exp_tvalue = 0.0021974284159453007
    exp_pval_twosided = 0.9982467752659725
    exp_pval_onesided = 0.49912338763298625

    # check
    assert obs_coef is not None
    assert isinstance(obs_coef, float)
    assert obs_coef == pytest.approx(exp_coef, rel=1e-10)
    assert obs_coef_se == pytest.approx(exp_coef_se, rel=1e-5)
    assert obs_tvalue == pytest.approx(exp_tvalue, rel=1e-5)
    assert obs_pval_twosided == pytest.approx(exp_pval_twosided, rel=5e-10)
    assert obs_pval_onesided == pytest.approx(exp_pval_onesided, rel=1e-5)


def test_gls_coef_positive_sub_matrix_random_phenotype0_lv800():
    phenotype_code = 0
    lv_code = "LV800"

    y = pd.read_pickle(
        DATA_DIR / f"multixcan-random_phenotype{phenotype_code}-pvalues.pkl.xz"
    )

    model = GLSPhenoplier(
        use_own_implementation=True,
        gene_corrs_file_path=DATA_DIR / "corr_mat.pkl.xz",
        debug_use_sub_gene_corr=True,
    )
    model.fit_named(lv_code, y)

    obs_coef = model.results.params.loc["lv"]
    obs_coef_se = model.results.bse.loc["lv"]
    obs_tvalue = model.results.tvalues.loc["lv"]
    obs_pval_twosided = model.results.pvalues.loc["lv"]
    obs_pval_onesided = model.results.pvalues_onesided.loc["lv"]

    exp_coef = 0.006103958765364128
    exp_coef_se = 0.013754152651896586
    exp_tvalue = 0.4437902442883271
    exp_pval_twosided = 0.6572091521023612
    exp_pval_onesided = 0.3286045760511806

    # check
    assert obs_coef is not None
    assert isinstance(obs_coef, float)
    assert obs_coef == pytest.approx(exp_coef, rel=1e-10)
    assert obs_coef_se == pytest.approx(exp_coef_se, rel=1e-10)
    assert obs_tvalue == pytest.approx(exp_tvalue, rel=1e-10)
    assert obs_pval_twosided == pytest.approx(exp_pval_twosided, rel=1e-10)
    assert obs_pval_onesided == pytest.approx(exp_pval_onesided, rel=1e-10)


def test_gls_sub_matrix_same_model_different_lvs():
    # run on same phenotype, but different lvs, using the same model
    # this mimics the use of GLSPhenoplier by gls_cli.py (console)
    phenotype_code = 6

    y = pd.read_pickle(
        DATA_DIR / f"multixcan-random_phenotype{phenotype_code}-pvalues.pkl.xz"
    )

    model = GLSPhenoplier(
        use_own_implementation=True,
        gene_corrs_file_path=DATA_DIR / "corr_mat.pkl.xz",
        debug_use_sub_gene_corr=True,
    )

    # first LV
    lv_code = "LV45"
    model.fit_named(lv_code, y)

    obs_coef = model.results.params.loc["lv"]
    obs_coef_se = model.results.bse.loc["lv"]
    obs_tvalue = model.results.tvalues.loc["lv"]
    obs_pval_twosided = model.results.pvalues.loc["lv"]
    obs_pval_onesided = model.results.pvalues_onesided.loc["lv"]

    exp_coef = -0.012985100862501646
    exp_coef_se = 0.011620815913625014
    exp_tvalue = -1.117400099873973
    exp_pval_twosided = 0.2638649762970155
    exp_pval_onesided = 0.8680675118514922

    # check
    assert obs_coef is not None
    assert isinstance(obs_coef, float)
    assert obs_coef == pytest.approx(exp_coef, rel=1e-5)
    assert obs_coef_se == pytest.approx(exp_coef_se, rel=1e-5)
    assert obs_tvalue == pytest.approx(exp_tvalue, rel=1e-5)
    assert obs_pval_twosided == pytest.approx(exp_pval_twosided, rel=1e-5)
    assert obs_pval_onesided == pytest.approx(exp_pval_onesided, rel=1e-5)

    # second LV
    lv_code = "LV455"
    model.fit_named(lv_code, y)

    obs_coef = model.results.params.loc["lv"]
    obs_coef_se = model.results.bse.loc["lv"]
    obs_tvalue = model.results.tvalues.loc["lv"]
    obs_pval_twosided = model.results.pvalues.loc["lv"]
    obs_pval_onesided = model.results.pvalues_onesided.loc["lv"]

    exp_coef = 0.005579886461567623
    exp_coef_se = 0.011701985846626066
    exp_tvalue = 0.4768324397842631
    exp_pval_twosided = 0.6334976228046719
    exp_pval_onesided = 0.31674881140233596

    # check
    assert obs_coef is not None
    assert isinstance(obs_coef, float)
    assert obs_coef == pytest.approx(exp_coef, rel=1e-5)
    assert obs_coef_se == pytest.approx(exp_coef_se, rel=1e-5)
    assert obs_tvalue == pytest.approx(exp_tvalue, rel=1e-5)
    assert obs_pval_twosided == pytest.approx(exp_pval_twosided, rel=1e-5)
    assert obs_pval_onesided == pytest.approx(exp_pval_onesided, rel=1e-5)


def test_gls_sub_matrix_same_model_different_lvs_gene_corr_is_folder():
    # run on same phenotype, but different lvs, using the same model
    # this mimics the use of GLSPhenoplier by gls_cli.py (console)
    phenotype_code = 6

    y = pd.read_pickle(
        DATA_DIR / f"multixcan-random_phenotype{phenotype_code}-pvalues.pkl.xz"
    )

    model = GLSPhenoplier(
        use_own_implementation=True,
        gene_corrs_file_path=DATA_DIR / "corr_mat_folder",
        debug_use_sub_gene_corr=True,
    )

    # first LV
    lv_code = "LV45"
    model.fit_named(lv_code, y)

    obs_coef = model.results.params.loc["lv"]
    obs_coef_se = model.results.bse.loc["lv"]
    obs_tvalue = model.results.tvalues.loc["lv"]
    obs_pval_twosided = model.results.pvalues.loc["lv"]
    obs_pval_onesided = model.results.pvalues_onesided.loc["lv"]

    exp_coef = -0.012985100862501646
    exp_coef_se = 0.011620815913625014
    exp_tvalue = -1.117400099873973
    exp_pval_twosided = 0.2638649762970155
    exp_pval_onesided = 0.8680675118514922

    # check
    assert obs_coef is not None
    assert isinstance(obs_coef, float)
    assert obs_coef == pytest.approx(exp_coef, rel=1e-5)
    assert obs_coef_se == pytest.approx(exp_coef_se, rel=1e-5)
    assert obs_tvalue == pytest.approx(exp_tvalue, rel=1e-5)
    assert obs_pval_twosided == pytest.approx(exp_pval_twosided, rel=1e-5)
    assert obs_pval_onesided == pytest.approx(exp_pval_onesided, rel=1e-5)

    # second LV
    lv_code = "LV455"
    model.fit_named(lv_code, y)

    obs_coef = model.results.params.loc["lv"]
    obs_coef_se = model.results.bse.loc["lv"]
    obs_tvalue = model.results.tvalues.loc["lv"]
    obs_pval_twosided = model.results.pvalues.loc["lv"]
    obs_pval_onesided = model.results.pvalues_onesided.loc["lv"]

    exp_coef = 0.005579886461567623
    exp_coef_se = 0.011701985846626066
    exp_tvalue = 0.4768324397842631
    exp_pval_twosided = 0.6334976228046719
    exp_pval_onesided = 0.31674881140233596

    # check
    assert obs_coef is not None
    assert isinstance(obs_coef, float)
    assert obs_coef == pytest.approx(exp_coef, rel=1e-5)
    assert obs_coef_se == pytest.approx(exp_coef_se, rel=1e-5)
    assert obs_tvalue == pytest.approx(exp_tvalue, rel=1e-5)
    assert obs_pval_twosided == pytest.approx(exp_pval_twosided, rel=1e-5)
    assert obs_pval_onesided == pytest.approx(exp_pval_onesided, rel=1e-5)


def test_gls_real_pheno_coef_positive_whooping_cough_lv570():
    phenotype_code = "whooping_cough"
    lv_code = "LV570"

    y = pd.read_pickle(
        DATA_DIR / f"multixcan-phenomexcan-{phenotype_code}-pvalues.pkl.xz"
    )

    model = GLSPhenoplier(
        use_own_implementation=True,
        gene_corrs_file_path=DATA_DIR / "corr_mat.pkl.xz",
        debug_use_sub_gene_corr=True,
    )
    model.fit_named(lv_code, y)

    obs_coef = model.results.params.loc["lv"]
    obs_coef_se = model.results.bse.loc["lv"]
    obs_tvalue = model.results.tvalues.loc["lv"]
    obs_pval_twosided = model.results.pvalues.loc["lv"]
    obs_pval_onesided = model.results.pvalues_onesided.loc["lv"]

    exp_coef = 0.010052199503207914
    exp_coef_se = 0.010343425212644251
    exp_tvalue = 0.971844364565006
    exp_pval_twosided = 0.3311644354099884
    exp_pval_onesided = 0.1655822177049942

    # check
    assert obs_coef is not None
    assert isinstance(obs_coef, float)
    assert obs_coef == pytest.approx(exp_coef, rel=1e-10)
    assert obs_coef_se == pytest.approx(exp_coef_se, rel=1e-10)
    assert obs_tvalue == pytest.approx(exp_tvalue, rel=1e-10)
    assert obs_pval_twosided == pytest.approx(exp_pval_twosided, rel=1e-10)
    assert obs_pval_onesided == pytest.approx(exp_pval_onesided, rel=1e-10)


def test_gls_real_pheno_coef_negative_wheezing_lv400():
    phenotype_code = "wheezing"
    lv_code = "LV400"

    y = pd.read_pickle(
        DATA_DIR / f"multixcan-phenomexcan-{phenotype_code}-pvalues.pkl.xz"
    )

    model = GLSPhenoplier(
        use_own_implementation=True,
        gene_corrs_file_path=DATA_DIR / "corr_mat.pkl.xz",
        debug_use_sub_gene_corr=True,
    )
    model.fit_named(lv_code, y)

    obs_coef = model.results.params.loc["lv"]
    obs_coef_se = model.results.bse.loc["lv"]
    obs_tvalue = model.results.tvalues.loc["lv"]
    obs_pval_twosided = model.results.pvalues.loc["lv"]
    obs_pval_onesided = model.results.pvalues_onesided.loc["lv"]

    exp_coef = -0.09501682279807443
    exp_coef_se = 0.01198940412636518
    exp_tvalue = -7.925066316609398
    exp_pval_twosided = 2.6672597928016686e-15
    exp_pval_onesided = 0.9999999999999987

    # check
    assert obs_coef is not None
    assert isinstance(obs_coef, float)
    assert obs_coef == pytest.approx(exp_coef, rel=1e-10)
    assert obs_coef_se == pytest.approx(exp_coef_se, rel=1e-10)
    assert obs_tvalue == pytest.approx(exp_tvalue, rel=1e-10)
    assert obs_pval_twosided == pytest.approx(exp_pval_twosided, rel=1e-10)
    assert obs_pval_onesided == pytest.approx(exp_pval_onesided, rel=1e-10)


def test_fit_with_phenotype_pandas_series():
    model = GLSPhenoplier(
        smultixcan_result_set_filepath=conf.PHENOMEXCAN[
            "SMULTIXCAN_MASHR_ZSCORES_FILE"
        ],
        use_own_implementation=True,
        debug_use_sub_gene_corr=True,
        gene_corrs_file_path=DATA_DIR / "corr_mat.pkl.xz",
    )

    # get number of genes to simulated phenotype
    lv_weights = GLSPhenoplier._get_lv_weights()

    np.random.seed(0)
    phenotype_data = pd.Series(
        np.abs(np.random.normal(size=lv_weights.shape[0])),
        index=lv_weights.index.copy(),
        name="Random phenotype",
    )
    model.fit_named("LV270", phenotype_data)

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
    model = GLSPhenoplier(
        smultixcan_result_set_filepath=conf.PHENOMEXCAN[
            "SMULTIXCAN_MASHR_ZSCORES_FILE"
        ],
        use_own_implementation=True,
        debug_use_sub_gene_corr=True,
        gene_corrs_file_path=DATA_DIR / "corr_mat.pkl.xz",
    )

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
    model = GLSPhenoplier(
        smultixcan_result_set_filepath=conf.PHENOMEXCAN[
            "SMULTIXCAN_MASHR_ZSCORES_FILE"
        ],
        use_own_implementation=True,
        debug_use_sub_gene_corr=True,
        gene_corrs_file_path=model.gene_corrs_file_path,
    )

    # np.random.shuffle(phenotype_data)
    phenotype_data = phenotype_data.sample(frac=1, random_state=0)
    model.fit_named("LV270", phenotype_data)
    assert model.results.df_resid == phenotype_data.shape[0] - 2

    pd.testing.assert_series_equal(expected_results.pvalues, model.results.pvalues)
    pd.testing.assert_series_equal(
        expected_results.pvalues_onesided, model.results.pvalues_onesided
    )


def test_fit_with_phenotype_pandas_series_less_genes():
    model = GLSPhenoplier(
        smultixcan_result_set_filepath=conf.PHENOMEXCAN[
            "SMULTIXCAN_MASHR_ZSCORES_FILE"
        ],
        use_own_implementation=True,
        debug_use_sub_gene_corr=True,
        gene_corrs_file_path=DATA_DIR / "corr_mat.pkl.xz",
    )

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
    model = GLSPhenoplier(
        smultixcan_result_set_filepath=conf.PHENOMEXCAN[
            "SMULTIXCAN_MASHR_ZSCORES_FILE"
        ],
        use_own_implementation=True,
        debug_use_sub_gene_corr=True,
        gene_corrs_file_path=model.gene_corrs_file_path,
    )

    # np.random.shuffle(phenotype_data)
    phenotype_data = phenotype_data.sample(frac=1, random_state=0)
    model.fit_named("LV270", phenotype_data)
    assert model.results.df_resid == phenotype_data_aligned.shape[0] - 2

    pd.testing.assert_series_equal(expected_results.pvalues, model.results.pvalues)
    pd.testing.assert_series_equal(
        expected_results.pvalues_onesided, model.results.pvalues_onesided
    )


def test_fit_with_phenotype_pandas_series_more_genes():
    model = GLSPhenoplier(
        smultixcan_result_set_filepath=conf.PHENOMEXCAN[
            "SMULTIXCAN_MASHR_ZSCORES_FILE"
        ],
        use_own_implementation=True,
        debug_use_sub_gene_corr=True,
        gene_corrs_file_path=DATA_DIR / "corr_mat.pkl.xz",
    )

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
    model = GLSPhenoplier(
        smultixcan_result_set_filepath=conf.PHENOMEXCAN[
            "SMULTIXCAN_MASHR_ZSCORES_FILE"
        ],
        use_own_implementation=True,
        debug_use_sub_gene_corr=True,
        gene_corrs_file_path=model.gene_corrs_file_path,
    )

    # np.random.shuffle(phenotype_data)
    phenotype_data = phenotype_data.sample(frac=1, random_state=0)
    with pytest.warns(UserWarning):
        model.fit_named("LV270", phenotype_data)
    assert model.results.df_resid == phenotype_data_aligned.shape[0] - 2

    pd.testing.assert_series_equal(expected_results.pvalues, model.results.pvalues)
    pd.testing.assert_series_equal(
        expected_results.pvalues_onesided, model.results.pvalues_onesided
    )


def test_fit_with_phenotype_pandas_series_with_nan_extra_genes_not_in_lv_models():
    model = GLSPhenoplier(
        smultixcan_result_set_filepath=conf.PHENOMEXCAN[
            "SMULTIXCAN_MASHR_ZSCORES_FILE"
        ],
        use_own_implementation=True,
        debug_use_sub_gene_corr=True,
        gene_corrs_file_path=DATA_DIR / "corr_mat.pkl.xz",
    )

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
    model = GLSPhenoplier(
        smultixcan_result_set_filepath=conf.PHENOMEXCAN[
            "SMULTIXCAN_MASHR_ZSCORES_FILE"
        ],
        use_own_implementation=True,
        debug_use_sub_gene_corr=True,
        gene_corrs_file_path=model.gene_corrs_file_path,
    )

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

    pd.testing.assert_series_equal(expected_results.pvalues, model.results.pvalues)
    pd.testing.assert_series_equal(
        expected_results.pvalues_onesided, model.results.pvalues_onesided
    )


def test_fit_with_phenotype_pandas_series_with_nan():
    model = GLSPhenoplier(
        smultixcan_result_set_filepath=conf.PHENOMEXCAN[
            "SMULTIXCAN_MASHR_ZSCORES_FILE"
        ],
        use_own_implementation=True,
        debug_use_sub_gene_corr=True,
        gene_corrs_file_path=DATA_DIR / "corr_mat.pkl.xz",
    )

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
    model = GLSPhenoplier(
        smultixcan_result_set_filepath=conf.PHENOMEXCAN[
            "SMULTIXCAN_MASHR_ZSCORES_FILE"
        ],
        use_own_implementation=True,
        debug_use_sub_gene_corr=True,
        gene_corrs_file_path=model.gene_corrs_file_path,
    )

    # generate same phenotype plus three genes with missing data
    phenotype_data.iloc[10] = np.nan
    phenotype_data.iloc[100] = np.nan
    phenotype_data.iloc[500] = np.nan
    assert phenotype_data.isna().sum() == 3

    model.fit_named("LV270", phenotype_data)
    assert (
        model.results.df_resid == phenotype_data_aligned.shape[0] - 2 - 3
    )  # remove nan genes

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
        use_own_implementation=True,
        debug_use_sub_gene_corr=True,
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
        use_own_implementation=True,
        debug_use_sub_gene_corr=True,
        gene_corrs_file_path=DATA_DIR / "sample-gene_corrs-1000g-en.pkl",
    )

    model.fit_named("LV270", phenotype_data)
    assert model.results.df_resid == phenotype_data.shape[0] - 2
    model2_results = model.results

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
        use_own_implementation=True,
        debug_use_sub_gene_corr=True,
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

    # now fit with OLS, results should be different
    model = GLSPhenoplier(
        conf.PHENOMEXCAN["SMULTIXCAN_MASHR_ZSCORES_FILE"],
        debug_use_ols=True,
    )

    model.fit_named("LV270", phenotype_data)
    assert model.results.df_resid == phenotype_data.shape[0] - 2
    model2_results = model.results

    assert not np.allclose(
        model1_results.pvalues.to_numpy(),
        model2_results.pvalues.to_numpy(),
    )

    assert not np.allclose(
        model1_results.pvalues_onesided.to_numpy(),
        model2_results.pvalues_onesided.to_numpy(),
    )
