"""
This file contains unit tests for the GLSPhenoplier class.
They are not here yet, only some tests description I want to include in the
future. The GLSPhenoplier class was tested using a notebook in the gls_testing
branch, and those will be moved here in the future.

This is reported in this issue: https://github.com/greenelab/phenoplier/issues/40
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import conf
from gls import GLSPhenoplier


DATA_DIR = (Path(__file__).parent / "data" / "gls").resolve()
assert DATA_DIR.exists()


def test_gls_coef_negative_full_matrix_random_phenotype():
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

    exp_coef = -0.003281047962518868
    exp_coef_se = 0.008589327735370812
    exp_tvalue = -0.3819912411779944
    exp_pval_twosided = 0.702480465360728
    exp_pval_onesided = 0.6487597673196361

    # check
    assert obs_coef is not None
    assert isinstance(obs_coef, float)
    assert obs_coef == pytest.approx(exp_coef, rel=1e-5)
    assert obs_coef_se == pytest.approx(exp_coef_se, rel=1e-5)
    assert obs_tvalue == pytest.approx(exp_tvalue, rel=1e-5)
    assert obs_pval_twosided == pytest.approx(exp_pval_twosided, rel=1e-5)
    assert obs_pval_onesided == pytest.approx(exp_pval_onesided, rel=1e-5)


def test_gls_coef_positive_full_matrix_random_phenotype():
    phenotype_code = 0
    lv_code = "LV801"

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

    exp_coef = 0.007927779814996835
    exp_coef_se = 0.010957049553795768
    exp_tvalue = 0.7235323502074036
    exp_pval_twosided = 0.4693791652354944
    exp_pval_onesided = 0.2346895826177472

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

    exp_coef = -0.003281047962518868
    exp_coef_se = 0.008589327735370812
    exp_tvalue = -0.3819912411779944
    exp_pval_twosided = 0.702480465360728
    exp_pval_onesided = 0.6487597673196361

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

    exp_coef = 0.0015604132867500819
    exp_coef_se = 0.011025668844737165
    exp_tvalue = 0.14152549915326973
    exp_pval_twosided = 0.8874592440993045
    exp_pval_onesided = 0.4437296220496523

    # check
    assert obs_coef is not None
    assert isinstance(obs_coef, float)
    assert obs_coef == pytest.approx(exp_coef, rel=1e-5)
    assert obs_coef_se == pytest.approx(exp_coef_se, rel=1e-5)
    assert obs_tvalue == pytest.approx(exp_tvalue, rel=1e-5)
    assert obs_pval_twosided == pytest.approx(exp_pval_twosided, rel=1e-5)
    assert obs_pval_onesided == pytest.approx(exp_pval_onesided, rel=1e-5)


def test_gls_with_covars_coef_negative_full_matrix_random_phenotype():
    phenotype_code = 6
    lv_code = "LV45"

    # make y a pandas.DataFrame
    # "y" is the dependant variable, and the rest are covariates
    y = pd.read_pickle(
        DATA_DIR / f"multixcan-random_phenotype{phenotype_code}-pvalues.pkl.xz"
    ).rename("y")
    y_covars = pd.read_pickle(
        DATA_DIR / f"multixcan-random_phenotype{phenotype_code}-covars.pkl.xz"
    )
    y = pd.concat([y, y_covars], axis=1)

    model = GLSPhenoplier(
        use_own_implementation=True,
        gene_corrs_file_path=DATA_DIR / "corr_mat.pkl.xz",
        debug_use_sub_gene_corr=False,
    )
    model.fit_named(lv_code, y)

    # make sure covariates were added
    assert model.results.params.shape[0] == 4
    assert model.results.bse.shape[0] == 4
    assert model.results.tvalues.shape[0] == 4
    assert model.results.pvalues.shape[0] == 4
    assert model.results.pvalues_onesided.shape[0] == 4

    assert model.results.pvalues.between(0.0, 1.0, inclusive="neither").all()
    assert model.results.pvalues_onesided.between(0.0, 1.0, inclusive="neither").all()

    obs_coef = model.results.params.loc["lv"]
    obs_coef_se = model.results.bse.loc["lv"]
    obs_tvalue = model.results.tvalues.loc["lv"]
    obs_pval_twosided = model.results.pvalues.loc["lv"]
    obs_pval_onesided = model.results.pvalues_onesided.loc["lv"]

    exp_coef = -0.0032326620431897984
    exp_coef_se = 0.008590802143381404
    exp_tvalue = -0.37629338788582534
    exp_pval_twosided = 0.7067111941216788
    exp_pval_onesided = 0.6466444029391605

    # check
    assert obs_coef is not None
    assert isinstance(obs_coef, float)
    assert obs_coef == pytest.approx(exp_coef, rel=1e-5)
    assert obs_coef_se == pytest.approx(exp_coef_se, rel=1e-5)
    assert obs_tvalue == pytest.approx(exp_tvalue, rel=1e-5)
    assert obs_pval_twosided == pytest.approx(exp_pval_twosided, rel=1e-5)
    assert obs_pval_onesided == pytest.approx(exp_pval_onesided, rel=1e-5)


def test_gls_with_covars_coef_positive_full_matrix_random_phenotype():
    phenotype_code = 6
    lv_code = "LV455"

    # make y a pandas.DataFrame
    # "y" is the dependant variable, and the rest are covariates
    y = pd.read_pickle(
        DATA_DIR / f"multixcan-random_phenotype{phenotype_code}-pvalues.pkl.xz"
    ).rename("y")
    y_covars = pd.read_pickle(
        DATA_DIR / f"multixcan-random_phenotype{phenotype_code}-covars.pkl.xz"
    )
    y = pd.concat([y, y_covars], axis=1)

    model = GLSPhenoplier(
        use_own_implementation=True,
        gene_corrs_file_path=DATA_DIR / "corr_mat.pkl.xz",
        debug_use_sub_gene_corr=False,
    )
    model.fit_named(lv_code, y)

    # make sure covariates were added
    assert model.results.params.shape[0] == 4
    assert model.results.bse.shape[0] == 4
    assert model.results.tvalues.shape[0] == 4
    assert model.results.pvalues.shape[0] == 4
    assert model.results.pvalues_onesided.shape[0] == 4

    assert model.results.pvalues.between(0.0, 1.0, inclusive="neither").all()
    assert model.results.pvalues_onesided.between(0.0, 1.0, inclusive="neither").all()

    obs_coef = model.results.params.loc["lv"]
    obs_coef_se = model.results.bse.loc["lv"]
    obs_tvalue = model.results.tvalues.loc["lv"]
    obs_pval_twosided = model.results.pvalues.loc["lv"]
    obs_pval_onesided = model.results.pvalues_onesided.loc["lv"]

    exp_coef = 0.0014516113831524813
    exp_coef_se = 0.011295356092071307
    exp_tvalue = 0.12851399914442976
    exp_pval_twosided = 0.8977462345701058
    exp_pval_onesided = 0.4488731172850529

    # check
    assert obs_coef is not None
    assert isinstance(obs_coef, float)
    assert obs_coef == pytest.approx(exp_coef, rel=1e-5)
    assert obs_coef_se == pytest.approx(exp_coef_se, rel=1e-5)
    assert obs_tvalue == pytest.approx(exp_tvalue, rel=1e-5)
    assert obs_pval_twosided == pytest.approx(exp_pval_twosided, rel=1e-5)
    assert obs_pval_onesided == pytest.approx(exp_pval_onesided, rel=1e-5)


def test_gls_with_covars_full_matrix_same_model_different_lvs():
    # run on same phenotype, but different lvs, using the same model
    # this mimics the use of GLSPhenoplier by gls_cli.py (console)
    phenotype_code = 6

    # make y a pandas.DataFrame
    # "y" is the dependant variable, and the rest are covariates
    y = pd.read_pickle(
        DATA_DIR / f"multixcan-random_phenotype{phenotype_code}-pvalues.pkl.xz"
    ).rename("y")
    y_covars = pd.read_pickle(
        DATA_DIR / f"multixcan-random_phenotype{phenotype_code}-covars.pkl.xz"
    )
    y = pd.concat([y, y_covars], axis=1)

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

    exp_coef = -0.0032326620431897984
    exp_coef_se = 0.008590802143381404
    exp_tvalue = -0.37629338788582534
    exp_pval_twosided = 0.7067111941216788
    exp_pval_onesided = 0.6466444029391605

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

    exp_coef = 0.0014516113831524813
    exp_coef_se = 0.011295356092071307
    exp_tvalue = 0.12851399914442976
    exp_pval_twosided = 0.8977462345701058
    exp_pval_onesided = 0.4488731172850529

    # check
    assert obs_coef is not None
    assert isinstance(obs_coef, float)
    assert obs_coef == pytest.approx(exp_coef, rel=1e-5)
    assert obs_coef_se == pytest.approx(exp_coef_se, rel=1e-5)
    assert obs_tvalue == pytest.approx(exp_tvalue, rel=1e-5)
    assert obs_pval_twosided == pytest.approx(exp_pval_twosided, rel=1e-5)
    assert obs_pval_onesided == pytest.approx(exp_pval_onesided, rel=1e-5)


def test_gls_with_covars_using_logarithms_full_matrix_random_phenotype():
    phenotype_code = 0
    lv_code = "LV801"

    # make y a pandas.DataFrame
    # "y" is the dependant variable, and the rest are covariates
    y = pd.read_pickle(
        DATA_DIR / f"multixcan-random_phenotype{phenotype_code}-pvalues.pkl.xz"
    ).rename("y")
    y_covars = pd.read_pickle(
        DATA_DIR / f"multixcan-random_phenotype{phenotype_code}-covars.pkl.xz"
    )
    y = pd.concat([y, y_covars], axis=1)

    model = GLSPhenoplier(
        use_own_implementation=True,
        gene_corrs_file_path=DATA_DIR / "corr_mat.pkl.xz",
        debug_use_sub_gene_corr=False,
    )
    model.fit_named(lv_code, y)

    # make sure covariates were added
    assert model.results.params.shape[0] == 6
    assert model.results.bse.shape[0] == 6
    assert model.results.tvalues.shape[0] == 6
    assert model.results.pvalues.shape[0] == 6
    assert model.results.pvalues_onesided.shape[0] == 6

    assert model.results.pvalues.between(0.0, 1.0, inclusive="neither").all()
    assert model.results.pvalues_onesided.between(0.0, 1.0, inclusive="neither").all()

    obs_coef = model.results.params.loc["lv"]
    obs_coef_se = model.results.bse.loc["lv"]
    obs_tvalue = model.results.tvalues.loc["lv"]
    obs_pval_twosided = model.results.pvalues.loc["lv"]
    obs_pval_onesided = model.results.pvalues_onesided.loc["lv"]

    exp_coef = 0.008380935035632255
    exp_coef_se = 0.010981760912289579
    exp_tvalue = 0.7631685940506349
    exp_pval_twosided = 0.4453908279763241
    exp_pval_onesided = 0.22269541398816206

    # check
    assert obs_coef is not None
    assert isinstance(obs_coef, float)
    assert obs_coef == pytest.approx(exp_coef, rel=1e-5)
    assert obs_coef_se == pytest.approx(exp_coef_se, rel=1e-5)
    assert obs_tvalue == pytest.approx(exp_tvalue, rel=1e-5)
    assert obs_pval_twosided == pytest.approx(exp_pval_twosided, rel=1e-5)
    assert obs_pval_onesided == pytest.approx(exp_pval_onesided, rel=1e-5)


def test_gls_with_snplevel_covars_coef_negative_full_matrix_random_phenotype():
    phenotype_code = 6
    lv_code = "LV45"

    # make y a pandas.DataFrame
    # "y" is the dependant variable, and the rest are covariates
    y = pd.read_pickle(
        DATA_DIR / f"multixcan-random_phenotype{phenotype_code}-pvalues.pkl.xz"
    ).rename("y")
    y_covars = pd.read_pickle(
        DATA_DIR / f"multixcan-random_phenotype{phenotype_code}-snplevel_covars.pkl.xz"
    )
    y = pd.concat([y, y_covars], axis=1)

    model = GLSPhenoplier(
        use_own_implementation=True,
        gene_corrs_file_path=DATA_DIR / "corr_mat.pkl.xz",
        debug_use_sub_gene_corr=False,
    )
    model.fit_named(lv_code, y)

    # make sure covariates were added
    assert model.results.params.shape[0] == 4
    assert model.results.bse.shape[0] == 4
    assert model.results.tvalues.shape[0] == 4
    assert model.results.pvalues.shape[0] == 4
    assert model.results.pvalues_onesided.shape[0] == 4

    assert model.results.pvalues.between(0.0, 1.0, inclusive="neither").all()
    assert model.results.pvalues_onesided.between(0.0, 1.0, inclusive="neither").all()

    obs_coef = model.results.params.loc["lv"]
    obs_coef_se = model.results.bse.loc["lv"]
    obs_tvalue = model.results.tvalues.loc["lv"]
    obs_pval_twosided = model.results.pvalues.loc["lv"]
    obs_pval_onesided = model.results.pvalues_onesided.loc["lv"]

    exp_coef = -0.003227698021391237
    exp_coef_se = 0.00859069240125604
    exp_tvalue = -0.37572035764187267
    exp_pval_twosided = 0.7071371805607842
    exp_pval_onesided = 0.646431409719608

    # check
    assert obs_coef is not None
    assert isinstance(obs_coef, float)
    assert obs_coef == pytest.approx(exp_coef, rel=1e-5)
    assert obs_coef_se == pytest.approx(exp_coef_se, rel=1e-5)
    assert obs_tvalue == pytest.approx(exp_tvalue, rel=1e-5)
    assert obs_pval_twosided == pytest.approx(exp_pval_twosided, rel=1e-5)
    assert obs_pval_onesided == pytest.approx(exp_pval_onesided, rel=1e-5)


def test_gls_with_snplevel_covars_coef_positive_full_matrix_random_phenotype():
    phenotype_code = 6
    lv_code = "LV455"

    # make y a pandas.DataFrame
    # "y" is the dependant variable, and the rest are covariates
    y = pd.read_pickle(
        DATA_DIR / f"multixcan-random_phenotype{phenotype_code}-pvalues.pkl.xz"
    ).rename("y")
    y_covars = pd.read_pickle(
        DATA_DIR / f"multixcan-random_phenotype{phenotype_code}-snplevel_covars.pkl.xz"
    )
    y = pd.concat([y, y_covars], axis=1)

    model = GLSPhenoplier(
        use_own_implementation=True,
        gene_corrs_file_path=DATA_DIR / "corr_mat.pkl.xz",
        debug_use_sub_gene_corr=False,
    )
    model.fit_named(lv_code, y)

    # make sure covariates were added
    assert model.results.params.shape[0] == 4
    assert model.results.bse.shape[0] == 4
    assert model.results.tvalues.shape[0] == 4
    assert model.results.pvalues.shape[0] == 4
    assert model.results.pvalues_onesided.shape[0] == 4

    assert model.results.pvalues.between(0.0, 1.0, inclusive="neither").all()
    assert model.results.pvalues_onesided.between(0.0, 1.0, inclusive="neither").all()

    obs_coef = model.results.params.loc["lv"]
    obs_coef_se = model.results.bse.loc["lv"]
    obs_tvalue = model.results.tvalues.loc["lv"]
    obs_pval_twosided = model.results.pvalues.loc["lv"]
    obs_pval_onesided = model.results.pvalues_onesided.loc["lv"]

    exp_coef = 0.0018276464593873108
    exp_coef_se = 0.01139385148318154
    exp_tvalue = 0.16040637900933666
    exp_pval_twosided = 0.8725659975131534
    exp_pval_onesided = 0.4362829987565767

    # check
    assert obs_coef is not None
    assert isinstance(obs_coef, float)
    assert obs_coef == pytest.approx(exp_coef, rel=1e-5)
    assert obs_coef_se == pytest.approx(exp_coef_se, rel=1e-5)
    assert obs_tvalue == pytest.approx(exp_tvalue, rel=1e-5)
    assert obs_pval_twosided == pytest.approx(exp_pval_twosided, rel=1e-5)
    assert obs_pval_onesided == pytest.approx(exp_pval_onesided, rel=1e-5)


def test_gls_with_snplevel_covars_using_logarithms_full_matrix_random_phenotype():
    phenotype_code = 0
    lv_code = "LV801"

    # make y a pandas.DataFrame
    # "y" is the dependant variable, and the rest are covariates
    y = pd.read_pickle(
        DATA_DIR / f"multixcan-random_phenotype{phenotype_code}-pvalues.pkl.xz"
    ).rename("y")
    y_covars = pd.read_pickle(
        DATA_DIR / f"multixcan-random_phenotype{phenotype_code}-snplevel_covars.pkl.xz"
    )
    y = pd.concat([y, y_covars], axis=1)

    model = GLSPhenoplier(
        use_own_implementation=True,
        gene_corrs_file_path=DATA_DIR / "corr_mat.pkl.xz",
        debug_use_sub_gene_corr=False,
    )
    model.fit_named(lv_code, y)

    # make sure covariates were added
    assert model.results.params.shape[0] == 6
    assert model.results.bse.shape[0] == 6
    assert model.results.tvalues.shape[0] == 6
    assert model.results.pvalues.shape[0] == 6
    assert model.results.pvalues_onesided.shape[0] == 6

    assert model.results.pvalues.between(0.0, 1.0, inclusive="neither").all()
    assert model.results.pvalues_onesided.between(0.0, 1.0, inclusive="neither").all()

    obs_coef = model.results.params.loc["lv"]
    obs_coef_se = model.results.bse.loc["lv"]
    obs_tvalue = model.results.tvalues.loc["lv"]
    obs_pval_twosided = model.results.pvalues.loc["lv"]
    obs_pval_onesided = model.results.pvalues_onesided.loc["lv"]

    exp_coef = 0.008384051885022486
    exp_coef_se = 0.010977363554150646
    exp_tvalue = 0.7637582415544938
    exp_pval_twosided = 0.44503932280432557
    exp_pval_onesided = 0.22251966140216278

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

    exp_coef = -0.007758262616074735
    exp_coef_se = 0.010320432763493127
    exp_tvalue = -0.7517381096186531
    exp_pval_twosided = 0.45223603891122643
    exp_pval_onesided = 0.7738819805443868

    # check
    assert obs_coef is not None
    assert isinstance(obs_coef, float)
    assert obs_coef == pytest.approx(exp_coef, rel=1e-10)
    assert obs_coef_se == pytest.approx(exp_coef_se, rel=1e-10)
    assert obs_tvalue == pytest.approx(exp_tvalue, rel=1e-10)
    assert obs_pval_twosided == pytest.approx(exp_pval_twosided, rel=1e-10)
    assert obs_pval_onesided == pytest.approx(exp_pval_onesided, rel=1e-10)


def test_gls_coef_positive_sub_matrix_random_phenotype():
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

    exp_coef = 0.0004030553781634736
    exp_coef_se = 0.004162484428442315
    exp_tvalue = 0.0968304831146972
    exp_pval_twosided = 0.9228640287589255
    exp_pval_onesided = 0.46143201437946274

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

    exp_coef = 0.004170133621705775
    exp_coef_se = 0.014013329781567654
    exp_tvalue = 0.2975833500465346
    exp_pval_twosided = 0.7660307826493885
    exp_pval_onesided = 0.38301539132469425

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

    exp_coef = -0.007758262616074735
    exp_coef_se = 0.010320432763493127
    exp_tvalue = -0.7517381096186531
    exp_pval_twosided = 0.45223603891122643
    exp_pval_onesided = 0.7738819805443868

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

    exp_coef = 0.006302906163160309
    exp_coef_se = 0.011669857436439947
    exp_tvalue = 0.5401013849131561
    exp_pval_twosided = 0.5891457914757958
    exp_pval_onesided = 0.2945728957378979

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

    exp_coef = -0.007758262616074735
    exp_coef_se = 0.010320432763493127
    exp_tvalue = -0.7517381096186531
    exp_pval_twosided = 0.45223603891122643
    exp_pval_onesided = 0.7738819805443868

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

    exp_coef = 0.006302906163160309
    exp_coef_se = 0.011669857436439947
    exp_tvalue = 0.5401013849131561
    exp_pval_twosided = 0.5891457914757958
    exp_pval_onesided = 0.2945728957378979

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

    exp_coef = 0.0049802005250721
    exp_coef_se = 0.010338597588777377
    exp_tvalue = 0.4817094854797467
    exp_pval_twosided = 0.6300287052648046
    exp_pval_onesided = 0.3150143526324023

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

    exp_coef = -0.1787005272685236
    exp_coef_se = 0.012616403393185567
    exp_tvalue = -14.164141847671436
    exp_pval_twosided = 7.167212012881414e-45
    exp_pval_onesided = 1.0

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
        use_own_implementation=True,
        gene_corrs_file_path=DATA_DIR / "corr_mat.pkl.xz",
    )

    y = pd.read_pickle(DATA_DIR / f"multixcan-random_phenotype0-pvalues.pkl.xz")
    model.fit_named("LV270", y)

    # get observed two-sided pvalue
    obs_pval_twosided = model.results.pvalues.loc["lv"]
    assert obs_pval_twosided is not None
    assert isinstance(obs_pval_twosided, float)
    assert obs_pval_twosided > 0.0
    assert obs_pval_twosided < 1.0

    # get observed one-sided pvalue
    obs_pval_onesided = model.results.pvalues_onesided.loc["lv"]
    assert obs_pval_onesided is not None
    assert isinstance(obs_pval_onesided, float)
    assert obs_pval_onesided > 0.0
    assert obs_pval_onesided < 1.0


def test_fit_with_phenotype_pandas_series_genes_not_aligned():
    model = GLSPhenoplier(
        use_own_implementation=True,
        gene_corrs_file_path=DATA_DIR / "corr_mat.pkl.xz",
    )

    lv_weights = GLSPhenoplier._get_lv_weights()
    gene_corrs = GLSPhenoplier._get_gene_corrs(model.gene_corrs_file_path)

    phenotype_data = pd.read_pickle(
        DATA_DIR / f"multixcan-random_phenotype6-pvalues.pkl.xz"
    )

    # match all genes and align
    phenotype_data, lv_weights = GLSPhenoplier.match_and_align_genes(
        phenotype_data, lv_weights, gene_corrs
    )[0:2]

    model.fit_named("LV270", phenotype_data)
    assert model.results.df_resid == phenotype_data.shape[0] - 2
    exp_pval_onesided = model.results.pvalues_onesided.loc["lv"]
    assert exp_pval_onesided is not None
    assert isinstance(exp_pval_onesided, float)
    assert exp_pval_onesided > 0.0
    assert exp_pval_onesided < 1.0

    expected_results = model.results

    # run again with genes reordered in the phenotype
    # results should be the same
    model = GLSPhenoplier(
        use_own_implementation=True,
        gene_corrs_file_path=DATA_DIR / "corr_mat.pkl.xz",
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
        use_own_implementation=True,
        gene_corrs_file_path=DATA_DIR / "corr_mat.pkl.xz",
    )

    # get data and simulate phenotype
    lv_weights = GLSPhenoplier._get_lv_weights()
    gene_corrs = GLSPhenoplier._get_gene_corrs(model.gene_corrs_file_path)

    n_genes = 3000
    phenotype_data = pd.read_pickle(
        DATA_DIR / f"multixcan-random_phenotype6-pvalues.pkl.xz"
    )
    phenotype_data = phenotype_data.sample(n=n_genes, random_state=0)

    # match all genes and align
    phenotype_data_aligned, lv_weights = GLSPhenoplier.match_and_align_genes(
        phenotype_data, lv_weights, gene_corrs
    )[0:2]

    model.fit_named("LV270", phenotype_data)
    assert model.results.df_resid == phenotype_data_aligned.shape[0] - 2
    exp_pval_onesided = model.results.pvalues_onesided.loc["lv"]
    assert exp_pval_onesided is not None
    assert isinstance(exp_pval_onesided, float)
    assert exp_pval_onesided > 0.0
    assert exp_pval_onesided < 1.0
    expected_results = model.results

    # run again with genes reordered in the phenotype
    # results should be the same
    model = GLSPhenoplier(
        use_own_implementation=True,
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
        use_own_implementation=True,
        gene_corrs_file_path=DATA_DIR / "corr_mat.pkl.xz",
    )

    # get data and simulate phenotype
    lv_weights = GLSPhenoplier._get_lv_weights()
    gene_corrs = GLSPhenoplier._get_gene_corrs(model.gene_corrs_file_path)

    np.random.seed(0)
    phenotype_data = pd.read_pickle(
        DATA_DIR / f"multixcan-random_phenotype6-pvalues.pkl.xz"
    )
    # add a gene (I made up a name: AGENE) that does not exist in LV models
    phenotype_data.loc["AGENE"] = np.random.rand()

    # match all genes and align
    phenotype_data_aligned, lv_weights = GLSPhenoplier.match_and_align_genes(
        phenotype_data, lv_weights, gene_corrs
    )[0:2]

    with pytest.warns(UserWarning):
        model.fit_named("LV270", phenotype_data)
    assert model.results.df_resid == phenotype_data_aligned.shape[0] - 2
    exp_pval_onesided = model.results.pvalues_onesided.loc["lv"]
    assert exp_pval_onesided is not None
    assert isinstance(exp_pval_onesided, float)
    assert exp_pval_onesided > 0.0
    assert exp_pval_onesided < 1.0
    expected_results = model.results

    # run again with genes reordered in the phenotype
    # results should be the same
    model = GLSPhenoplier(
        use_own_implementation=True,
        gene_corrs_file_path=model.gene_corrs_file_path,
    )

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
        use_own_implementation=True,
        gene_corrs_file_path=DATA_DIR / "corr_mat.pkl.xz",
    )

    # original run
    # get data and simulate phenotype
    lv_weights = GLSPhenoplier._get_lv_weights()
    gene_corrs = GLSPhenoplier._get_gene_corrs(model.gene_corrs_file_path)

    phenotype_data = pd.read_pickle(
        DATA_DIR / f"multixcan-random_phenotype6-pvalues.pkl.xz"
    )

    # match all genes and align
    phenotype_data_aligned, lv_weights = GLSPhenoplier.match_and_align_genes(
        phenotype_data, lv_weights, gene_corrs
    )[0:2]

    model.fit_named("LV270", phenotype_data)
    assert model.results.df_resid == phenotype_data_aligned.shape[0] - 2
    exp_pval_onesided = model.results.pvalues_onesided.loc["lv"]
    assert exp_pval_onesided is not None
    assert isinstance(exp_pval_onesided, float)
    assert exp_pval_onesided > 0.0
    assert exp_pval_onesided < 1.0
    expected_results = model.results

    # run again with extra genes with missing data
    # results should be the same
    model = GLSPhenoplier(
        use_own_implementation=True,
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
        use_own_implementation=True,
        gene_corrs_file_path=DATA_DIR / "corr_mat.pkl.xz",
    )

    # original run
    # get data and simulate phenotype
    lv_weights = GLSPhenoplier._get_lv_weights()
    gene_corrs = GLSPhenoplier._get_gene_corrs(model.gene_corrs_file_path)

    phenotype_data = pd.read_pickle(
        DATA_DIR / f"multixcan-random_phenotype6-pvalues.pkl.xz"
    )

    # match all genes and align
    phenotype_data_aligned, lv_weights = GLSPhenoplier.match_and_align_genes(
        phenotype_data, lv_weights, gene_corrs
    )[0:2]

    model.fit_named("LV270", phenotype_data)
    assert model.results.df_resid == phenotype_data_aligned.shape[0] - 2
    exp_pval_onesided = model.results.pvalues_onesided.loc["lv"]
    assert exp_pval_onesided is not None
    assert isinstance(exp_pval_onesided, float)
    assert exp_pval_onesided > 0.0
    assert exp_pval_onesided < 1.0
    expected_results = model.results

    # run again with _existing_ genes with missing data
    # results should NOT be the same
    model = GLSPhenoplier(
        use_own_implementation=True,
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
        use_own_implementation=True,
        gene_corrs_file_path=DATA_DIR / "sample-gene_corrs-gtex_v8-mashr.pkl",
    )

    # get data and simulate phenotype
    lv_weights = GLSPhenoplier._get_lv_weights()
    gene_corrs = GLSPhenoplier._get_gene_corrs(model.gene_corrs_file_path)

    phenotype_data = pd.read_pickle(
        DATA_DIR / f"multixcan-random_phenotype6-pvalues.pkl.xz"
    )

    # match all genes and align
    phenotype_data, lv_weights = GLSPhenoplier.match_and_align_genes(
        phenotype_data, lv_weights, gene_corrs
    )[0:2]

    # fit with gtex/mashr
    model.fit_named("LV270", phenotype_data)
    assert model.results.df_resid == phenotype_data.shape[0] - 2
    exp_pval_onesided = model.results.pvalues_onesided.loc["lv"]
    assert exp_pval_onesided is not None
    assert isinstance(exp_pval_onesided, float)
    assert exp_pval_onesided > 0.0
    assert exp_pval_onesided < 1.0
    model1_results = model.results

    # fit with 1000g/elastic net
    model = GLSPhenoplier(
        use_own_implementation=True,
        gene_corrs_file_path=DATA_DIR / "sample-gene_corrs-1000g-en.pkl",
    )

    model.fit_named("LV270", phenotype_data)
    assert model.results.df_resid == phenotype_data.shape[0] - 2
    exp_pval_onesided = model.results.pvalues_onesided.loc["lv"]
    assert exp_pval_onesided is not None
    assert isinstance(exp_pval_onesided, float)
    assert exp_pval_onesided > 0.0
    assert exp_pval_onesided < 1.0
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
        use_own_implementation=True,
        gene_corrs_file_path=DATA_DIR / "sample-gene_corrs-gtex_v8-mashr.pkl",
    )

    # get data and simulate phenotype
    lv_weights = GLSPhenoplier._get_lv_weights()
    gene_corrs = GLSPhenoplier._get_gene_corrs(model.gene_corrs_file_path)

    phenotype_data = pd.read_pickle(
        DATA_DIR / f"multixcan-random_phenotype6-pvalues.pkl.xz"
    )

    # match all genes and align
    phenotype_data, lv_weights = GLSPhenoplier.match_and_align_genes(
        phenotype_data, lv_weights, gene_corrs
    )[0:2]

    # fit with using GLS model
    model.fit_named("LV270", phenotype_data)
    assert model.results.df_resid == phenotype_data.shape[0] - 2
    exp_pval_onesided = model.results.pvalues_onesided.loc["lv"]
    assert exp_pval_onesided is not None
    assert isinstance(exp_pval_onesided, float)
    assert exp_pval_onesided > 0.0
    assert exp_pval_onesided < 1.0
    model1_results = model.results

    # now fit with OLS, results should be different
    model = GLSPhenoplier(
        debug_use_ols=True,
    )

    model.fit_named("LV270", phenotype_data)
    assert model.results.df_resid == phenotype_data.shape[0] - 2
    exp_pval_onesided = model.results.pvalues_onesided.loc["lv"]
    assert exp_pval_onesided is not None
    assert isinstance(exp_pval_onesided, float)
    assert exp_pval_onesided > 0.0
    assert exp_pval_onesided < 1.0
    model2_results = model.results

    assert not np.allclose(
        model1_results.pvalues.to_numpy(),
        model2_results.pvalues.to_numpy(),
    )

    assert not np.allclose(
        model1_results.pvalues_onesided.to_numpy(),
        model2_results.pvalues_onesided.to_numpy(),
    )
