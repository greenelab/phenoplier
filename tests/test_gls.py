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

    exp_coef = -0.0032814567822982274
    exp_coef_se = 0.008590718523010138
    exp_tvalue = -0.38197698754869974
    exp_pval_twosided = 0.7024910374237221
    exp_pval_onesided = 0.648754481288139

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

    exp_coef = 0.007824760447674654
    exp_coef_se = 0.010959087123929062
    exp_tvalue = 0.7139974670508243
    exp_pval_twosided = 0.47525462160232723
    exp_pval_onesided = 0.23762731080116362

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

    exp_coef = -0.0032814567822982274
    exp_coef_se = 0.008590718523010138
    exp_tvalue = -0.38197698754869974
    exp_pval_twosided = 0.7024910374237221
    exp_pval_onesided = 0.648754481288139

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

    exp_coef = 0.0015724547818453105
    exp_coef_se = 0.011027453856403382
    exp_tvalue = 0.1425945465128583
    exp_pval_twosided = 0.8866148655455224
    exp_pval_onesided = 0.4433074327727612

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

    exp_coef = -0.0032341246762881432
    exp_coef_se = 0.00859220860058952
    exp_tvalue = -0.3764020203217886
    exp_pval_twosided = 0.7066304479272314
    exp_pval_onesided = 0.6466847760363843

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

    exp_coef = 0.0014838894195672275
    exp_coef_se = 0.011297197897620825
    exp_tvalue = 0.13135021914414127
    exp_pval_twosided = 0.895502370998393
    exp_pval_onesided = 0.4477511854991965

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

    exp_coef = -0.0032341246762881432
    exp_coef_se = 0.00859220860058952
    exp_tvalue = -0.3764020203217886
    exp_pval_twosided = 0.7066304479272314
    exp_pval_onesided = 0.6466847760363843

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

    exp_coef = 0.0014838894195672275
    exp_coef_se = 0.011297197897620825
    exp_tvalue = 0.13135021914414127
    exp_pval_twosided = 0.895502370998393
    exp_pval_onesided = 0.4477511854991965

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

    exp_coef = 0.008257877072158701
    exp_coef_se = 0.01098379035278686
    exp_tvalue = 0.7518239885253702
    exp_pval_twosided = 0.4521844060369917
    exp_pval_onesided = 0.22609220301849586

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

    exp_coef = -0.0032535573271900795
    exp_coef_se = 0.008591651358118488
    exp_tvalue = -0.3786882394984177
    exp_pval_twosided = 0.7049318746949735
    exp_pval_onesided = 0.6475340626525132

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

    exp_coef = 0.0011536250507955133
    exp_coef_se = 0.011122650387146025
    exp_tvalue = 0.10371853925470038
    exp_pval_twosided = 0.9173959444428127
    exp_pval_onesided = 0.45869797222140635

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

    exp_coef = 0.011057584219855475
    exp_coef_se = 0.01127211381890071
    exp_tvalue = 0.980968112769983
    exp_pval_twosided = 0.32664533937794404
    exp_pval_onesided = 0.16332266968897202

    # check
    assert obs_coef is not None
    assert isinstance(obs_coef, float)
    assert obs_coef == pytest.approx(exp_coef, rel=1e-5)
    assert obs_coef_se == pytest.approx(exp_coef_se, rel=1e-5)
    assert obs_tvalue == pytest.approx(exp_tvalue, rel=1e-5)
    assert obs_pval_twosided == pytest.approx(exp_pval_twosided, rel=1e-5)
    assert obs_pval_onesided == pytest.approx(exp_pval_onesided, rel=1e-5)


@pytest.mark.skip(
    reason="wait for null simulations on full matrix before testing this option"
)
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


@pytest.mark.skip(
    reason="wait for null simulations on full matrix before testing this option"
)
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


@pytest.mark.skip(
    reason="wait for null simulations on full matrix before testing this option"
)
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


@pytest.mark.skip(
    reason="wait for null simulations on full matrix before testing this option"
)
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


@pytest.mark.skip(
    reason="wait for null simulations on full matrix before testing this option"
)
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


def test_gls_real_pheno_coef_negative_whooping_cough_lv570():
    phenotype_code = "whooping_cough"
    lv_code = "LV570"

    y = pd.read_pickle(
        DATA_DIR / f"multixcan-phenomexcan-{phenotype_code}-pvalues.pkl.xz"
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

    exp_coef = -0.005757975915714376
    exp_coef_se = 0.008790558023021578
    exp_tvalue = -0.6550182480605693
    exp_pval_twosided = 0.5124793998895305
    exp_pval_onesided = 0.7437603000552349

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
