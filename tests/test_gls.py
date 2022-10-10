"""
This file contains unit tests for the GLSPhenoplier class, which has a more
efficient implementation of the GLS model. Here, GLSPhenoplier is compared with
use cases from statsmodels.GLS generated in notebook:
    nbs/15_gsa_gls/misc/10_10-gls-generate_cases-cases.ipynb
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

    exp_coef = -0.10018201664770203
    exp_coef_se = 0.09298021617384379
    exp_tvalue = -1.0774551917624404
    exp_pval_twosided = 0.28131731604765614
    exp_pval_onesided = 0.859341341976172

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

    exp_coef = 0.10885793222623774
    exp_coef_se = 0.11026400471153004
    exp_tvalue = 0.9872481279002079
    exp_pval_twosided = 0.32355810271013086
    exp_pval_onesided = 0.16177905135506543

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

    exp_coef = -0.10018201664770203
    exp_coef_se = 0.09298021617384379
    exp_tvalue = -1.0774551917624404
    exp_pval_twosided = 0.28131731604765614
    exp_pval_onesided = 0.859341341976172

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

    exp_coef = 0.0784587858266203
    exp_coef_se = 0.1152051853461905
    exp_tvalue = 0.6810351946472929
    exp_pval_twosided = 0.4958737072729271
    exp_pval_onesided = 0.24793685363646356

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

    exp_coef = -0.10052902446730924
    exp_coef_se = 0.09300042682237371
    exp_tvalue = -1.0809522913192084
    exp_pval_twosided = 0.27975882566803706
    exp_pval_onesided = 0.8601205871659815

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

    exp_coef = -0.13122859992487934
    exp_coef_se = 0.14197922645725486
    exp_tvalue = -0.9242802852175549
    exp_pval_twosided = 0.3553750345424116
    exp_pval_onesided = 0.8223124827287942

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

    exp_coef = 0.01045784784294252
    exp_coef_se = 0.12718873964387706
    exp_tvalue = 0.08222306371007403
    exp_pval_twosided = 0.9344718886260288
    exp_pval_onesided = 0.4672359443130144

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

    exp_coef = 0.28001523677025164
    exp_coef_se = 0.1561087448671094
    exp_tvalue = 1.79371909631725
    exp_pval_twosided = 0.07290491285969262
    exp_pval_onesided = 0.03645245642984631

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

    exp_coef = -0.13122859992487934
    exp_coef_se = 0.14197922645725486
    exp_tvalue = -0.9242802852175549
    exp_pval_twosided = 0.3553750345424116
    exp_pval_onesided = 0.8223124827287942

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

    exp_coef = 0.14024604432028576
    exp_coef_se = 0.12772108665506693
    exp_tvalue = 1.0980649162424108
    exp_pval_twosided = 0.2722171958691433
    exp_pval_onesided = 0.13610859793457164

    # check
    assert obs_coef is not None
    assert isinstance(obs_coef, float)
    assert obs_coef == pytest.approx(exp_coef, rel=1e-5)
    assert obs_coef_se == pytest.approx(exp_coef_se, rel=1e-5)
    assert obs_tvalue == pytest.approx(exp_tvalue, rel=1e-5)
    assert obs_pval_twosided == pytest.approx(exp_pval_twosided, rel=1e-5)
    assert obs_pval_onesided == pytest.approx(exp_pval_onesided, rel=1e-5)


def test_gls_with_covars_coef_negative_sub_matrix_random_phenotype():
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
        debug_use_sub_gene_corr=True,
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

    exp_coef = -0.13057289396289432
    exp_coef_se = 0.14208177608559344
    exp_tvalue = -0.9189981823160354
    exp_pval_twosided = 0.35813095213395807
    exp_pval_onesided = 0.820934523933021

    # check
    assert obs_coef is not None
    assert isinstance(obs_coef, float)
    assert obs_coef == pytest.approx(exp_coef, rel=1e-5)
    assert obs_coef_se == pytest.approx(exp_coef_se, rel=1e-5)
    assert obs_tvalue == pytest.approx(exp_tvalue, rel=1e-5)
    assert obs_pval_twosided == pytest.approx(exp_pval_twosided, rel=1e-5)
    assert obs_pval_onesided == pytest.approx(exp_pval_onesided, rel=1e-5)


def test_gls_with_covars_coef_negative_sub_matrix_random_phenotype_gene_corr_is_folder():
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
        gene_corrs_file_path=DATA_DIR / "corr_mat_folder",
        debug_use_sub_gene_corr=True,
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

    exp_coef = -0.13057289396289432
    exp_coef_se = 0.14208177608559344
    exp_tvalue = -0.9189981823160354
    exp_pval_twosided = 0.35813095213395807
    exp_pval_onesided = 0.820934523933021

    # check
    assert obs_coef is not None
    assert isinstance(obs_coef, float)
    assert obs_coef == pytest.approx(exp_coef, rel=1e-2)
    assert obs_coef_se == pytest.approx(exp_coef_se, rel=1e-3)
    assert obs_tvalue == pytest.approx(exp_tvalue, rel=1e-2)
    assert obs_pval_twosided == pytest.approx(exp_pval_twosided, rel=1e-2)
    assert obs_pval_onesided == pytest.approx(exp_pval_onesided, rel=1e-2)


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

    exp_coef = -0.13122859992487934
    exp_coef_se = 0.14197922645725486
    exp_tvalue = -0.9242802852175549
    exp_pval_twosided = 0.3553750345424116
    exp_pval_onesided = 0.8223124827287942

    # check
    assert obs_coef is not None
    assert isinstance(obs_coef, float)
    assert obs_coef == pytest.approx(exp_coef, rel=1e-2)
    assert obs_coef_se == pytest.approx(exp_coef_se, rel=1e-4)
    assert obs_tvalue == pytest.approx(exp_tvalue, rel=1e-2)
    assert obs_pval_twosided == pytest.approx(exp_pval_twosided, rel=1e-2)
    assert obs_pval_onesided == pytest.approx(exp_pval_onesided, rel=1e-2)

    # second LV
    lv_code = "LV455"
    model.fit_named(lv_code, y)

    obs_coef = model.results.params.loc["lv"]
    obs_coef_se = model.results.bse.loc["lv"]
    obs_tvalue = model.results.tvalues.loc["lv"]
    obs_pval_twosided = model.results.pvalues.loc["lv"]
    obs_pval_onesided = model.results.pvalues_onesided.loc["lv"]

    exp_coef = 0.14024604432028576
    exp_coef_se = 0.12772108665506693
    exp_tvalue = 1.0980649162424108
    exp_pval_twosided = 0.2722171958691433
    exp_pval_onesided = 0.13610859793457164

    # check
    assert obs_coef is not None
    assert isinstance(obs_coef, float)
    assert obs_coef == pytest.approx(exp_coef, rel=1e-2)
    assert obs_coef_se == pytest.approx(exp_coef_se, rel=1e-2)
    assert obs_tvalue == pytest.approx(exp_tvalue, rel=1e-2)
    assert obs_pval_twosided == pytest.approx(exp_pval_twosided, rel=2e-2)
    assert obs_pval_onesided == pytest.approx(exp_pval_onesided, rel=2e-2)


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
