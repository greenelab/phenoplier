# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
#     formats: ipynb,py//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] tags=[]
# # Description

# %% [markdown] tags=[]
# It generates tests cases for the GLS model.

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
import sys

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

import conf
import utils
from entity import Gene

# %% [markdown] tags=[]
# # Settings

# %% tags=["parameters"]
# a cohort name (it could be something like UK_BIOBANK, etc)
COHORT_NAME = "1000G_EUR"

# reference panel such as 1000G or GTEX_V8
REFERENCE_PANEL = "1000G"

# predictions models such as MASHR or ELASTIC_NET
EQTL_MODEL = "MASHR"

# %%
OUTPUT_DIR = utils.get_git_repository_path() / "tests" / "data" / "gls"
display(OUTPUT_DIR)
assert OUTPUT_DIR.exists()

# %% [markdown] tags=[]
# # Load data

# %% [markdown] tags=[]
# ## MultiPLIER Z

# %% tags=[]
multiplier_z = pd.read_pickle(conf.MULTIPLIER["MODEL_Z_MATRIX_FILE"])

# %% tags=[]
multiplier_z_genes = multiplier_z.index.tolist()

# %% tags=[]
len(multiplier_z_genes)

# %% tags=[]
multiplier_z_genes[:10]


# %% [markdown]
# ## Function to load MultiXcan's results on random phenotypes

# %%
def load_multixcan_random_phenotype(phenotype_code):
    multixcan_random_results = pd.read_csv(
        conf.RESULTS["GLS_NULL_SIMS"]
        / "twas"
        / "smultixcan"
        / f"random.pheno{phenotype_code}-gtex_v8-mashr-smultixcan.txt",
        sep="\t",
        index_col="gene_name",
    )

    return multixcan_random_results


# %%
load_multixcan_random_phenotype(0).head()

# %% [markdown] tags=[]
# ## MultiXcan real results (PhenomeXcan)

# %%
multixcan_real_results = pd.read_pickle(
    conf.PHENOMEXCAN["SMULTIXCAN_EFO_PARTIAL_MASHR_ZSCORES_FILE"]
).rename(index=Gene.GENE_ID_TO_NAME_MAP)

# %%
multixcan_real_results = multixcan_real_results[
    ~multixcan_real_results.index.duplicated(keep="first")
].dropna(how="all", axis=0)

# %%
multixcan_real_results.shape

# %%
multixcan_real_results.head()

# %%
assert not multixcan_real_results.isna().any(None)

# %% [markdown] tags=[]
# ## Load full correlation matrix

# %%
orig_corr_mat = pd.read_pickle(OUTPUT_DIR / "corr_mat.pkl.xz")

# %%
orig_corr_mat.shape

# %%
orig_corr_mat.head()

# %% [markdown] tags=[]
# ## Load cohort metadata

# %%
gene_tissues_df = pd.read_pickle(
    OUTPUT_DIR / "cohort_1000g_eur_metadata" / "gene_tissues.pkl.gz"
).set_index("gene_name")

# %%
gene_tissues_df.shape

# %%
gene_tissues_df = gene_tissues_df.loc[~gene_tissues_df.index.duplicated(keep="first")]

# %%
gene_tissues_df.shape

# %%
assert gene_tissues_df.index.is_unique

# %%
gene_tissues_df.head()


# %% [markdown] tags=[]
# # Functions

# %%
def get_data(
    lv_code,
    random_phenotype_code,
    add_covars=False,
    add_covars_logs=False,
):
    target_data = load_multixcan_random_phenotype(random_phenotype_code)["pvalue"]
    y = pd.Series(
        data=-np.log10(target_data.to_numpy()),
        index=target_data.index.copy(),
    )

    y = y[~y.index.duplicated(keep="first")]
    y = y.dropna()

    X = multiplier_z[lv_code].copy()

    common_genes = orig_corr_mat.index.intersection(y.index).intersection(X.index)
    y = y.loc[common_genes]
    X = X.loc[common_genes]

    # binarize
    x_perc = 0.01
    x_q = X.quantile(1.0 - x_perc)
    x_binarized = X.copy()
    # make sure top genes have nonzero weights
    x_cond = (x_binarized > 0.0) & (x_binarized >= x_q)
    x_binarized[x_cond] = 1.0
    x_binarized[~x_cond] = 0.0
    X = x_binarized

    X = sm.add_constant(X)

    if add_covars:
        covars = load_multixcan_random_phenotype(random_phenotype_code)[
            ["n", "n_indep"]
        ]
        covars = covars[~covars.index.duplicated(keep="first")]
        covars = covars.loc[X.index]

        # gene-level covariates (S-MultiXcan)
        covars = covars.rename(
            columns={
                "n_indep": "gene_size",
            }
        )
        covars = covars.assign(
            gene_density=covars.apply(lambda x: x["gene_size"] / x["n"], axis=1)
        )

        if add_covars_logs:
            covars["gene_size_log"] = np.log(covars["gene_size"])
            covars["gene_density_log"] = -np.log(covars["gene_density"])

        # if add_covars:
        covars = covars.drop(columns=[c for c in covars.columns if c in ("n",)])

        X = X.join(covars)

    return X, y


# %%
# testing
_X, _y = get_data("LV7", 10)
assert _X.shape[0] < 7000
assert _X.shape[1] == 2
assert "LV7" in _X.columns
assert "const" in _X.columns
assert not _X.isna().any(None)

assert _y.shape[0] == _X.shape[0]
assert not _y.isna().any(None)

x_summary = _X["LV7"].value_counts()
assert x_summary.shape[0] == 2, "Wrong binarization"
n_pos = int(x_summary.loc[1.0])
n_neg = int(x_summary.loc[0.0])
assert n_pos > 10
assert n_neg > 10
assert n_pos < n_neg

# %%
_X.head()

# %%
_y.head()

# %%
# testing
_X, _y = get_data("LV7", 10, add_covars=True)
assert _X.shape[0] < 7000
assert _X.shape[1] == 2 + 4
assert "LV7" in _X.columns
assert "const" in _X.columns
assert "gene_size" in _X.columns
assert "gene_density" in _X.columns
assert "gene_n_snps_used" in _X.columns
assert "gene_n_snps_used_sharing" in _X.columns
assert not _X.isna().any(None)

assert _y.shape[0] == _X.shape[0]
assert not _y.isna().any(None)

# %%
_X.head()

# %%
# load_multixcan_random_phenotype(10).loc["TNFRSF18"]
gene_tissues_df.loc["TNFRSF18"]

# %%
assert _X.loc["TNFRSF18", "gene_size"] == 4.0
assert _X.loc["TNFRSF18", "gene_density"] == 4 / 45.0
assert _X.loc["TNFRSF18", "gene_n_snps_used"] == 65
assert _X.loc["TNFRSF18", "gene_n_snps_used_sharing"] == 65 / 27.0

# %%
_y.head()

# %%
# testing
_X, _y = get_data("LV7", 10, add_covars=True, add_covars_logs=True)
assert _X.shape[0] < 7000
assert _X.shape[1] == 2 + 4 + 4
assert "LV7" in _X.columns
assert "const" in _X.columns
assert "gene_size" in _X.columns
assert "gene_size_log" in _X.columns
assert "gene_density" in _X.columns
assert "gene_density_log" in _X.columns
assert "gene_n_snps_used" in _X.columns
assert "gene_n_snps_used_log" in _X.columns
assert "gene_n_snps_used_sharing" in _X.columns
assert "gene_n_snps_used_sharing_log" in _X.columns
assert not _X.isna().any(None)

assert _X["gene_density"].between(0.0, 1.0, inclusive="right").all()
assert _X["gene_density_log"].min() >= 0.0
assert _X["gene_size"].min() >= 0.0
assert _X["gene_size_log"].min() >= 0.0
assert _X["gene_n_snps_used"].min() >= 0.0
assert _X["gene_n_snps_used_log"].min() >= 0.0
assert _X["gene_n_snps_used_sharing"].min() >= 0.0
assert _X["gene_n_snps_used_sharing_log"].min() >= 0.0

assert _y.shape[0] == _X.shape[0]
assert not _y.isna().any(None)

# %%
_X.head()

# %%
_X.describe()

# %%
_y.head()


# %%
def standardize_data(X, y):
    X = X.copy()
    y = y.copy()

    c = [c for c in X.columns if c != "const"]
    X[c] = (X[c] - X[c].mean()) / X[c].std()

    return X, (y - y.mean()) / y.std()


# %%
def get_aligned_corr_mat(X, perc=0.01):
    # perc == 1.0 means select all genes
    # perc == 0.01 means select top 1% of genes
    # perc = None means do not subset the correlation matrix
    gene_corrs = orig_corr_mat.loc[X.index, X.index]

    if perc is None:
        return gene_corrs

    corr_mat_sub = pd.DataFrame(
        np.eye(gene_corrs.shape[0]),
        index=gene_corrs.index.copy(),
        columns=gene_corrs.columns.copy(),
    )

    lv_col = X.columns[1]
    assert lv_col.startswith("LV")
    X = X = multiplier_z[lv_col].copy()

    # X_non_zero = X[X > 0]
    X_thres = X.quantile(1.0 - perc)
    lv_nonzero_genes = X[X >= X_thres].index

    lv_nonzero_genes = lv_nonzero_genes.intersection(gene_corrs.index)
    corr_mat_sub.loc[lv_nonzero_genes, lv_nonzero_genes] = gene_corrs.loc[
        lv_nonzero_genes, lv_nonzero_genes
    ]

    return corr_mat_sub


# %%
# testing
_X_test = pd.DataFrame(
    {
        "const": 1.0,
        "LV1": [1.0, 0.4, 0.0],  # the last gene has zero weight
    },
    index=[
        "PSMB10",  # the first two genes have a high sum of correlations, to make sure the sum is not close to 1.0
        "SLC12A4",
        "ACD",
    ],
)

# do not subset
_tmp_corr = get_aligned_corr_mat(_X_test, perc=None)
assert _tmp_corr.shape == (_X_test.shape[0], _X_test.shape[0])
assert np.array_equal(
    _tmp_corr.round(2).to_numpy(),
    np.array(
        [
            [1.0, 0.77, 0.73],
            [0.77, 1.0, 0.63],
            [0.73, 0.63, 1.00],
        ]
    ),
)

# do subset: include all non-zero LV genes
_tmp_corr = get_aligned_corr_mat(_X_test, perc=1.0)
assert _tmp_corr.shape == (_X_test.shape[0], _X_test.shape[0])
assert np.array_equal(
    _tmp_corr.round(2).to_numpy(),
    np.array(
        [
            [1.0, 0.77, 0.73],
            [0.77, 1.0, 0.63],
            [0.73, 0.63, 1.00],
        ]
    ),
)

# do subset: but perc is so low that it doesn't select  any genes
_tmp_corr = get_aligned_corr_mat(_X_test, perc=0.001)
assert _tmp_corr.shape == (_X_test.shape[0], _X_test.shape[0])
assert np.array_equal(
    _tmp_corr.round(2).to_numpy(),
    np.array(
        [
            [1.0, 0.00, 0.00],
            [0.00, 1.0, 0.00],
            [0.00, 0.00, 1.00],
        ]
    ),
)


# %%
def train_statsmodels_gls(X, y, corr_mat):
    gls_model = sm.GLS(y, X, sigma=corr_mat)
    gls_results = gls_model.fit()
    return gls_results


# %% [markdown] tags=[]
# # [full corr matrix] GLS on randomly generated phenotypes

# %%
PERC_NONZERO_GENES = None

# %% [markdown]
# ## Random phenotype 6 / LV45

# %%
lv_code = "LV45"
phenotype_code = 6

phenotype_name = f"multixcan-random_phenotype{phenotype_code}-pvalues"
display(phenotype_name)

# %%
X, y = get_data(lv_code, random_phenotype_code=phenotype_code)
corr_mat = get_aligned_corr_mat(X, perc=PERC_NONZERO_GENES)

# %%
y

# %%
_, ys = standardize_data(X, y)
_gls_results = train_statsmodels_gls(X, ys, corr_mat)
print(_gls_results.summary())

# %%
# print full numbers
with np.printoptions(threshold=sys.maxsize, precision=20):
    print(
        f"""
exp_coef = {_gls_results.params.to_numpy()[1]}
exp_coef_se = {_gls_results.bse.to_numpy()[1]}
exp_tvalue = {_gls_results.tvalues.to_numpy()[1]}
exp_pval_twosided = {_gls_results.pvalues.to_numpy()[1]}
exp_pval_onesided = {stats.t.sf(_gls_results.tvalues.to_numpy()[1], _gls_results.df_resid)}
    """
    )

# %%
X.sort_values(lv_code, ascending=False)

# %%
y.sort_values(ascending=False)

# %%
ys.sort_values(ascending=False)

# %%
# save phenotype
y.to_pickle(OUTPUT_DIR / f"{phenotype_name}.pkl.xz")

# %%
# for debugging purposes I print the OLS results also
_tmp_model = sm.OLS(ys, Xs)
_tmp_results = _tmp_model.fit()
print(_tmp_results.summary())

# %% [markdown]
# ## Random phenotype 6 / LV455

# %%
lv_code = "LV455"
phenotype_code = 6

phenotype_name = f"multixcan-random_phenotype{phenotype_code}-pvalues"
display(phenotype_name)

# %%
X, y = get_data(lv_code, random_phenotype_code=phenotype_code)
corr_mat = get_aligned_corr_mat(X, perc=PERC_NONZERO_GENES)

# %%
y

# %%
_, ys = standardize_data(X, y)
_gls_results = train_statsmodels_gls(X, ys, corr_mat)
print(_gls_results.summary())

# %%
# print full numbers
with np.printoptions(threshold=sys.maxsize, precision=20):
    print(
        f"""
exp_coef = {_gls_results.params.to_numpy()[1]}
exp_coef_se = {_gls_results.bse.to_numpy()[1]}
exp_tvalue = {_gls_results.tvalues.to_numpy()[1]}
exp_pval_twosided = {_gls_results.pvalues.to_numpy()[1]}
exp_pval_onesided = {stats.t.sf(_gls_results.tvalues.to_numpy()[1], _gls_results.df_resid)}
    """
    )

# %%
X.sort_values(lv_code, ascending=False)

# %%
y.sort_values(ascending=False)

# %%
ys.sort_values(ascending=False)

# %%
# save phenotype
y.to_pickle(OUTPUT_DIR / f"{phenotype_name}.pkl.xz")

# %%
# for debugging purposes I print the OLS results also
_tmp_model = sm.OLS(ys, Xs)
_tmp_results = _tmp_model.fit()
print(_tmp_results.summary())

# %% [markdown]
# ## Random phenotype 0 / LV801

# %%
lv_code = "LV801"
phenotype_code = 0

phenotype_name = f"multixcan-random_phenotype{phenotype_code}-pvalues"
display(phenotype_name)

# %%
X, y = get_data(lv_code, random_phenotype_code=phenotype_code)
corr_mat = get_aligned_corr_mat(X, perc=PERC_NONZERO_GENES)

# %%
y

# %%
_, ys = standardize_data(X, y)
_gls_results = train_statsmodels_gls(X, ys, corr_mat)
print(_gls_results.summary())

# %%
# print full numbers
with np.printoptions(threshold=sys.maxsize, precision=20):
    print(
        f"""
exp_coef = {_gls_results.params.to_numpy()[1]}
exp_coef_se = {_gls_results.bse.to_numpy()[1]}
exp_tvalue = {_gls_results.tvalues.to_numpy()[1]}
exp_pval_twosided = {_gls_results.pvalues.to_numpy()[1]}
exp_pval_onesided = {stats.t.sf(_gls_results.tvalues.to_numpy()[1], _gls_results.df_resid)}
    """
    )

# %%
X.sort_values(lv_code, ascending=False)

# %%
y.sort_values(ascending=False)

# %%
ys.sort_values(ascending=False)

# %%
# save phenotype
y.to_pickle(OUTPUT_DIR / f"{phenotype_name}.pkl.xz")

# %%
# for debugging purposes I print the OLS results also
_tmp_model = sm.OLS(ys, Xs)
_tmp_results = _tmp_model.fit()
print(_tmp_results.summary())

# %% [markdown] tags=[]
# # [sub corr matrix ] GLS on randomly generated phenotypes

# %%
PERC_NONZERO_GENES = 0.01

# %% [markdown]
# ## Random phenotype 6 / LV45

# %%
lv_code = "LV45"
phenotype_code = 6

phenotype_name = f"multixcan-random_phenotype{phenotype_code}-pvalues"
display(phenotype_name)

# %%
X, y = get_data(lv_code, random_phenotype_code=phenotype_code)
corr_mat = get_aligned_corr_mat(X, perc=PERC_NONZERO_GENES)

# %%
_, ys = standardize_data(X, y)
_gls_results = train_statsmodels_gls(X, ys, corr_mat)
print(_gls_results.summary())

# %%
# print full numbers
with np.printoptions(threshold=sys.maxsize, precision=20):
    print(
        f"""
exp_coef = {_gls_results.params.to_numpy()[1]}
exp_coef_se = {_gls_results.bse.to_numpy()[1]}
exp_tvalue = {_gls_results.tvalues.to_numpy()[1]}
exp_pval_twosided = {_gls_results.pvalues.to_numpy()[1]}
exp_pval_onesided = {stats.t.sf(_gls_results.tvalues.to_numpy()[1], _gls_results.df_resid)}
    """
    )

# %%
X.sort_values(lv_code, ascending=False)

# %%
y.sort_values(ascending=False)

# %%
ys.sort_values(ascending=False)

# %%
# save phenotype
y.to_pickle(OUTPUT_DIR / f"{phenotype_name}.pkl.xz")

# %% [markdown]
# ## Random phenotype 6 / LV455

# %%
lv_code = "LV455"
phenotype_code = 6

phenotype_name = f"multixcan-random_phenotype{phenotype_code}-pvalues"
display(phenotype_name)

# %%
X, y = get_data(lv_code, random_phenotype_code=phenotype_code)
corr_mat = get_aligned_corr_mat(X, perc=PERC_NONZERO_GENES)

# %%
_, ys = standardize_data(X, y)
_gls_results = train_statsmodels_gls(X, ys, corr_mat)
print(_gls_results.summary())

# %%
# print full numbers
with np.printoptions(threshold=sys.maxsize, precision=20):
    print(
        f"""
exp_coef = {_gls_results.params.to_numpy()[1]}
exp_coef_se = {_gls_results.bse.to_numpy()[1]}
exp_tvalue = {_gls_results.tvalues.to_numpy()[1]}
exp_pval_twosided = {_gls_results.pvalues.to_numpy()[1]}
exp_pval_onesided = {stats.t.sf(_gls_results.tvalues.to_numpy()[1], _gls_results.df_resid)}
    """
    )

# %%
X.sort_values(lv_code, ascending=False)

# %%
y.sort_values(ascending=False)

# %%
ys.sort_values(ascending=False)

# %%
# save phenotype
y.to_pickle(OUTPUT_DIR / f"{phenotype_name}.pkl.xz")

# %% [markdown]
# ## Random phenotype 10 / LV100

# %%
lv_code = "LV100"
phenotype_code = 10

phenotype_name = f"multixcan-random_phenotype{phenotype_code}-pvalues"
display(phenotype_name)

# %%
X, y = get_data(lv_code, random_phenotype_code=phenotype_code)
corr_mat = get_aligned_corr_mat(X, perc=PERC_NONZERO_GENES)

# %%
_, ys = standardize_data(X, y)
_gls_results = train_statsmodels_gls(X, ys, corr_mat)
print(_gls_results.summary())

# %%
# print full numbers
with np.printoptions(threshold=sys.maxsize, precision=20):
    print(
        f"""
exp_coef = {_gls_results.params.to_numpy()[1]}
exp_coef_se = {_gls_results.bse.to_numpy()[1]}
exp_tvalue = {_gls_results.tvalues.to_numpy()[1]}
exp_pval_twosided = {_gls_results.pvalues.to_numpy()[1]}
exp_pval_onesided = {stats.t.sf(_gls_results.tvalues.to_numpy()[1], _gls_results.df_resid)}
    """
    )

# %%
X.sort_values(lv_code, ascending=False)

# %%
y.sort_values(ascending=False)

# %%
ys.sort_values(ascending=False)

# %%
# save phenotype
y.to_pickle(OUTPUT_DIR / f"{phenotype_name}.pkl.xz")

# %% [markdown]
# ## Random phenotype 0 / LV800

# %%
lv_code = "LV800"
phenotype_code = 0

phenotype_name = f"multixcan-random_phenotype{phenotype_code}-pvalues"
display(phenotype_name)

# %%
X, y = get_data(lv_code, random_phenotype_code=phenotype_code)
corr_mat = get_aligned_corr_mat(X, perc=PERC_NONZERO_GENES)

# %%
_, ys = standardize_data(X, y)
_gls_results = train_statsmodels_gls(X, ys, corr_mat)
print(_gls_results.summary())

# %%
# print full numbers
with np.printoptions(threshold=sys.maxsize, precision=20):
    print(
        f"""
exp_coef = {_gls_results.params.to_numpy()[1]}
exp_coef_se = {_gls_results.bse.to_numpy()[1]}
exp_tvalue = {_gls_results.tvalues.to_numpy()[1]}
exp_pval_twosided = {_gls_results.pvalues.to_numpy()[1]}
exp_pval_onesided = {stats.t.sf(_gls_results.tvalues.to_numpy()[1], _gls_results.df_resid)}
    """
    )

# %%
X.sort_values(lv_code, ascending=False)

# %%
y.sort_values(ascending=False)

# %%
ys.sort_values(ascending=False)

# %%
# save phenotype
y.to_pickle(OUTPUT_DIR / f"{phenotype_name}.pkl.xz")

# %% [markdown]
# # Test different covariates

# %%
covars = load_multixcan_random_phenotype(0)[["n", "n_indep"]]
covars = covars[~covars.index.duplicated(keep="first")]
covars = covars.dropna()
covars = covars.join(gene_tissues_df, how="inner")
assert not covars.isna().any(None)

# %%
covars = covars.drop(columns=[c for c in covars.columns if "model" in c])

# %%
covars.shape

# %%
covars.head()

# %%
# gene_size and gene_density
covars = covars.rename(
    columns={
        "n_indep": "gene_size",
    }
)
covars = covars.assign(
    gene_density=covars.apply(lambda x: x["gene_size"] / x["n"], axis=1)
)
covars = covars.drop(columns=["n"])

# %%
_final_covars = [
    "gene_size",
    "gene_density",
]

# %%
covars[_final_covars].head()

# %%
covars[_final_covars].describe()

# %%
_tmp = covars.assign(**{f"{c}_log": np.log(covars[c]) for c in _final_covars})
display(_tmp[[c for c in _tmp.columns if "_log" in c]].describe())

# %%
covars[_final_covars].corr()

# %%
_tmp[[c for c in _tmp.columns if "_log" in c]].corr()

# %% [markdown] tags=[]
# ## [full corr matrix] GLS on randomly generated phenotypes using gene-level covariates

# %%
PERC_NONZERO_GENES = None

# %% [markdown]
# ### Random phenotype 6 / LV45

# %%
lv_code = "LV45"
phenotype_code = 6

phenotype_name_base = f"multixcan-random_phenotype{phenotype_code}"
phenotype_name = f"{phenotype_name_base}-pvalues"
display(phenotype_name)

# %%
X, y = get_data(lv_code, random_phenotype_code=phenotype_code, add_covars=True)
corr_mat = get_aligned_corr_mat(X, perc=PERC_NONZERO_GENES)

# %%
X.head()

# %%
_, ys = standardize_data(X, y)
_gls_results = train_statsmodels_gls(X, ys, corr_mat)
print(_gls_results.summary())

# %%
# print full numbers
with np.printoptions(threshold=sys.maxsize, precision=20):
    print(
        f"""
exp_coef = {_gls_results.params.to_numpy()[1]}
exp_coef_se = {_gls_results.bse.to_numpy()[1]}
exp_tvalue = {_gls_results.tvalues.to_numpy()[1]}
exp_pval_twosided = {_gls_results.pvalues.to_numpy()[1]}
exp_pval_onesided = {stats.t.sf(_gls_results.tvalues.to_numpy()[1], _gls_results.df_resid)}
    """
    )

# %%
X.sort_values(lv_code, ascending=False)

# %%
y.sort_values(ascending=False)

# %%
ys.sort_values(ascending=False)

# %%
# save phenotype
y.to_pickle(OUTPUT_DIR / f"{phenotype_name}.pkl.xz")

# %%
# for debugging purposes I print the OLS results also
_tmp_model = sm.OLS(ys, Xs)
_tmp_results = _tmp_model.fit()
print(_tmp_results.summary())

# %% [markdown] tags=[]
# ## [sub corr matrix] GLS on randomly generated phenotypes using SNP-level covariates

# %%
PERC_NONZERO_GENES = 0.01

# %% [markdown]
# ### Random phenotype 6 / LV45

# %%
lv_code = "LV45"
phenotype_code = 6

phenotype_name_base = f"multixcan-random_phenotype{phenotype_code}"
phenotype_name = f"{phenotype_name_base}-pvalues"
display(phenotype_name)

# %%
X, y = get_data(lv_code, random_phenotype_code=phenotype_code, add_covars=True)
corr_mat = get_aligned_corr_mat(X, perc=PERC_NONZERO_GENES)

# %%
_, ys = standardize_data(X, y)
_gls_results = train_statsmodels_gls(X, ys, corr_mat)
print(_gls_results.summary())

# %%
# print full numbers
with np.printoptions(threshold=sys.maxsize, precision=20):
    print(
        f"""
exp_coef = {_gls_results.params.to_numpy()[1]}
exp_coef_se = {_gls_results.bse.to_numpy()[1]}
exp_tvalue = {_gls_results.tvalues.to_numpy()[1]}
exp_pval_twosided = {_gls_results.pvalues.to_numpy()[1]}
exp_pval_onesided = {stats.t.sf(_gls_results.tvalues.to_numpy()[1], _gls_results.df_resid)}
    """
    )

# %%
X.sort_values(lv_code, ascending=False)

# %%
y.sort_values(ascending=False)

# %%
ys.sort_values(ascending=False)

# %%
# save phenotype
y.to_pickle(OUTPUT_DIR / f"{phenotype_name}.pkl.xz")

# %%
