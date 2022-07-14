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
# (Please, take a look at the README.md file in this directory for instructions on how to run this notebook)
#
# This notebook reads all gene correlations across all chromosomes and computes a single correlation matrix by assembling a big correlation matrix with all genes.

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
import sys

import numpy as np
from scipy.spatial.distance import squareform
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

import conf
import utils
from entity import Gene

# %% [markdown] tags=[]
# # Settings

# %% tags=["parameters"]
# reference panel
REFERENCE_PANEL = "GTEX_V8"
# REFERENCE_PANEL = "1000G"

# prediction models
## mashr
EQTL_MODEL = "MASHR"
EQTL_MODEL_FILES_PREFIX = "mashr_"

# ## elastic net
# EQTL_MODEL = "ELASTIC_NET"
# EQTL_MODEL_FILES_PREFIX = "en_"

# make it read the prefix from conf.py
EQTL_MODEL_FILES_PREFIX = None

# %% tags=["injected-parameters"]
# Parameters
REFERENCE_PANEL = "1000G"
EQTL_MODEL = "MASHR"


# %% tags=[]
if EQTL_MODEL_FILES_PREFIX is None:
    EQTL_MODEL_FILES_PREFIX = conf.PHENOMEXCAN["PREDICTION_MODELS"][
        f"{EQTL_MODEL}_PREFIX"
    ]

# %% tags=[]
display(f"Using eQTL model: {EQTL_MODEL} / {EQTL_MODEL_FILES_PREFIX}")

# %% tags=[]
REFERENCE_PANEL_DIR = conf.PHENOMEXCAN["LD_BLOCKS"][f"{REFERENCE_PANEL}_GENOTYPE_DIR"]

# %% tags=[]
display(f"Using reference panel folder: {str(REFERENCE_PANEL_DIR)}")

# %% tags=[]
OUTPUT_DIR_BASE = (
    conf.PHENOMEXCAN["LD_BLOCKS"][f"GENE_CORRS_DIR"]
    / REFERENCE_PANEL.lower()
    / EQTL_MODEL.lower()
)
display(OUTPUT_DIR_BASE)
OUTPUT_DIR_BASE.mkdir(parents=True, exist_ok=True)

# %% tags=[]
display(f"Using output dir base: {OUTPUT_DIR_BASE}")

# %%
OUTPUT_DIR = utils.get_git_repository_path() / "tests" / "data" / "gls"
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

# %% tags=[]
output_file_name_template = conf.PHENOMEXCAN["LD_BLOCKS"][
    "GENE_CORRS_FILE_NAME_TEMPLATES"
]["GENE_CORR_AVG"]

# output_file = OUTPUT_DIR_BASE / "multiplier_genes-gene_correlations-gene_symbols.pkl"
output_file = (
    OUTPUT_DIR_BASE
    / "multiplier_genes-gene_correlations_within_distance-gene_symbols.pkl"
)
display(output_file)

# %%
full_corr_matrix_gene_symbols = pd.read_pickle(output_file)

# %% tags=[]
full_corr_matrix_gene_symbols.shape

# %% tags=[]
full_corr_matrix_gene_symbols.head()

# %% [markdown] tags=[]
# ### Make matrix compatible with GLS

# %%
_eigvals = np.linalg.eigvals(full_corr_matrix_gene_symbols)
display(_eigvals[_eigvals < 0].shape[0])
display(_eigvals[_eigvals < 0])

# %%
try:
    np.linalg.cholesky(full_corr_matrix_gene_symbols)
    print("No need to fix")
except Exception as e:
    print(f"Failed with:\n {str(e)}")

# %%
orig_corr_mat = full_corr_matrix_gene_symbols

# %%
orig_corr_mat.to_pickle(OUTPUT_DIR / "corr_mat.pkl.xz")

# %%
# # %load_ext rpy2.ipython

# %%
# corr_mat_r = full_corr_matrix_gene_symbols.to_numpy()

# %%
# # %Rpush corr_mat_r

# %%
# # %%R
# # taken from https://www.r-bloggers.com/2013/08/correcting-a-pseudo-correlation-matrix-to-be-positive-semidefinite/
# CorrectCM <- function(CM, p = 0) {
#   n <- dim(CM)[1L]
#   E <- eigen(CM)
#   CM1 <- E$vectors %*% tcrossprod(diag(pmax(E$values, p), n), E$vectors)
#   Balance <- diag(1 / sqrt(diag(CM1)))
#   CM2 <- Balance %*% CM1 %*% Balance
#   return(CM2)
# }

# %%
# # %%R -o corr_mat_r_fixed
# corr_mat_r_fixed <- CorrectCM(corr_mat_r, 1e-5)

# %%
# corr_mat_r_fixed

# %%
# corr_mat_r_fixed = pd.DataFrame(
#     corr_mat_r_fixed,
#     index=full_corr_matrix_gene_symbols.index.tolist(),
#     columns=full_corr_matrix_gene_symbols.columns.tolist()
# )

# %%
# corr_mat_r_fixed.shape

# %%
# corr_mat_r_fixed.head()

# %%
# corr_mat_r_fixed.equals(full_corr_matrix_gene_symbols)

# %%
# del full_corr_matrix_gene_symbols

# %%
# orig_corr_mat = corr_mat_r_fixed

# %%
# orig_corr_mat.to_pickle(OUTPUT_DIR / "corr_mat_fixed.pkl.xz")

# %% [markdown] tags=[]
# # Functions

# %% tags=[]
import statsmodels.api as sm
from sklearn.preprocessing import scale


# %%
def get_data(lv_code, random_phenotype_code=None, real_phenotype_code=None):
    if random_phenotype_code is not None:
        target_data = load_multixcan_random_phenotype(random_phenotype_code)["pvalue"]
        y = pd.Series(
            data=np.abs(stats.norm.ppf(target_data.to_numpy() / 2)),
            index=target_data.index.copy(),
        )
    elif real_phenotype_code is not None:
        y = multixcan_real_results[real_phenotype_code]

    y = y[~y.index.duplicated(keep="first")]
    y = y.dropna()

    common_genes = orig_corr_mat.index.intersection(y.index)
    y = y.loc[common_genes]

    X = multiplier_z[lv_code].copy()
    X = X.loc[common_genes]
    # X = (X - X.mean()) / X.std()
    X = sm.add_constant(X)

    return X, y


# %%
def standardize_data(X, y):
    X = X.copy()
    y = y.copy()

    c = [c for c in X.columns if c != "const"]
    X[c] = (X[c] - X[c].mean()) / X[c].std()

    return X, (y - y.mean()) / y.std()


# %%
def get_aligned_corr_mat(X, perc=1.0):
    # perc == 1.0 means select all nonzero genes
    gene_corrs = orig_corr_mat.loc[X.index, X.index]

    corr_mat_sub = pd.DataFrame(
        np.identity(gene_corrs.shape[0]),
        index=gene_corrs.index.copy(),
        columns=gene_corrs.columns.copy(),
    )

    X = X.iloc[:, 1]

    X_non_zero = X[X > 0]
    X_thres = X_non_zero.quantile(1 - perc)
    lv_nonzero_genes = X[X >= X_thres].index

    lv_nonzero_genes = lv_nonzero_genes.intersection(gene_corrs.index)
    corr_mat_sub.loc[lv_nonzero_genes, lv_nonzero_genes] = gene_corrs.loc[
        lv_nonzero_genes, lv_nonzero_genes
    ]

    return corr_mat_sub


# %%
# testing

# %%
def train_statsmodels_gls(X, y, corr_mat):
    gls_model = sm.GLS(y, X, sigma=corr_mat)
    gls_results = gls_model.fit()
    return gls_results


# %% [markdown] tags=[]
# # Make sure statsmodels (Python) and gls from R give the same results

# %% [markdown]
# ## Random phenotype 0

# %%
# lv_code = "LV1"

# %%
# X, y = get_data(lv_code, random_phenotype_code=1, transformation="log10")

# %%
# X.shape

# %%
# y.shape

# %%
# corr_mat = get_aligned_corr_mat(X)

# %%
# corr_mat.shape

# %% [markdown]
# ## statsmodels.GLS

# %%
# _gls_results = train_statsmodels_gls(X, y, corr_mat)

# %%
# print(_gls_results.summary())

# %%
# # print full numbers
# with np.printoptions(threshold=sys.maxsize, precision=20):
#     print(_gls_results.params.to_numpy())
#     print(_gls_results.bse.to_numpy())
#     print(_gls_results.tvalues.to_numpy())
#     print(_gls_results.pvalues.to_numpy())

# %% [markdown]
# ## R gls

# %%
# training_data = pd.concat([X, y], axis=1)

# %%
# training_data

# %%
# # %load_ext rpy2.ipython

# %%
# corr_mat_r = corr_mat.to_numpy()

# %%
# # %Rpush corr_mat_r

# %%
# # %%R -i training_data
# library(nlme)

# C <- corSymm(corr_mat_r[lower.tri(corr_mat_r)], fixed = T)

# g <- gls(pvalue ~ LV1, correlation=C, data=training_data)

# %%
# # %%R
# summary(g)$tTable

# %%
# # %%R -o r_gls_results
# r_gls_results <- summary(g)$tTable

# %%
# r_gls_results_df = pd.DataFrame(r_gls_results, index=["(Intercept)", lv_code], columns=["Value", "Std.Error", "t-value", "p-value"])

# %%
# r_gls_results_df

# %%
# assert np.allclose(r_gls_results_df["Value"].to_numpy().flatten(), _gls_results.params.values, atol=0.0, rtol=1e-5)

# %%
# assert np.allclose(r_gls_results_df["Std.Error"].to_numpy().flatten(), _gls_results.bse.values, atol=0.0, rtol=1e-5)

# %%
# assert np.allclose(r_gls_results_df["t-value"].to_numpy().flatten(), _gls_results.tvalues, atol=0.0, rtol=1e-5)

# %%
# assert np.allclose(r_gls_results_df["p-value"].to_numpy().flatten(), _gls_results.pvalues, atol=0.0, rtol=1e-5)

# %% [markdown] tags=[]
# # GLS on randomly generated phenotypes

# %%
PERC_NONZERO_GENES = 1.00

# %% [markdown]
# ## Random phenotype 6 / LV45

# %%
lv_code = "LV45"
phenotype_code = 6

phenotype_name = f"multixcan-random_phenotype{phenotype_code}-pvalues"
display(phenotype_name)

# %%
X, y = get_data(lv_code, random_phenotype_code=phenotype_code)
corr_mat = get_aligned_corr_mat(X, PERC_NONZERO_GENES)

Xs, ys = standardize_data(X, y)
_gls_results = train_statsmodels_gls(Xs, ys, corr_mat)

# %%
print(_gls_results.summary())

# %%
# print full numbers
with np.printoptions(threshold=sys.maxsize, precision=20):
    print(_gls_results.params.to_numpy()[1])
    print(_gls_results.bse.to_numpy()[1])
    print(_gls_results.tvalues.to_numpy()[1])
    print(_gls_results.pvalues.to_numpy()[1])

# %%
# save phenotype
y.to_pickle(OUTPUT_DIR / f"{phenotype_name}.pkl.xz")

# %%
y

# %% [markdown]
# ## Random phenotype 10 / LV10

# %%
lv_code = "LV100"
phenotype_code = 10

phenotype_name = f"multixcan-random_phenotype{phenotype_code}-pvalues"
display(phenotype_name)

# %%
X, y = get_data(lv_code, random_phenotype_code=phenotype_code)
corr_mat = get_aligned_corr_mat(X, PERC_NONZERO_GENES)

Xs, ys = standardize_data(X, y)
_gls_results = train_statsmodels_gls(Xs, ys, corr_mat)

# %%
print(_gls_results.summary())

# %%
# print full numbers
with np.printoptions(threshold=sys.maxsize, precision=20):
    print(_gls_results.params.to_numpy()[1])
    print(_gls_results.bse.to_numpy()[1])
    print(_gls_results.tvalues.to_numpy()[1])
    print(_gls_results.pvalues.to_numpy()[1])

# %%
# save phenotype
y.to_pickle(OUTPUT_DIR / f"{phenotype_name}.pkl.xz")

# %%
y

# %% [markdown]
# ## Random phenotype 0 / LV800

# %%
lv_code = "LV800"
phenotype_code = 0

phenotype_name = f"multixcan-random_phenotype{phenotype_code}-pvalues"
display(phenotype_name)

# %%
X, y = get_data(lv_code, random_phenotype_code=phenotype_code)
corr_mat = get_aligned_corr_mat(X, PERC_NONZERO_GENES)

Xs, ys = standardize_data(X, y)
_gls_results = train_statsmodels_gls(Xs, ys, corr_mat)

# %%
print(_gls_results.summary())

# %%
# print full numbers
with np.printoptions(threshold=sys.maxsize, precision=20):
    print(_gls_results.params.to_numpy()[1])
    print(_gls_results.bse.to_numpy()[1])
    print(_gls_results.tvalues.to_numpy()[1])
    print(_gls_results.pvalues.to_numpy()[1])

# %%
# save phenotype
y.to_pickle(OUTPUT_DIR / f"{phenotype_name}.pkl.xz")

# %%
y

# %% [markdown] tags=[]
# # GLS on real phenotypes

# %%
multixcan_real_results.columns

# %% [markdown]
# ## whooping cough / LV570

# %%
lv_code = "LV570"
phenotype_code = "whooping cough"

phenotype_name = f"multixcan-phenomexcan-{phenotype_code.replace(' ', '_')}-pvalues"
display(phenotype_name)

# %%
X, y = get_data(lv_code, real_phenotype_code=phenotype_code)
corr_mat = get_aligned_corr_mat(X, PERC_NONZERO_GENES)

Xs, ys = standardize_data(X, y)
_gls_results = train_statsmodels_gls(Xs, ys, corr_mat)

# %%
print(_gls_results.summary())

# %%
# print full numbers
with np.printoptions(threshold=sys.maxsize, precision=20):
    print(_gls_results.params.to_numpy()[1])
    print(_gls_results.bse.to_numpy()[1])
    print(_gls_results.tvalues.to_numpy()[1])
    print(_gls_results.pvalues.to_numpy()[1])

# %%
# save phenotype
y.to_pickle(OUTPUT_DIR / f"{phenotype_name}.pkl.xz")

# %%
y

# %% [markdown]
# ## wheezing and LV400

# %%
lv_code = "LV400"
phenotype_code = "wheezing"

phenotype_name = f"multixcan-phenomexcan-{phenotype_code.replace(' ', '_')}-pvalues"
display(phenotype_name)

# %%
X, y = get_data(lv_code, real_phenotype_code=phenotype_code)
corr_mat = get_aligned_corr_mat(X, PERC_NONZERO_GENES)

Xs, ys = standardize_data(X, y)
_gls_results = train_statsmodels_gls(Xs, ys, corr_mat)

# %%
print(_gls_results.summary())

# %%
# print full numbers
with np.printoptions(threshold=sys.maxsize, precision=20):
    print(_gls_results.params.to_numpy()[1])
    print(_gls_results.bse.to_numpy()[1])
    print(_gls_results.tvalues.to_numpy()[1])
    print(_gls_results.pvalues.to_numpy()[1])

# %%
# save phenotype
y.to_pickle(OUTPUT_DIR / f"{phenotype_name}.pkl.xz")

# %%
y

# %%
