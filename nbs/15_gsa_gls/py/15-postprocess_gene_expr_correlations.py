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
import pickle

import numpy as np
from scipy.spatial.distance import squareform
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import conf
from entity import Gene

# %% [markdown] tags=[]
# # Settings

# %% tags=["parameters"]
# a cohort name (it could be something like UK_BIOBANK, etc)
COHORT_NAME = None

# reference panel such as 1000G or GTEX_V8
REFERENCE_PANEL = None

# predictions models such as MASHR or ELASTIC_NET
EQTL_MODEL = None

# %%
assert COHORT_NAME is not None and len(COHORT_NAME) > 0, "A cohort name must be given"

COHORT_NAME = COHORT_NAME.lower()
display(f"Cohort name: {COHORT_NAME}")

# %%
assert (
    REFERENCE_PANEL is not None and len(REFERENCE_PANEL) > 0
), "A reference panel must be given"

display(f"Reference panel: {REFERENCE_PANEL}")

# %%
assert (
    EQTL_MODEL is not None and len(EQTL_MODEL) > 0
), "A prediction/eQTL model must be given"

EQTL_MODEL_FILES_PREFIX = conf.PHENOMEXCAN["PREDICTION_MODELS"][f"{EQTL_MODEL}_PREFIX"]
display(f"eQTL model: {EQTL_MODEL}) / {EQTL_MODEL_FILES_PREFIX}")

# %%
OUTPUT_DIR_BASE = (
    conf.RESULTS["GLS"]
    / "gene_corrs"
    / "cohorts"
    / COHORT_NAME.lower()
    / REFERENCE_PANEL.lower()
    / EQTL_MODEL.lower()
)
OUTPUT_DIR_BASE.mkdir(parents=True, exist_ok=True)

display(f"Using output dir base: {OUTPUT_DIR_BASE}")

# %%
COHORT_INPUT_DIR = conf.RESULTS["GLS"] / "gene_corrs" / "cohorts" / COHORT_NAME

display(f"Cohort input dir: {COHORT_INPUT_DIR}")
assert COHORT_INPUT_DIR.exists()

# %%
INPUT_DIR = OUTPUT_DIR_BASE / "by_chr"

display(f"Gene correlations input dir: {INPUT_DIR}")
assert INPUT_DIR.exists()

# %% [markdown] tags=[]
# # Load data

# %% [markdown] tags=[]
# ## Gene correlations

# %% tags=[]
all_gene_corr_files = list(INPUT_DIR.glob("gene_corrs-chr*.pkl"))

# %%
# sort by chromosome
all_gene_corr_files = sorted(
    all_gene_corr_files, key=lambda x: int(x.name.split("-chr")[1].split(".pkl")[0])
)

# %% tags=[]
len(all_gene_corr_files)

# %% tags=[]
all_gene_corr_files

# %% tags=[]
assert len(all_gene_corr_files) == 22

# %% [markdown] tags=[]
# ## Get common genes

# %%
gene_ids = set()
for f in all_gene_corr_files:
    chr_genes = pd.read_pickle(f).index.tolist()
    gene_ids.update(chr_genes)

# %%
display(len(gene_ids))

# %%
sorted(list(gene_ids))[:5]

# %% [markdown] tags=[]
# ## Gene info

# %%
genes_info = pd.read_pickle(COHORT_INPUT_DIR / "genes_info.pkl")

# %%
genes_info.shape

# %%
genes_info.head()

# %%
# keep genes in correlation matrices only
genes_info = genes_info[genes_info["id"].isin(gene_ids)]

# %%
genes_info.shape

# %%
assert not genes_info.isna().any().any()

# %%
genes_info.dtypes

# %%
genes_info["chr"] = genes_info["chr"].apply(pd.to_numeric, downcast="integer")
genes_info["start_position"] = genes_info["start_position"].astype(int)
genes_info["end_position"] = genes_info["end_position"].astype(int)
genes_info["gene_length"] = genes_info["gene_length"].astype(int)

# %%
genes_info.dtypes

# %%
assert not genes_info.isna().any(None)

# %%
genes_info.head()

# %% [markdown] tags=[]
# # Create full correlation matrix

# %%
genes_info = genes_info.sort_values(["chr", "start_position"])

# %%
genes_info

# %%
full_corr_matrix = pd.DataFrame(
    np.zeros((genes_info.shape[0], genes_info.shape[0])),
    index=genes_info["id"].tolist(),
    columns=genes_info["id"].tolist(),
)

# %%
assert full_corr_matrix.index.is_unique & full_corr_matrix.columns.is_unique

# %%
for chr_corr_file in all_gene_corr_files:
    print(chr_corr_file.name, flush=True)

    corr_data = pd.read_pickle(chr_corr_file)
    full_corr_matrix.loc[corr_data.index, corr_data.columns] = corr_data

# %%
full_corr_matrix.shape

# %%
full_corr_matrix.head()

# %%
np.all(full_corr_matrix.to_numpy().diagonal() == 1.0)

# %% [markdown] tags=[]
# ## Some checks

# %%
assert not full_corr_matrix.isna().any(None)
assert not np.isinf(full_corr_matrix.to_numpy()).any()
assert not np.iscomplex(full_corr_matrix.to_numpy()).any()

# %%
_min_val = full_corr_matrix.min().min()
display(_min_val)
assert _min_val >= 0.0

# %%
_max_val = full_corr_matrix.max().max()  # this will capture the 1.0 in the diagonal
display(_max_val)
assert _max_val <= 1.00

# %% [markdown]
# # Positive definiteness

# %%
# print negative eigenvalues
eigs = np.linalg.eigvals(full_corr_matrix.to_numpy())
display(len(eigs[eigs < 0]))
display(eigs[eigs < 0])

# %%
CHOL_DECOMPOSITION_FAILED = None

try:
    chol_mat = np.linalg.cholesky(full_corr_matrix.to_numpy())
    cov_inv = np.linalg.inv(chol_mat)
    print("Works!")
    CHOL_DECOMPOSITION_FAILED = False
except Exception as e:
    print(f"Cholesky decomposition failed: {str(e)}")
    CHOL_DECOMPOSITION_FAILED = True

# %% [markdown]
# ## Adjust

# %%
# %load_ext rpy2.ipython

# %% language="R"
# # taken and adapted from https://www.r-bloggers.com/2013/08/correcting-a-pseudo-correlation-matrix-to-be-positive-semidefinite/
# CorrectCM <- function(CM, p = 0) {
#   n <- dim(CM)[1L]
#   E <- eigen(CM)
#   CM1 <- E$vectors %*% tcrossprod(diag(pmax(E$values, p), n), E$vectors)
#   Balance <- diag(1 / sqrt(diag(CM1)))
#   CM2 <- Balance %*% CM1 %*% Balance
#   return(CM2)
# }

# %%
if CHOL_DECOMPOSITION_FAILED:
    corr_mat_r = full_corr_matrix.to_numpy()

    # %Rpush corr_mat_r
    # %R -o corr_mat_r_fixed corr_mat_r_fixed <- CorrectCM(corr_mat_r, 1e-5)

    display(corr_mat_r_fixed.shape)

    full_corr_matrix_fixed = pd.DataFrame(
        corr_mat_r_fixed,
        index=full_corr_matrix.index.copy(),
        columns=full_corr_matrix.columns.copy(),
    )
    display(full_corr_matrix_fixed.shape)
    display(full_corr_matrix_fixed)
else:
    print("No adjustment was necessary")

# %% [markdown]
# ## Make sure the new matrix is positive definite

# %%
if CHOL_DECOMPOSITION_FAILED:
    # print negative eigenvalues
    eigs = np.linalg.eigvals(full_corr_matrix_fixed.to_numpy())
    display(len(eigs[eigs < 0]))
    display(eigs[eigs < 0])

    chol_mat = np.linalg.cholesky(full_corr_matrix_fixed.to_numpy())
    cov_inv = np.linalg.inv(chol_mat)

    assert not np.isnan(chol_mat).any()
    assert not np.isinf(chol_mat).any()
    assert not np.iscomplex(chol_mat).any()

    assert not np.isnan(cov_inv).any()
    assert not np.isinf(cov_inv).any()
    assert not np.iscomplex(cov_inv).any()
else:
    print("No adjustment was necessary")

# %% [markdown]
# ## Compare adjusted and original correlation matrix

# %%
if CHOL_DECOMPOSITION_FAILED:
    # print the element-wise difference between the original and the adjusted matrix
    _diff = ((full_corr_matrix - full_corr_matrix_fixed) ** 2).unstack().sum()
    display(_diff)
    assert _diff < 1e-5
else:
    print("No adjustment was necessary")

# %% [markdown]
# ## Replace original matrix with adjusted one

# %%
if CHOL_DECOMPOSITION_FAILED:
    full_corr_matrix = full_corr_matrix_fixed
else:
    print("No adjustment was necessary")

# %% [markdown] tags=[]
# # Stats

# %% tags=[]
full_corr_matrix_flat = full_corr_matrix.mask(
    np.triu(np.ones(full_corr_matrix.shape)).astype(bool)
).stack()

# %%
display(full_corr_matrix_flat.shape)
assert full_corr_matrix_flat.shape[0] == int(
    full_corr_matrix.shape[0] * (full_corr_matrix.shape[0] - 1) / 2
)

# %% [markdown]
# ## On all correlations

# %%
_corr_mat = full_corr_matrix_flat

# %%
_corr_mat.shape

# %%
_corr_mat.head()

# %%
_corr_mat.describe().apply(str)

# %%
display(_corr_mat.quantile(np.arange(0, 1, 0.05)))

# %%
display(_corr_mat.quantile(np.arange(0, 0.001, 0.0001)))

# %%
display(_corr_mat.quantile(np.arange(0.999, 1.0, 0.0001)))

# %% [markdown]
# ### Plot: distribution

# %%
with sns.plotting_context("paper", font_scale=1.5):
    g = sns.displot(_corr_mat, kde=True, height=7)
    g.ax.set_title("Distribution of gene correlation values in all chromosomes")

# %% [markdown]
# ### Plot: heatmap

# %%
vmin_val = 0.0
vmax_val = max(0.05, _corr_mat.quantile(0.99))
display(f"{vmin_val} / {vmax_val}")

# %%
f, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(
    full_corr_matrix,
    xticklabels=False,
    yticklabels=False,
    square=True,
    vmin=vmin_val,
    vmax=vmax_val,
    cmap="rocket_r",
    ax=ax,
)
ax.set_title("Gene correlations in all chromosomes")

# %% [markdown]
# ## On nonzero correlations

# %%
nonzero_corrs = full_corr_matrix_flat[full_corr_matrix_flat > 0.0]

# %%
_corr_mat = nonzero_corrs

# %%
_corr_mat.shape

# %%
_corr_mat.head()

# %%
_corr_mat.describe().apply(str)

# %%
display(_corr_mat.quantile(np.arange(0, 1, 0.05)))

# %%
display(_corr_mat.quantile(np.arange(0, 0.001, 0.0001)))

# %%
display(_corr_mat.quantile(np.arange(0.999, 1.0, 0.0001)))

# %% [markdown]
# ### Plot: distribution

# %%
with sns.plotting_context("paper", font_scale=1.5):
    g = sns.displot(_corr_mat, kde=True, height=7)
    g.ax.set_title("Distribution of gene correlation values in all chromosomes")

# %% [markdown] tags=[]
# # Save

# %% [markdown] tags=[]
# ## With ensemble ids

# %% [markdown] tags=[]
# ## With gene symbols

# %%
output_file = OUTPUT_DIR_BASE / "gene_corrs-symbols.pkl"
display(output_file)

# %% tags=[]
gene_corrs = full_corr_matrix.rename(
    index=Gene.GENE_ID_TO_NAME_MAP, columns=Gene.GENE_ID_TO_NAME_MAP
)

# %%
assert not gene_corrs.isna().any(None)
assert not np.isinf(gene_corrs.to_numpy()).any()
assert not np.iscomplex(gene_corrs.to_numpy()).any()

# %% tags=[]
assert gene_corrs.index.is_unique
assert gene_corrs.columns.is_unique

# %% tags=[]
gene_corrs.shape

# %% tags=[]
gene_corrs.head()

# %% tags=[]
gene_corrs.to_pickle(output_file)

# %% tags=[]
