# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
#     formats: ipynb,py//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] tags=[]
# # Description

# %% [markdown] tags=[]
# This notebook reads all gene correlations across all chromosomes and computes a single correlation matrix by assembling a big correlation matrix with all genes.
#
# It has specicfic parameters for papermill (see under `Settings` below).
#
# This notebook should not be directly run. It is used by other notebooks.

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.distance import squareform
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import conf
from entity import Gene
from correlations import (
    check_pos_def,
    compare_matrices,
    correct_corr_mat,
    adjust_non_pos_def,
)

# %% [markdown] tags=[]
# # Settings

# %% tags=["parameters"]
# a cohort name (it could be something like UK_BIOBANK, etc)
COHORT_NAME = None

# reference panel such as 1000G or GTEX_V8
REFERENCE_PANEL = "GTEX_V8"

# predictions models such as MASHR or ELASTIC_NET
EQTL_MODEL = "MASHR"

# output dir
OUTPUT_DIR_BASE = None

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

display(f"eQTL model: {EQTL_MODEL})")

# %%
assert (
    OUTPUT_DIR_BASE is not None and len(OUTPUT_DIR_BASE) > 0
), "Output directory path must be given"

OUTPUT_DIR_BASE = (Path(OUTPUT_DIR_BASE) / "gene_corrs" / COHORT_NAME).resolve()

OUTPUT_DIR_BASE.mkdir(parents=True, exist_ok=True)

display(f"Using output dir base: {OUTPUT_DIR_BASE}")

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
genes_info = pd.read_pickle(OUTPUT_DIR_BASE / "genes_info.pkl")

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
assert not genes_info.isna().any(axis=None)

# %%
genes_info.dtypes

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
    print(chr_corr_file.name, flush=True, end="... ")

    # get correlation matrix for this chromosome
    corr_data = pd.read_pickle(chr_corr_file)

    # save gene correlation matrix
    full_corr_matrix.loc[corr_data.index, corr_data.columns] = corr_data

    # save inverse of Cholesky decomposition of gene correlation matrix
    # first, adjust correlation matrix if it is not positive definite
    is_pos_def = check_pos_def(corr_data)

    if is_pos_def:
        print("all good.", flush=True, end="\n")
    else:
        print("not positive definite, fixing... ", flush=True, end="")
        corr_data_adjusted = adjust_non_pos_def(corr_data)

        is_pos_def = check_pos_def(corr_data_adjusted)
        assert is_pos_def, "Could not adjust gene correlation matrix"

        print("fixed! comparing...", flush=True, end="\n")
        compare_matrices(corr_data, corr_data_adjusted)

        corr_data = corr_data_adjusted

        # save
        full_corr_matrix.loc[corr_data.index, corr_data.columns] = corr_data

    print("\n")

# %%
full_corr_matrix.shape

# %%
full_corr_matrix.head()

# %%
np.all(full_corr_matrix.to_numpy().diagonal() == 1.0)

# %% [markdown] tags=[]
# ## Some checks

# %%
_min_val = full_corr_matrix.min().min()
display(_min_val)
# assert _min_val >= -0.05

# %%
_max_val = full_corr_matrix.max().max()
display(_max_val)
# assert _max_val <= 1.05

# %% [markdown] tags=[]
# ## Positive definiteness

# %% [markdown] tags=[]
# In some cases, even if the submatrices are adjusted, the whole one is not.
#
# So here I check that again.

# %%
is_pos_def = check_pos_def(full_corr_matrix)

if is_pos_def:
    print("all good.", flush=True, end="\n")
else:
    print("not positive definite, fixing... ", flush=True, end="")
    corr_data_adjusted = adjust_non_pos_def(full_corr_matrix)

    is_pos_def = check_pos_def(corr_data_adjusted)
    assert is_pos_def, "Could not adjust gene correlation matrix"

    print("fixed! comparing...", flush=True, end="\n")
    compare_matrices(full_corr_matrix, corr_data_adjusted)

    full_corr_matrix = corr_data_adjusted

# %% [markdown] tags=[]
# ## Save

# %% [markdown] tags=[]
# ### Gene corrs with gene symbols

# %%
output_file = OUTPUT_DIR_BASE / "gene_corrs-symbols.pkl"
display(output_file)

# %% tags=[]
gene_corrs = full_corr_matrix.rename(
    index=Gene.GENE_ID_TO_NAME_MAP, columns=Gene.GENE_ID_TO_NAME_MAP
)

# %%
assert not gene_corrs.isna().any(axis=None)
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

# %%
del gene_corrs

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

# %% tags=[]
