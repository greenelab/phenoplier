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
# This notebook reads the correlation matrix generated and creates new matrices with different "within distances" across genes.
# For example, it generates a new correlation matrix with only genes within a distance of 10mb.
#
# It has specicfic parameters for papermill (see under `Settings` below).
#
# This notebook is not directly run. See README.md.

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
REFERENCE_PANEL = None

# predictions models such as MASHR or ELASTIC_NET
EQTL_MODEL = None

# a list with different distances to generate
DISTANCES = [10, 5, 2]

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
OUTPUT_DIR_BASE = (
    conf.RESULTS["GLS"]
    / "gene_corrs"
    / "cohorts"
    / COHORT_NAME
    / REFERENCE_PANEL.lower()
    / EQTL_MODEL.lower()
)
assert OUTPUT_DIR_BASE.exists()

display(f"Using output dir base: {OUTPUT_DIR_BASE}")

# %% [markdown] tags=[]
# # Load data

# %% [markdown] tags=[]
# ## Gene correlations

# %% tags=[]
gene_corrs = pd.read_pickle(OUTPUT_DIR_BASE / "gene_corrs-symbols.pkl")

# %%
gene_corrs.shape

# %%
gene_corrs.head()

# %%
genes_corrs_sum = gene_corrs.sum()
n_genes_included = genes_corrs_sum[genes_corrs_sum > 1.0].shape[0]
display(f"Number of genes with correlations with other genes: {n_genes_included}")

# %%
genes_corrs_nonzero_sum = (gene_corrs > 0.0).astype(int).sum().sum()
display(f"Number of nonzero cells: {genes_corrs_nonzero_sum}")

# %% [markdown] tags=[]
# ## Get gene objects

# %%
gene_objs = [Gene(name=gene_name) for gene_name in gene_corrs.index]

# %%
display(len(gene_objs))

# %% [markdown] tags=[]
# # Subset full correlation matrix using difference "within distances" across genes

# %%
for full_distance in DISTANCES:
    distance = full_distance / 2.0
    print(f"Using within distance: {distance}", flush=True)

    # compute a boolean within distance matrix using the given distance
    genes_within_distance = np.eye(len(gene_objs)).astype(bool)
    for g0_idx in range(len(gene_objs) - 1):
        g0_obj = gene_objs[g0_idx]

        for g1_idx in range(g0_idx + 1, len(gene_objs)):
            g1_obj = gene_objs[g1_idx]

            g0_g1_wd = g0_obj.within_distance(g1_obj, distance * 1e6)

            genes_within_distance[g0_idx, g1_idx] = g0_g1_wd
            genes_within_distance[g1_idx, g0_idx] = g0_g1_wd

    genes_within_distance = pd.DataFrame(
        genes_within_distance,
        index=gene_corrs.index.copy(),
        columns=gene_corrs.columns.copy(),
    )

    # subset full correlation matrix
    gene_corrs_within_distance = gene_corrs[genes_within_distance].fillna(0.0)
    assert not gene_corrs_within_distance.equals(gene_corrs)
    assert not np.allclose(gene_corrs_within_distance.to_numpy(), gene_corrs.to_numpy())
    display(gene_corrs_within_distance)

    # check if the new matrix is positive definite
    is_pos_def = check_pos_def(gene_corrs_within_distance)

    if is_pos_def:
        print("all good.", flush=True, end="\n")
    else:
        print("not positive definite, fixing... ", flush=True, end="")
        corr_data_adjusted = adjust_non_pos_def(gene_corrs_within_distance)

        is_pos_def = check_pos_def(corr_data_adjusted)
        assert is_pos_def, "Could not adjust gene correlation matrix"

        print("fixed! comparing...", flush=True, end="\n")
        compare_matrices(gene_corrs_within_distance, corr_data_adjusted)

        # save
        gene_corrs_within_distance = corr_data_adjusted

    # checks
    assert not gene_corrs_within_distance.isna().any(None)
    assert not np.isinf(gene_corrs_within_distance.to_numpy()).any()
    assert not np.iscomplex(gene_corrs_within_distance.to_numpy()).any()

    # show stats
    genes_corrs_sum = gene_corrs_within_distance.sum()
    n_genes_included = genes_corrs_sum[genes_corrs_sum > 1.0].shape[0]
    display(f"Number of genes with correlations with other genes: {n_genes_included}")

    genes_corrs_nonzero_sum = (gene_corrs_within_distance > 0.0).astype(int).sum().sum()
    display(f"Number of nonzero cells: {genes_corrs_nonzero_sum}")

    corr_matrix_flat = gene_corrs_within_distance.mask(
        np.triu(np.ones(gene_corrs_within_distance.shape)).astype(bool)
    ).stack()
    display(corr_matrix_flat.describe().apply(str))

    # save file
    output_filepath = (
        OUTPUT_DIR_BASE
        / f"gene_corrs-symbols-within_distance_{int(full_distance)}mb.pkl"
    )
    display(output_filepath)

    gene_corrs_within_distance.to_pickle(output_filepath)

    print("\n")

# %% tags=[]
