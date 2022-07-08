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
# **TODO** update

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.gofplots import qqplot_2samples
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import conf
from entity import Gene

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
INPUT_DIR = conf.RESULTS["GLS_NULL_SIMS"] / "phenoplier"
display(INPUT_DIR)

# %% [markdown]
# # Load MultiXcan genes present in results

# %%
_tmp = pd.read_csv(
    conf.RESULTS["GLS_NULL_SIMS"]
    / "twas"
    / "smultixcan"
    / "random.pheno0-gtex_v8-mashr-smultixcan.txt",
    sep="\t",
)

# %%
_tmp.shape

# %%
_tmp.head()

# %%
multixcan_genes = set(_tmp["gene_name"])
display(len(multixcan_genes))
display(list(multixcan_genes)[:10])

# %% [markdown]
# # Load MultiPLIER Z matrix

# %%
multiplier_z = pd.read_pickle(conf.MULTIPLIER["MODEL_Z_MATRIX_FILE"])

# %%
multiplier_z.shape

# %%
# keep genes only present in MultiXcan
multiplier_z = multiplier_z.loc[
    sorted(multixcan_genes.intersection(multiplier_z.index))
]

# %%
multiplier_z.shape

# %%
multiplier_z.head()

# %% [markdown] tags=[]
# # Common genes

# %%
common_genes = multixcan_genes.intersection(set(multiplier_z.index))

# %%
len(common_genes)

# %%
list(common_genes)[:10]

# %% tags=[]
common_genes_objs = {
    gene_name: Gene(name=gene_name)
    for gene_name in common_genes
    if gene_name in Gene.GENE_NAME_TO_ID_MAP
}

# %% tags=[]
len(common_genes_objs)

# %% tags=[]
common_genes_objs["GAS6"].ensembl_id

# %% tags=[]
_gene_obj = list(common_genes_objs.values())

genes_info = pd.DataFrame(
    {
        "name": [g.name for g in _gene_obj],
        "id": [g.ensembl_id for g in _gene_obj],
        "chr": [g.chromosome for g in _gene_obj],
        "band": [g.band for g in _gene_obj],
        "start_position": [g.get_attribute("start_position") for g in _gene_obj],
        "end_position": [g.get_attribute("end_position") for g in _gene_obj],
    }
)

# %% tags=[]
genes_info.shape

# %%
genes_info[genes_info.isna().any(axis=1)]

# %%
genes_info = genes_info.dropna()
display(genes_info.shape)

# %%
genes_info["chr"] = genes_info["chr"].astype(int)
genes_info["start_position"] = genes_info["start_position"].astype(int)
genes_info["end_position"] = genes_info["end_position"].astype(int)

# %%
genes_info.dtypes

# %% tags=[]
genes_info.head()

# %% [markdown]
# # List genes by chromosome and position

# %%
genes_info.sort_values(["chr", "start_position"])

# %% [markdown]
# ## Same chromosome and close

# %%
with pd.option_context("display.max_rows", None):
    _tmp = genes_info[genes_info["band"].str.startswith("17q")].sort_values(
        ["start_position"]
    )
    display(_tmp)

# %% [markdown]
# ## Same chromosome but far away

# %%
genes_info[genes_info["chr"] == 6].sort_values(["start_position"])

# %% [markdown]
# # Explore specific LVs

# %%
multiplier_z["LV45"].sort_values(ascending=False).head(20)

# %%
lv_top_genes = multiplier_z["LV45"].sort_values(ascending=False).head(20).index.tolist()

# %%
genes_info[genes_info["name"].isin(lv_top_genes)].sort_values(["chr", "start_position"])

# %%
