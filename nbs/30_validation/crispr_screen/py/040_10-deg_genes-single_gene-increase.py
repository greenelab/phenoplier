# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
#     formats: ipynb,py//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Description

# %% [markdown]
# TODO

# %% [markdown]
# # Modules loading

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import HTML

from entity import Gene

# from clustering.methods import ClusterInterpreter
# from data.recount2 import LVAnalysis
from data.cache import read_data

# from utils import generate_result_set_name
import conf

# %% [markdown]
# # Settings

# %% [markdown]
# # Data loading

# %% [markdown]
# ## MultiPLIER Z matrix

# %%
multiplier_z = read_data(conf.MULTIPLIER["MODEL_Z_MATRIX_FILE"])

# %%
multiplier_z.shape

# %%
multiplier_z.head()

# %% [markdown]
# ## S-MultiXcan results

# %% tags=[]
smultixcan_results_filename = conf.PHENOMEXCAN[
    "SMULTIXCAN_EFO_PARTIAL_MASHR_ZSCORES_FILE"
]

display(smultixcan_results_filename)

# %% tags=[]
results_filename_stem = smultixcan_results_filename.stem
display(results_filename_stem)

# %% tags=[]
smultixcan_results = pd.read_pickle(smultixcan_results_filename)

# %% tags=[]
smultixcan_results = smultixcan_results.rename(index=Gene.GENE_ID_TO_NAME_MAP)

# %% tags=[]
smultixcan_results.index[smultixcan_results.index.duplicated(keep="first")]

# %% tags=[]
smultixcan_results = smultixcan_results.loc[
    ~smultixcan_results.index.duplicated(keep="first")
]

# %% tags=[]
smultixcan_results.shape

# %% tags=[]
smultixcan_results.head()

# %% tags=[]
# standardize
_tmp = smultixcan_results.apply(lambda x: x / x.sum())

# %% tags=[]
_tmp.shape

# %% tags=[]
assert _tmp.shape == smultixcan_results.shape

# %% tags=[]
# some testing
_trait = "body height"
_gene = "SCYL3"
assert (
    _tmp.loc[_gene, _trait]
    == smultixcan_results.loc[_gene, _trait] / smultixcan_results[_trait].sum()
)

_trait = "100001_raw-Food_weight"
_gene = "DPM1"
assert (
    _tmp.loc[_gene, _trait]
    == smultixcan_results.loc[_gene, _trait] / smultixcan_results[_trait].sum()
)

_trait = "estrogen-receptor negative breast cancer"
_gene = "CFH"
assert (
    _tmp.loc[_gene, _trait]
    == smultixcan_results.loc[_gene, _trait] / smultixcan_results[_trait].sum()
)

_trait = "asthma"
_gene = "C1orf112"
assert (
    _tmp.loc[_gene, _trait]
    == smultixcan_results.loc[_gene, _trait] / smultixcan_results[_trait].sum()
)

# %% tags=[]
smultixcan_results = _tmp

# %% [markdown]
# ## fastENLOC results

# %% tags=[]
# input_filename = conf.PHENOMEXCAN["FASTENLOC_EFO_PARTIAL_TORUS_RCP_FILE"]
# display(input_filename)

# %% [markdown]
# ## Differentially expressed genes

# %%
input_filepath = Path(conf.CRISPR["BASE_DIR"], "lipid_DEG.csv")
display(input_filepath)

# %%
deg_genes = pd.read_csv(input_filepath)

# %%
# _bool_cols = [c for c in deg_genes.columns if c.endswith(".DEG")]

# %%
# _bool_cols

# %%
# deg_genes[_bool_cols] = deg_genes[_bool_cols].astype(str)

# %%
# deg_genes.dtypes

# %%
deg_genes.shape

# %%
deg_genes.head()

# %% [markdown]
# ### Select gene set

# %%
df = deg_genes.query("(rank == 3) | (rank == 2)")
# df = deg_genes.query("(rank == -3)")

# %%
# assert df.shape[0] == n_exp_genes

# %%
df.shape

# %%
df_genes = df["gene_name"].unique().tolist()
# assert len(df_genes) == n_exp_genes

display(len(df_genes))
display(df_genes[:10])

# %%
# keep genes present in S-MultiXcan results
df_genes_present = smultixcan_results.index.intersection(df_genes).tolist()
display(len(df_genes_present))
display(df_genes_present[:10])

# %%
# comment out if want to further filter by MultiPLIER models

# # keep only those genes present in the MultiPLIER models
# df_genes_present = multiplier_z.index.intersection(df_genes).tolist()
# display(len(df_genes_present))
# display(df_genes_present[:10])

# %% [markdown]
# # Significant threshold

# %% [markdown]
# # Look at individual genes

# %%
smultixcan_results.loc[df_genes_present[0]].sort_values(ascending=False).head(20)

# %% [markdown]
# # Analysis

# %%
traits = []

for g in df_genes_present:
    _tmp = smultixcan_results.loc[g]

    #     _tmp = _tmp.head(20)

    _tmp = _tmp[_tmp > 0.0].sort_values(ascending=False)

    #     _tmp = _tmp.head(50)
    traits.append(_tmp)

# %%
_tmp = (
    pd.concat(traits)
    .reset_index()
    .groupby("index")
    .sum()
    .sort_values(0, ascending=False)
    .reset_index()
).rename(columns={"index": "trait", 0: "value"})

# %%
top_traits = _tmp.head(100)

# %%
with pd.option_context(
    "display.max_rows", None, "display.max_columns", None, "max_colwidth", None
):
    display(top_traits)

# %% [markdown]
# # Summary using trait categories

# %%
from entity import Trait

# %%
trait_code_to_trait_obj = [
    Trait.get_trait(full_code=t)
    if not Trait.is_efo_label(t)
    else Trait.get_traits_from_efo(t)
    for t in top_traits["trait"]
]

# %%
top_traits = top_traits.assign(
    category=[
        t.category
        if not isinstance(t, list)
        else t[0].category  # FIXME just taking the first one
        for t in trait_code_to_trait_obj
    ]
)

# for t in trait_code_to_trait_obj:
#     _tmp = multiplier_model_summary[
#         (multiplier_model_summary["LV index"] == lv[2:])
#         & (
#             (multiplier_model_summary["FDR"] < 0.05)
#             | (multiplier_model_summary["AUC"] > 0.75)
#         )
#     ]
#     pathways.extend(_tmp["pathway"].tolist())

#     _tmp = phenomexcan_data[lv]
#     _tmp = _tmp[_tmp > 0.0].sort_values(ascending=False)

#     _tmp = _tmp.head(50)

#     _tmp = _tmp.reset_index().rename(columns={lv: "lv", "index": "trait"})

#     _tmp["trait_category"] = [
#         trait_code_to_trait_obj[t].category
#         if not isinstance(trait_code_to_trait_obj[t], list)
#         else trait_code_to_trait_obj[t][0].category
#         for t in _tmp["trait"]
#     ]

#     traits.append(_tmp)

# %%
top_traits.shape

# %%
top_traits.head()

# %%
_tmp = (
    top_traits.groupby("category")
    .mean()
    .sort_values("value", ascending=False)
    .reset_index()
)

# %%
_tmp.head()

# %%
with pd.option_context(
    "display.max_rows", None, "display.max_columns", None, "max_colwidth", None
):
    display(_tmp)

# %%
for row_idx, row in _tmp.iterrows():
    category = row["category"]
    display(HTML(f"<h2>{category}</h2>"))

    _df = (
        top_traits[top_traits["category"] == category]
        .groupby("trait")["value"]
        .mean()
        .sort_values()
    )
    display(_df)

# %%
