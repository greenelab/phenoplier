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
# This notebook contains the interpretation of a cluster (which features/latent variables in the original data are useful to distinguish traits in the cluster).
#
# See section [LV analysis](#lv_analysis) below

# %% [markdown]
# # Modules loading

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import pickle
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# from IPython.display import HTML
from tqdm import tqdm

from clustering.methods import ClusterInterpreter
from data.recount2 import LVAnalysis
from data.cache import read_data
import conf

# %% [markdown]
# # Settings

# %% tags=["parameters"]
# PARTITION_K = 26
# PARTITION_CLUSTER_ID = 18

# %% [markdown]
# # Load MultiPLIER summary

# %%
multiplier_model_summary = read_data(conf.MULTIPLIER["MODEL_SUMMARY_FILE"])

# %%
multiplier_model_summary.shape

# %%
multiplier_model_summary.head()

# %% [markdown]
# # Load data

# %% [markdown]
# ## Original data

# %% tags=[]
INPUT_SUBSET = "z_score_std"

# %% tags=[]
INPUT_STEM = "projection-smultixcan-efo_partial-mashr-zscores"

# %% tags=[]
input_filepath = Path(
    conf.RESULTS["DATA_TRANSFORMATIONS_DIR"],
    INPUT_SUBSET,
    f"{INPUT_SUBSET}-{INPUT_STEM}.pkl",
).resolve()
display(input_filepath)

assert input_filepath.exists(), "Input file does not exist"

input_filepath_stem = input_filepath.stem
display(input_filepath_stem)

# %% tags=[]
data = pd.read_pickle(input_filepath)

# %% tags=[]
data.shape

# %% tags=[]
data.head()

# %% [markdown]
# ## Clustering partitions

# %%
# CONSENSUS_CLUSTERING_DIR = Path(
#     conf.RESULTS["CLUSTERING_DIR"], "consensus_clustering"
# ).resolve()

# display(CONSENSUS_CLUSTERING_DIR)

# %%
# input_file = Path(CONSENSUS_CLUSTERING_DIR, "best_partitions_by_k.pkl").resolve()
# display(input_file)

# %%
# best_partitions = pd.read_pickle(input_file)

# %%
# best_partitions.shape

# %%
# best_partitions.head()

# %% [markdown]
# # Functions

# %%
# def show_cluster_stats(data, partition, cluster):
#     cluster_traits = data[partition == cluster].index
#     display(f"Cluster '{cluster}' has {len(cluster_traits)} traits")
#     display(cluster_traits)

# %% [markdown]
# # LV analysis
# <a id="lv_analysis"></a>

# %% [markdown]
# ## Associated traits

# %%
# display(best_partitions.loc[PARTITION_K])
# part = best_partitions.loc[PARTITION_K, "partition"]

# %%
# show_cluster_stats(data, part, PARTITION_CLUSTER_ID)

# %% [markdown]
# ## Associated latent variables

# %%
# ci = ClusterInterpreter(
#     threshold=1.0,
#     max_features=20,
#     max_features_to_explore=100,
# )

# %%
# ci.fit(data, part, PARTITION_CLUSTER_ID)

# %%
# ci.features_

# %%
# # save interpreter instance
# output_dir = Path(
#     conf.RESULTS["CLUSTERING_INTERPRETATION"]["BASE_DIR"],
#     "cluster_lvs",
#     f"part{PARTITION_K}",
# )
# output_dir.mkdir(exist_ok=True, parents=True)

# %%
# output_file = Path(
#     output_dir, f"cluster_interpreter-part{PARTITION_K}_k{PARTITION_CLUSTER_ID}.pkl"
# )
# display(output_file)

# %%
# ci.features_.to_pickle(output_file)

# %% [markdown]
# ## Top attributes

# %%
# lvs_list = "LV116,LV931,LV744,LV697,LV885,LV536,LV550,LV220,LV272,LV739,LV678,LV470,LV66,LV189,LV517,LV840,LV246,LV502,LV525,LV85".split(",")

# %%
lvs_list = pd.read_pickle("/tmp/niacin_lv_list.pkl").index.tolist()

# %%
len(lvs_list)

# %%
lvs_list[:10]


# %%
def _my_func(x):
    _cols = [c for c in x.index if not c.startswith("LV")]
    _tmp = x[_cols].dropna()
    if _tmp.shape[0] > 0:
        return _tmp.iloc[0]

    return None


# %%
cell_type_dfs = []
tissue_dfs = []

pbar = tqdm(lvs_list[:50])
for lv_name in pbar:
    pbar.set_description(lv_name)

    #     lv_name = lv_info["name"]
    lv_obj = LVAnalysis(lv_name, data)

    #     # show lv prior knowledge match (pathways)
    #     lv_pathways = multiplier_model_summary[
    #         multiplier_model_summary["LV index"].isin((lv_name[2:],))
    #         & (
    #             (multiplier_model_summary["FDR"] < 0.05)
    #             | (multiplier_model_summary["AUC"] >= 0.75)
    #         )
    #     ]
    #     display(lv_pathways)

    lv_data = lv_obj.get_experiments_data()

    #     display("")
    #     display(lv_obj.lv_traits.head(20))
    #     display("")
    #     display(lv_obj.lv_genes.head(10))

    # get cell type attributes
    #     lv_attrs = lv_obj.get_attributes_variation_score()
    lv_attrs = pd.Series(lv_data.columns.tolist())
    lv_attrs = lv_attrs[
        lv_attrs.str.match(
            "(?:cell[^\w]*type$)",
            case=False,
            flags=re.IGNORECASE,
        ).values
    ].sort_values(ascending=False)
    display(lv_attrs)

    lv_attrs_data = lv_data[lv_attrs.tolist() + [lv_name]]
    lv_attrs_data = lv_attrs_data.assign(attr=lv_attrs_data.apply(_my_func, axis=1))
    lv_attrs_data = lv_attrs_data.drop(columns=lv_attrs.tolist())
    lv_attrs_data = lv_attrs_data.dropna().sort_values(lv_name, ascending=False)
    lv_attrs_data = lv_attrs_data.rename(columns={lv_name: "lv"})
    cell_type_dfs.append(lv_attrs_data)

    # get tissue attributes
    lv_attrs = pd.Series(lv_data.columns.tolist())
    lv_attrs = lv_attrs[
        lv_attrs.str.match(
            "(?:tissue$)|(?:tissue[^\w]*type$)",
            case=False,
            flags=re.IGNORECASE,
        ).values
    ].sort_values(ascending=False)
    display(lv_attrs)

    lv_attrs_data = lv_data[lv_attrs.tolist() + [lv_name]]
    lv_attrs_data = lv_attrs_data.assign(attr=lv_attrs_data.apply(_my_func, axis=1))
    lv_attrs_data = lv_attrs_data.drop(columns=lv_attrs.tolist())
    lv_attrs_data = lv_attrs_data.dropna().sort_values(lv_name, ascending=False)
    lv_attrs_data = lv_attrs_data.rename(columns={lv_name: "lv"})
    tissue_dfs.append(lv_attrs_data)

# %% [markdown]
# # LVs selection

# %%
N_TOP_LVS = 20


# %% [markdown]
# # Cell types

# %%
def _get_lv_rank(data):
    data = data.copy()
    data["lv"] = data["lv"].rank()
    return data


# %%
df = pd.concat(cell_type_dfs[:N_TOP_LVS], ignore_index=True)
# df = pd.concat([_get_lv_rank(x) for x in cell_type_dfs[:N_TOP_LVS]], ignore_index=True)

# %%
df = df[~df["attr"].str.lower().str.contains("cancer")]

# %%
df.groupby("attr").median().squeeze().sort_values(ascending=False).head(25)

# %%
df = df.replace(
    {
        "attr": {
            "mural granulosa cells": "Granulosa cells",
            "cumulus granulosa cells": "Granulosa cells",
            "WAT": "White adipose tissue",
            "BAT": "Brown adipose tissue",
            "human adipose-derived stem cells": "Adipose-derived stem cells",
            "Primary Monocytes(BC8)": "Primary monocytes",
            "Primary Monocytes(BC9)": "Primary monocytes",
            "Primary Monocytes(BC12)": "Primary monocytes",
            "Primary Monocytes(BC11)": "Primary monocytes",
        }
    }
)

# %%
cat_order = df.groupby("attr").median().squeeze()
cat_order = cat_order.sort_values(ascending=False)
cat_order = cat_order.head(20)
cat_order = cat_order.index

# %%
with sns.plotting_context("paper", font_scale=1.0), sns.axes_style("whitegrid"):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax = sns.boxplot(
        data=df,
        x="attr",
        y="lv",
        order=cat_order,
        linewidth=None,
        ax=ax,
    )
    plt.xticks(rotation=45, horizontalalignment="right")

# %% [markdown]
# # Tissues

# %%
df = pd.concat(tissue_dfs[:N_TOP_LVS], ignore_index=True)
# df = pd.concat([_get_lv_rank(x) for x in tissue_dfs[:N_TOP_LVS]], ignore_index=True)

# %%
df.groupby("attr").mean().squeeze().sort_values(ascending=False).head(50)

# %% [markdown]
# # LV analysis

# %%
lv_obj1 = LVAnalysis("LV931", data)

# %%
lv_obj1.lv_genes.head(20)

# %%
lv_data1 = lv_obj1.get_experiments_data()

# %%
lv_data1.shape

# %%
_tmp = lv_data1[["cell type", "LV931"]].dropna()

# %%
_tmp[_tmp["cell type"].str.contains("CD14 cells")]

# %%
_tmp.loc["SRP059735"]

# %%

# %%

# %%

# %%

# %%

# %%
lv_obj3 = LVAnalysis("LV536", data)

# %%
lv_obj3.lv_genes.head(20)

# %%
lv_data3 = lv_obj3.get_experiments_data()

# %%
lv_data3.shape

# %%
_tmp = lv_data3[["cell type", "cancer or normal", "LV536"]].dropna()

# %%
_tmp[_tmp["cell type"].str.contains("transitional")].sort_values("cancer or normal")

# %%

# %%

# %%

# %%
lv_obj = LVAnalysis("LV116", data)

# %%
lv_obj.lv_traits.to_frame().loc["atherosclerosis"]

# %%
lv_obj.lv_traits.to_frame().loc["hypertension"]

# %%
lv_obj.lv_traits.to_frame().loc["MAGNETIC_HDL.C"]

# %%
lv_obj.lv_traits.to_frame().loc["MAGNETIC_CH2.DB.ratio"]

# %%
lv_obj.lv_traits.to_frame().loc["coronary artery disease"]

# %%
lv_obj.lv_genes.head(20)

# %%
lv_obj.lv_genes[lv_obj.lv_genes["gene_name"].str.startswith("ABC")]

# %%
lv_data = lv_obj.get_experiments_data()

# %%
lv_data.shape

# %%
# _tmp = lv_data[["cell type", "ad type", "treatment", "LV116"]].dropna()
_tmp = lv_data[["cell type", "LV116"]].dropna()

# %%
_tmp[_tmp["cell type"].str.contains("Monocyte")]

# %%
_tmp.loc["SRP066356"]

# %%
_tmp[_tmp["cell type"].str.contains("PBMCs")].groupby("treatment").describe()

# %%

# %%

# %%
lv_obj2 = LVAnalysis("LV881", data)

# %%
lv_obj2.lv_genes.head(20)

# %%
lv_attrs = lv_obj.get_attributes_variation_score()
_tmp = pd.Series(lv_attrs.index)
lv_attrs = lv_attrs[
    _tmp.str.match(
        "(?:cell.+type$)|(?:tissue$)|(?:tissue.+type$)",
        case=False,
        flags=re.IGNORECASE,
    ).values
].sort_values(ascending=False)
display(lv_attrs)

# %%
with sns.plotting_context("paper", font_scale=1.0), sns.axes_style("whitegrid"):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax = lv_obj2.plot_attribute("tissue", top_x_values=20)

# %%
with sns.plotting_context("paper", font_scale=1.0), sns.axes_style("whitegrid"):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax = lv_obj2.plot_attribute("cell type", top_x_values=20)

# %%
