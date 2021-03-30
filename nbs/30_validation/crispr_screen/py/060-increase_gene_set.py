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

# %% [markdown] tags=[]
# # Description

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
import pickle
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import HTML

from clustering.methods import ClusterInterpreter
from data.recount2 import LVAnalysis
from data.cache import read_data
import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=["parameters"]
# PARTITION_K = None
# PARTITION_CLUSTER_ID = None

# %% tags=["injected-parameters"]
# # Parameters
# PARTITION_K = 22
# PARTITION_CLUSTER_ID = 19


# %% [markdown] tags=[]
# # Load MultiPLIER summary

# %% tags=[]
multiplier_model_summary = read_data(conf.MULTIPLIER["MODEL_SUMMARY_FILE"])

# %% tags=[]
multiplier_model_summary.shape

# %% tags=[]
multiplier_model_summary.head()

# %% [markdown] tags=[]
# # Load data

# %% [markdown] tags=[]
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

# %% [markdown] tags=[]
# ## Clustering partitions

# %% tags=[]
# CONSENSUS_CLUSTERING_DIR = Path(
#     conf.RESULTS["CLUSTERING_DIR"], "consensus_clustering"
# ).resolve()

# display(CONSENSUS_CLUSTERING_DIR)

# %% tags=[]
# input_file = Path(CONSENSUS_CLUSTERING_DIR, "best_partitions_by_k.pkl").resolve()
# display(input_file)

# %% tags=[]
# best_partitions = pd.read_pickle(input_file)

# %% tags=[]
# best_partitions.shape

# %% tags=[]
# best_partitions.head()

# %% [markdown] tags=[]
# # Functions

# %% tags=[]
# def show_cluster_stats(data, partition, cluster):
#     cluster_traits = data[partition == cluster].index
#     display(f"Cluster '{cluster}' has {len(cluster_traits)} traits")
#     display(cluster_traits)

# %% [markdown] tags=[]
# # LV analysis
# <a id="lv_analysis"></a>

# %% [markdown] tags=[]
# ## Associated traits

# %% tags=[]
# display(best_partitions.loc[PARTITION_K])
# part = best_partitions.loc[PARTITION_K, "partition"]

# %% tags=[]
# show_cluster_stats(data, part, PARTITION_CLUSTER_ID)

# %% [markdown] tags=[]
# ## Top attributes

# %% [markdown] tags=[]
# Here we go through the list of associated latent variables and, for each, we show associated pathways (prior knowledge), top traits, top genes and the top tissues/cell types where those genes are expressed.

# %%
# lv_obj = LVAnalysis("LV678", data)

# %%
# lv_data = lv_obj.get_experiments_data()

# %%
# _tmp = lv_data[["tissue", "LV678"]].dropna()

# %%
# _tmp[_tmp["tissue"].str.contains("Muscle")]

# %%
selected_lvs = [
    "LV707",
    "LV905",
    "LV915",
    "LV750",
    "LV341",
    "LV310",
    "LV48",
    "LV509",
    "LV467",
    "LV64",
    "LV490",
    "LV550",
    "LV621",
    "LV775",
    "LV415",
    "LV504",
    "LV507",
    "LV494",
    "LV399",
    "LV246",
    "LV120",
    "LV122",
    "LV489",
    "LV515",
    "LV783",
    "LV768",
    "LV577",
]

# %%
LV_OBJS = {}

# %% tags=[]
for lv_name in selected_lvs:
    display(HTML(f"<h2>{lv_name}</h2>"))

    #     lv_name = lv_info["name"]
    lv_obj = lv_exp = LVAnalysis(lv_name, data)

    # show lv prior knowledge match (pathways)
    lv_pathways = multiplier_model_summary[
        multiplier_model_summary["LV index"].isin((lv_name[2:],))
        & (
            (multiplier_model_summary["FDR"] < 0.05)
            | (multiplier_model_summary["AUC"] >= 0.75)
        )
    ]
    display(lv_pathways)

    lv_data = lv_obj.get_experiments_data()

    display("")
    display(lv_obj.lv_traits.head(20))
    display("")
    display(lv_obj.lv_genes.head(10))

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

    for _lva in lv_attrs.index:
        display(HTML(f"<h3>{_lva}</h3>"))
        display(lv_data[_lva].dropna().reset_index()["project"].unique())

        with sns.plotting_context("paper", font_scale=1.0), sns.axes_style("whitegrid"):
            fig, ax = plt.subplots(figsize=(14, 8))
            ax = lv_obj.plot_attribute(_lva, top_x_values=20)
            if ax is None:
                plt.close(fig)
                continue
            display(fig)
            plt.close(fig)

    LV_OBJS[lv_name] = lv_obj

# %% tags=[]
