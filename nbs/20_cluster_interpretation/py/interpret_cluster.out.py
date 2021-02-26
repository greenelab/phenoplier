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
# TODO: ADD MORE SECTIONS AND EXPLAIN WHERE TO FIND THINGS.

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

from clustering.methods import ClusterInterpreter
from data.recount2 import LVAnalysis
from data.cache import read_data
from utils import generate_result_set_name
import conf

# %% [markdown]
# # Settings

# %% tags=["parameters"]
PARTITION_K = None
PARTITION_CLUSTER_ID = None

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
# # Load best partitions

# %%
CONSENSUS_CLUSTERING_DIR = Path(
    conf.RESULTS["CLUSTERING_DIR"], "consensus_clustering"
).resolve()

display(CONSENSUS_CLUSTERING_DIR)

# %%
input_file = Path(CONSENSUS_CLUSTERING_DIR, "best_partitions_by_k.pkl").resolve()
display(input_file)

# %%
best_partitions = pd.read_pickle(input_file)

# %%
best_partitions.shape

# %%
best_partitions.head()


# %% [markdown]
# # Functions

# %%
def show_cluster_stats(data, partition, cluster):
    cluster_traits = data[partition == cluster].index
    display(f"Cluster '{cluster}' has {len(cluster_traits)} traits")
    display(cluster_traits)


# %% [markdown]
# # LV analysis

# %%
display(best_partitions.loc[PARTITION_K])
part = best_partitions.loc[PARTITION_K, "partition"]

# %%
show_cluster_stats(data, part, PARTITION_CLUSTER_ID)

# %%
ci = ClusterInterpreter()

# %%
ci.fit(data, part, PARTITION_CLUSTER_ID)

# %%
ci.features_

# %%
for lv_idx, lv_data in ci.features_.iterrows():
    display(HTML(f"<h2>LV{lv_idx}</h2>"))

    _lv = lv_data["name"]
    _lv_obj = lv_exp = LVAnalysis(_lv, data)

    # show lv prior knowledge match
    _lv_gene_sets = multiplier_model_summary[
        multiplier_model_summary["LV index"].isin((_lv[2:],))
        & (
            (multiplier_model_summary["FDR"] < 0.05)
            | (multiplier_model_summary["AUC"] >= 0.75)
        )
    ]
    display(_lv_gene_sets)

    _lv_data = _lv_obj.get_experiments_data()

    display("")
    display(_lv_obj.lv_traits.head(20))
    display("")
    display(_lv_obj.lv_genes.head(10))

    lv_attrs = _lv_obj.get_attributes_variation_score()
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
        display(_lv_data[_lva].dropna().reset_index()["project"].unique())

        with sns.plotting_context("paper", font_scale=1.0), sns.axes_style("whitegrid"):
            fig, ax = plt.subplots(figsize=(14, 8))
            ax = _lv_obj.plot_attribute(_lva, top_x_values=20)
            if ax is None:
                plt.close(fig)
                continue
            display(fig)
            plt.close(fig)

# %%
