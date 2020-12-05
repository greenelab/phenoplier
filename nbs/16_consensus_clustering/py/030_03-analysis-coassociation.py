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
# **TODO:** This section of the notebook will be updated when I start actively writing the manuscript. Here I left some code to see the percentage of times a group of traits was clustered togethern across the entire ensemble.

# %% [markdown]
# # Modules loading

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import generate_result_set_name
import conf

# %% [markdown]
# # Settings

# %%
# output dir for this notebook
CONSENSUS_CLUSTERING_DIR = Path(
    conf.RESULTS["CLUSTERING_DIR"], "consensus_clustering"
).resolve()

display(CONSENSUS_CLUSTERING_DIR)

# %% [markdown]
# ## Functions

# %%
from IPython.display import HTML


# %%
def plot_cluster(data, partition, cluster_number, figsize=None):
    k = np.unique(partition).shape[0]

    display(HTML(f"<h3>Cluster {k}.{cluster_number}</h3>"))

    k_traits = data.loc[partition == cluster_number].index

    with sns.plotting_context("paper"):
        f, ax = plt.subplots(figsize=figsize)  # (figsize=(8, 8))

        display(
            sns.heatmap(
                data=coassoc_matrix.loc[k_traits, k_traits],
                vmin=coassoc_matrix_stats["50%"],
                vmax=1.0,
                annot=True,
                fmt=".2f",
                square=True,
            )
        )


# %% [markdown]
# ## Load data

# %% tags=[]
INPUT_SUBSET = "umap"

# %% tags=[]
INPUT_STEM = "z_score_std-projection-smultixcan-efo_partial-mashr-zscores"

# %% tags=[]
DR_OPTIONS = {
    "n_components": 5,
    "metric": "euclidean",
    "n_neighbors": 15,
    "random_state": 0,
}

# %% tags=[]
input_filepath = Path(
    conf.RESULTS["DATA_TRANSFORMATIONS_DIR"],
    INPUT_SUBSET,
    generate_result_set_name(
        DR_OPTIONS, prefix=f"{INPUT_SUBSET}-{INPUT_STEM}-", suffix=".pkl"
    ),
).resolve()
display(input_filepath)

assert input_filepath.exists(), "Input file does not exist"

input_filepath_stem = input_filepath.stem
display(input_filepath_stem)

# %%
data_umap = pd.read_pickle(input_filepath)

# %%
data_umap.shape

# %%
data_umap.head()

# %% [markdown]
# # Load best partitions

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
# # Load coassociation matrix

# %%
input_file = Path(CONSENSUS_CLUSTERING_DIR, "ensemble_coassoc_matrix.npy").resolve()
display(input_file)

# %%
coassoc_matrix = np.load(input_file)

# %%
coassoc_matrix = pd.DataFrame(
    data=1.0 - coassoc_matrix,
    index=data_umap.index.copy(),
    columns=data_umap.index.copy(),
)

# %%
coassoc_matrix.shape

# %%
coassoc_matrix.head()

# %% [markdown]
# ## Stats

# %%
df = coassoc_matrix.where(np.triu(np.ones(coassoc_matrix.shape)).astype(np.bool))
df = df.stack().reset_index()

coassoc_matrix_stats = df[0].describe()

# %%
coassoc_matrix_stats

# %%
# show the general stats for the coassociation matrix, useful to compare results of clusters
coassoc_matrix_stats.apply(str)

# %% [markdown]
# # Plot coassociation of clusters

# %%
k = 5
display(HTML(f"<h2>k: {k}</h2>"))
display(best_partitions.loc[k])

part = best_partitions.loc[k, "partition"]
part_stats = pd.Series(part).value_counts()
display(part_stats)

# %%
plot_cluster(data_umap, part, 4)

# %%
plot_cluster(data_umap, part, 2, figsize=(10, 10))

# %%
