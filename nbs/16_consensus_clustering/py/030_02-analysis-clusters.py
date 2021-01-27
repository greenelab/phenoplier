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
# The goal of this notebook is very simple: it just lists the content (traits/diseases) that belong to each cluster across all selected "best partitions". Although one would take a look at them here to check whether clusters of traits make sense, that analysis is carried out first by looking at the clustering trees (which are generated later). Then, this notebooks serves as a simple list with the content of the clusters.

# %% [markdown]
# # Modules loading

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from IPython.display import display
from pathlib import Path

import pandas as pd

from utils import generate_result_set_name
import conf

# %% [markdown]
# # Settings

# %%
CONSENSUS_CLUSTERING_DIR = Path(
    conf.RESULTS["CLUSTERING_DIR"], "consensus_clustering"
).resolve()

display(CONSENSUS_CLUSTERING_DIR)

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
# # Analysis of clusterings

# %%
from IPython.display import HTML


# %%
def show_cluster_stats(clustering_data, selected_partition, selected_cluster):
    traits = [t for t in clustering_data[selected_partition == selected_cluster].index]
    display(len(traits))
    display(traits)


# %%
selected_k_values = best_partitions[best_partitions["selected"]].index.tolist()
selected_k_values.sort()
display(selected_k_values)

# %%
for k in selected_k_values:
    display(HTML(f"<h2>Partition with k={k}</h2>"))
    display(best_partitions.loc[k])

    part = best_partitions.loc[k, "partition"]
    display(part.shape)

    part_stats = pd.Series(part).value_counts()
    display(part_stats)

    for cluster_number in part_stats.index.sort_values():
        display(HTML(f"<h3>Cluster {k}.{cluster_number}</h3>"))

        cluster_traits = data_umap[part == cluster_number].index
        display(len(cluster_traits))
        display(cluster_traits)

# %%
