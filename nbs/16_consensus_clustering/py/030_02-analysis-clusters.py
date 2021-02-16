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
# The goal of this notebook is very simple: it just lists the content (traits/diseases) that belong to each cluster across all selected "best partitions". Although one would take a look at them here to check whether clusters of traits make sense, that analysis is carried out first by looking at the clustering trees (which are generated later). Then, this notebooks serves as a simple list with the content of the clusters.

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
from IPython.display import display
from pathlib import Path

import pandas as pd

from utils import generate_result_set_name
import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
CONSENSUS_CLUSTERING_DIR = Path(
    conf.RESULTS["CLUSTERING_DIR"], "consensus_clustering"
).resolve()

display(CONSENSUS_CLUSTERING_DIR)

# %% [markdown] tags=[]
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

# %% tags=[]
data_umap = pd.read_pickle(input_filepath)

# %% tags=[]
data_umap.shape

# %% tags=[]
data_umap.head()

# %% [markdown] tags=[]
# # Load best partitions

# %% tags=[]
input_file = Path(CONSENSUS_CLUSTERING_DIR, "best_partitions_by_k.pkl").resolve()
display(input_file)

# %% tags=[]
best_partitions = pd.read_pickle(input_file)

# %% tags=[]
best_partitions.shape

# %% tags=[]
best_partitions.head()

# %% [markdown] tags=[]
# # Analysis of clusterings

# %% tags=[]
from IPython.display import HTML


# %% tags=[]
def show_cluster_stats(clustering_data, selected_partition, selected_cluster):
    traits = [t for t in clustering_data[selected_partition == selected_cluster].index]
    display(len(traits))
    display(traits)


# %% tags=[]
selected_k_values = best_partitions[best_partitions["selected"]].index.tolist()
selected_k_values.sort()
display(selected_k_values)

# %% tags=[]
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

# %% tags=[]
