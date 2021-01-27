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
# It analyzes how clusters of traits were grouped across the ensemble partitions. For example, a stable cluster (obtained from consensus partitions) of cardiovascular diseases can show that all traits were always grouped together across all partitions of the ensemble; another cluster might show that some traits were clustered more often than others, representing a less stable group of traits.

# %% [markdown]
# **TODO:** This section of the notebook will be updated again when I start actively writing the results section of the manuscript. Here I left some code as example for some clusters.

# %% [markdown]
# # Modules loading

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from IPython.display import display
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
# The coassociation matrix shows the percentage of times a pair of traits was clustered together across the ensemble partitions.

# %% [markdown]
# ## Stats

# %% [markdown]
# Here I show some general stats of the coassociation matrix, useful to compare results below. For instance, if a pair of traits got clustered together 61% of the times, how strong is that?

# %%
df = coassoc_matrix.where(np.triu(np.ones(coassoc_matrix.shape)).astype(np.bool))
df = df.stack().reset_index()

coassoc_matrix_stats = df[0].describe(
    percentiles=[0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99]
)

# %%
coassoc_matrix_stats.apply(str)

# %% [markdown]
# On average, a pair of clusters appear together in 45% of the clusters in the ensemble (the median is 48%). That makes sense, since for some partitions the resolution (number of clusters) might not be enough to get smaller clusters.

# %% [markdown]
# # Plot coassociation values

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


# %%
k = 5
display(HTML(f"<h2>k: {k}</h2>"))
display(best_partitions.loc[k])

part = best_partitions.loc[k, "partition"]
part_stats = pd.Series(part).value_counts()
display(part_stats)

# %%
plot_cluster(data_umap, part, 4)

# %% [markdown]
# The plot above shows that these 8 keratometry measurements (such as 3mm weak meridian left) were always clustered together in all partitions of the ensemble, representing a very strong/stable grouping.

# %%
plot_cluster(data_umap, part, 2, figsize=(10, 10))

# %% [markdown]
# The "heel bone mineral density" cluster is not as strong as the keratometry one, since some trait pairs have a coassociation value of 0.75. However, 0.75 is quite higher than the 99 percentile of the coassociation values (which is 0.63).

# %%
