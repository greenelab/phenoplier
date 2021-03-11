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
# It analyzes how clusters of traits were grouped across the ensemble partitions. For example, a stable cluster (obtained from consensus partitions) of cardiovascular diseases can show that all traits were always grouped together across all partitions of the ensemble; another cluster might show that some traits were clustered more often than others, representing a less stable group of traits.

# %% [markdown] tags=[]
# **TODO:** This section of the notebook will be updated again when I start actively writing the results section of the manuscript. Here I left some code as example for some clusters.

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
from IPython.display import display
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
# # Load coassociation matrix

# %% tags=[]
input_file = Path(CONSENSUS_CLUSTERING_DIR, "ensemble_coassoc_matrix.npy").resolve()
display(input_file)

# %% tags=[]
coassoc_matrix = np.load(input_file)

# %% tags=[]
coassoc_matrix = pd.DataFrame(
    data=1.0 - coassoc_matrix,
    index=data_umap.index.copy(),
    columns=data_umap.index.copy(),
)

# %% tags=[]
coassoc_matrix.shape

# %% tags=[]
coassoc_matrix.head()

# %% [markdown] tags=[]
# The coassociation matrix shows the percentage of times a pair of traits was clustered together across the ensemble partitions.

# %% [markdown] tags=[]
# ## Stats

# %% [markdown] tags=[]
# Here I show some general stats of the coassociation matrix, useful to compare results below. For instance, if a pair of traits got clustered together 61% of the times, how strong is that?

# %% tags=[]
df = coassoc_matrix.where(np.triu(np.ones(coassoc_matrix.shape)).astype(np.bool))
df = df.stack().reset_index()

coassoc_matrix_stats = df[0].describe(
    percentiles=[0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99]
)

# %% tags=[]
coassoc_matrix_stats.apply(str)

# %% [markdown] tags=[]
# On average, a pair of clusters appear together in 45% of the clusters in the ensemble (the median is 48%). That makes sense, since for some partitions the resolution (number of clusters) might not be enough to get smaller clusters.

# %% [markdown] tags=[]
# # Plot coassociation values

# %% [markdown] tags=[]
# ## Functions

# %% tags=[]
from IPython.display import HTML


# %% tags=[]
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


# %% tags=[]
k = 5
display(HTML(f"<h2>k: {k}</h2>"))
display(best_partitions.loc[k])

part = best_partitions.loc[k, "partition"]
part_stats = pd.Series(part).value_counts()
display(part_stats)

# %% tags=[]
plot_cluster(data_umap, part, 1)

# %% [markdown] tags=[]
# The plot above shows that these 8 keratometry measurements (such as 3mm weak meridian left) were always clustered together in all partitions of the ensemble, representing a very strong/stable grouping.

# %% tags=[]
plot_cluster(data_umap, part, 3, figsize=(10, 10))

# %% [markdown] tags=[]
# The "heel bone mineral density" cluster is not as strong as the keratometry one, since some trait pairs have a coassociation value of 0.89. However, 0.89 is quite higher than the 99 percentile of the coassociation values (which is 0.69).

# %% tags=[]
