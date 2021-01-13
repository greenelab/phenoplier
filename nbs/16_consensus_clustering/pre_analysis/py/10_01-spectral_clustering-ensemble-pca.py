# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill
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
# See description in notebook `10_00-spectral_clustering...`.

# %% [markdown] tags=[]
# # Environment variables

# %% tags=[] trusted=true
from IPython.display import display

import conf

N_JOBS = conf.GENERAL["N_JOBS"]
display(N_JOBS)

# %% tags=[] trusted=true
# %env MKL_NUM_THREADS=$N_JOBS
# %env OPEN_BLAS_NUM_THREADS=$N_JOBS
# %env NUMEXPR_NUM_THREADS=$N_JOBS
# %env OMP_NUM_THREADS=$N_JOBS

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[] trusted=true
# %load_ext autoreload
# %autoreload 2

# %% tags=[] trusted=true
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import generate_result_set_name

# %% [markdown] tags=[]
# # Settings

# %% tags=[] trusted=true
INITIAL_RANDOM_STATE = 100000

# %% trusted=true
CLUSTERING_METHOD_NAME = "DeltaSpectralClustering"

# %% trusted=true
# output dir for this notebook
CONSENSUS_CLUSTERING_DIR = Path(
    conf.RESULTS["CLUSTERING_DIR"], "consensus_clustering"
).resolve()

display(CONSENSUS_CLUSTERING_DIR)

# %% [markdown]
# # Load data

# %% tags=[] trusted=true
INPUT_SUBSET = "pca"

# %% tags=[] trusted=true
INPUT_STEM = "z_score_std-projection-smultixcan-efo_partial-mashr-zscores"

# %% tags=[] trusted=true
DR_OPTIONS = {
    "n_components": 50,
    "svd_solver": "full",
    "random_state": 0,
}

# %% tags=[] trusted=true
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

# %% tags=[] trusted=true
data = pd.read_pickle(input_filepath)

# %% tags=[] trusted=true
data.shape

# %% tags=[] trusted=true
data.head()

# %% trusted=true
traits = data.index.tolist()

# %% trusted=true
len(traits)

# %% [markdown] tags=[]
# # Ensemble (coassociation matrix)

# %% trusted=true
input_file = Path(CONSENSUS_CLUSTERING_DIR, "ensemble_coassoc_matrix.npy").resolve()
display(input_file)

# %% trusted=true
coassoc_matrix = np.load(input_file)

# %% trusted=true
coassoc_matrix = pd.DataFrame(
    data=coassoc_matrix,
    index=traits,
    columns=traits,
)

# %% trusted=true
coassoc_matrix.shape

# %% trusted=true
coassoc_matrix.head()

# %% trusted=true
dist_matrix = coassoc_matrix

# %% [markdown] tags=[]
# # Clustering

# %% tags=[] trusted=true
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

# %% [markdown] tags=[]
# ## Extended test

# %% tags=[] trusted=true
CLUSTERING_OPTIONS = {}

CLUSTERING_OPTIONS["K_RANGE"] = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40]
CLUSTERING_OPTIONS["N_REPS_PER_K"] = 5
CLUSTERING_OPTIONS["KMEANS_N_INIT"] = 10
CLUSTERING_OPTIONS["DELTAS"] = [
    5.00,
    2.00,
    1.00,
    0.90,
    0.75,
    0.50,
    0.30,
    0.25,
    0.20,
]

display(CLUSTERING_OPTIONS)

# %% [markdown] tags=[]
# ## Generate ensemble

# %% tags=[] trusted=true
import tempfile

# %% tags=[] trusted=true
ensemble_folder = Path(
    tempfile.gettempdir(),
    f"pre_cluster_analysis",
    CLUSTERING_METHOD_NAME,
).resolve()
ensemble_folder.mkdir(parents=True, exist_ok=True)

# %% tags=[] trusted=true
ensemble_file = Path(
    ensemble_folder,
    generate_result_set_name(CLUSTERING_OPTIONS, prefix=f"ensemble-", suffix=".pkl"),
)
display(ensemble_file)

# %% trusted=true
assert ensemble_file.exists(), "Ensemble file does not exists"

# %% tags=[] trusted=true
ensemble = pd.read_pickle(ensemble_file)

# %% tags=[] trusted=true
ensemble.shape

# %% tags=[] trusted=true
ensemble.head()

# %% [markdown] tags=[]
# ### Add clustering quality measures

# %% tags=[] trusted=true
ensemble = ensemble.assign(
    #     si_score=ensemble["partition"].apply(lambda x: silhouette_score(dist_matrix, x, metric="precomputed")),
    ch_score=ensemble["partition"].apply(lambda x: calinski_harabasz_score(data, x)),
    db_score=ensemble["partition"].apply(lambda x: davies_bouldin_score(data, x)),
)

# %% tags=[] trusted=true
ensemble.shape

# %% tags=[] trusted=true
ensemble.head()

# %% [markdown] tags=[]
# # Cluster quality

# %% tags=[] trusted=true
with pd.option_context("display.max_rows", None, "display.max_columns", None):
    _df = ensemble.groupby(["n_clusters", "delta"]).mean()
    display(_df)

# %% tags=[] trusted=true
with sns.plotting_context("talk", font_scale=0.75), sns.axes_style(
    "whitegrid", {"grid.linestyle": "--"}
):
    fig = plt.figure(figsize=(14, 6))
    ax = sns.pointplot(data=ensemble, x="n_clusters", y="si_score", hue="delta")
    ax.set_ylabel("Silhouette index\n(higher is better)")
    ax.set_xlabel("Number of clusters ($k$)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.grid(True)
    plt.tight_layout()

# %% tags=[] trusted=true
with sns.plotting_context("talk", font_scale=0.75), sns.axes_style(
    "whitegrid", {"grid.linestyle": "--"}
):
    fig = plt.figure(figsize=(14, 6))
    ax = sns.pointplot(data=ensemble, x="n_clusters", y="ch_score", hue="delta")
    ax.set_ylabel("Calinski-Harabasz index\n(higher is better)")
    ax.set_xlabel("Number of clusters ($k$)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.grid(True)
    plt.tight_layout()

# %% tags=[] trusted=true
with sns.plotting_context("talk", font_scale=0.75), sns.axes_style(
    "whitegrid", {"grid.linestyle": "--"}
):
    fig = plt.figure(figsize=(14, 6))
    ax = sns.pointplot(data=ensemble, x="n_clusters", y="db_score", hue="delta")
    ax.set_ylabel("Davies-Bouldin index\n(lower is better)")
    ax.set_xlabel("Number of clusters ($k$)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.grid(True)
    plt.tight_layout()

# %% tags=[] trusted=true
