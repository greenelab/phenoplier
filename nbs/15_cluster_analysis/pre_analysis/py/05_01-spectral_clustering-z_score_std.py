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
# This notebook runs some pre-analyses using spectral clustering to explore the best set of parameters to cluster `z_score_std` data version.

# %% [markdown] tags=[]
# # Environment variables

# %% tags=[]
from IPython.display import display

import conf

N_JOBS = conf.GENERAL["N_JOBS"]
display(N_JOBS)

# %% tags=[]
# %env MKL_NUM_THREADS=$N_JOBS
# %env OPEN_BLAS_NUM_THREADS=$N_JOBS
# %env NUMEXPR_NUM_THREADS=$N_JOBS
# %env OMP_NUM_THREADS=$N_JOBS

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import generate_result_set_name

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
INITIAL_RANDOM_STATE = 30000

# %% [markdown] tags=[]
# # Z-score standardized data

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
# # Clustering

# %% tags=[]
from sklearn.cluster import SpectralClustering

from clustering.utils import compute_performance

# %% [markdown] tags=[]
# ## `gamma` parameter

# %% [markdown] tags=[]
# ### Using default value (`gamma=1.0`)

# %% tags=[]
with warnings.catch_warnings():
    warnings.filterwarnings("always")

    clus = SpectralClustering(
        eigen_solver="arpack",
        #         eigen_tol=1e-2,
        n_clusters=2,
        n_init=1,
        affinity="rbf",
        gamma=1.00,
        random_state=INITIAL_RANDOM_STATE,
    )

    part = clus.fit_predict(data)

# %% tags=[]
# show number of clusters and their size
_tmp = pd.Series(part).value_counts()
display(_tmp)
assert _tmp.shape[0] == 1

# %% [markdown] tags=[]
# The algorithm does not work with the default `gamma=1.0`. Other values for this parameter should be explored.

# %% [markdown] tags=[]
# ### Using `gamma=5.00`

# %% tags=[]
with warnings.catch_warnings():
    warnings.filterwarnings("always")

    clus = SpectralClustering(
        eigen_solver="arpack",
        #         eigen_tol=1e-2,
        n_clusters=2,
        n_init=1,
        affinity="rbf",
        gamma=5.00,
        random_state=INITIAL_RANDOM_STATE,
    )

    part = clus.fit_predict(data)

# %% tags=[]
# show number of clusters and their size
_tmp = pd.Series(part).value_counts()
display(_tmp)
assert _tmp.shape[0] == 1

# %% [markdown] tags=[]
# The algorithm does not work either with `gamma>1.0`.

# %% [markdown] tags=[]
# ### Using `gamma=0.01`

# %% tags=[]
with warnings.catch_warnings():
    warnings.filterwarnings("always")

    clus = SpectralClustering(
        eigen_solver="arpack",
        eigen_tol=1e-3,
        n_clusters=2,
        n_init=1,
        affinity="rbf",
        gamma=0.01,
        random_state=INITIAL_RANDOM_STATE,
    )

    part = clus.fit_predict(data)

# %% tags=[]
# show number of clusters and their size
_tmp = pd.Series(part).value_counts()
display(_tmp)
assert _tmp.shape[0] == 2
assert 3 < _tmp.loc[1] < 10

# %% tags=[]
# show some clustering performance measures to assess the quality of the partition
_tmp = compute_performance(data, part)
assert 0.30 < _tmp["si"] < 0.31
assert 10.0 < _tmp["ch"] < 11.00
assert 1.30 < _tmp["db"] < 1.50

# %% [markdown] tags=[]
# For values around `gamma=0.01` the algorithm takes a lot of time to converge (here I used `eigen_tol=1e-03` to force convergence).

# %% [markdown] tags=[]
# ### Using `gamma=0.001`

# %% tags=[]
with warnings.catch_warnings():
    warnings.filterwarnings("always")

    clus = SpectralClustering(
        eigen_solver="arpack",
        #         eigen_tol=1e-3,
        n_clusters=2,
        n_init=1,
        affinity="rbf",
        gamma=0.001,
        random_state=INITIAL_RANDOM_STATE,
    )

    part = clus.fit_predict(data)

# %% tags=[]
# show number of clusters and their size
_tmp = pd.Series(part).value_counts()
display(_tmp)
assert _tmp.shape[0] == 2
assert 80 < _tmp.loc[1] < 90

# %% tags=[]
# show some clustering performance measures to assess the quality of the partition
_tmp = compute_performance(data, part)
assert 0.10 < _tmp["si"] < 0.16
assert 40.0 < _tmp["ch"] < 42.00
assert 3.00 < _tmp["db"] < 4.00

# %% [markdown] tags=[]
# For values around `gamma=0.001` now the algorithm converges, although most of the performance measures are worse. This suggests smaller values should be explored for this parameter.

# %% [markdown] tags=[]
# ## Extended test

# %% [markdown] tags=[]
# Here I run some test across several `k` and `gamma` values; then I check how results perform with different clustering quality measures.

# %% tags=[]
CLUSTERING_OPTIONS = {}

CLUSTERING_OPTIONS["K_RANGE"] = [2, 4, 6, 8, 10, 20, 30, 40, 50, 60]
CLUSTERING_OPTIONS["N_REPS_PER_K"] = 5
CLUSTERING_OPTIONS["KMEANS_N_INIT"] = 10
CLUSTERING_OPTIONS["GAMMAS"] = [
    1e-03,
    #     1e-04,
    #     1e-05,
    1e-05,
    #     1e-06,
    #     1e-07,
    #     1e-08,
    #     1e-09,
    1e-10,
    #     1e-11,
    #     1e-12,
    #     1e-13,
    #     1e-14,
    1e-15,
    1e-17,
    1e-20,
    1e-30,
    1e-40,
    1e-50,
]
CLUSTERING_OPTIONS["AFFINITY"] = "rbf"

display(CLUSTERING_OPTIONS)

# %% tags=[]
CLUSTERERS = {}

idx = 0
random_state = INITIAL_RANDOM_STATE

for k in CLUSTERING_OPTIONS["K_RANGE"]:
    for gamma_value in CLUSTERING_OPTIONS["GAMMAS"]:
        for i in range(CLUSTERING_OPTIONS["N_REPS_PER_K"]):
            clus = SpectralClustering(
                eigen_solver="arpack",
                n_clusters=k,
                n_init=CLUSTERING_OPTIONS["KMEANS_N_INIT"],
                affinity=CLUSTERING_OPTIONS["AFFINITY"],
                gamma=gamma_value,
                random_state=random_state,
            )

            method_name = type(clus).__name__
            CLUSTERERS[f"{method_name} #{idx}"] = clus

            random_state = random_state + 1
            idx = idx + 1

# %% tags=[]
display(len(CLUSTERERS))

# %% tags=[]
_iter = iter(CLUSTERERS.items())
display(next(_iter))
display(next(_iter))

# %% tags=[]
clustering_method_name = method_name
display(clustering_method_name)

# %% [markdown] tags=[]
# ## Generate ensemble

# %% tags=[]
import tempfile
from clustering.ensembles.utils import generate_ensemble

# %% tags=[]
ensemble = generate_ensemble(
    data,
    CLUSTERERS,
    attributes=["n_clusters", "gamma"],
)

# %% tags=[]
ensemble.shape

# %% tags=[]
ensemble.head()

# %% tags=[]
ensemble["gamma"] = ensemble["gamma"].apply(lambda x: f"{x:.1e}")

# %% tags=[]
ensemble["n_clusters"].value_counts()

# %% tags=[]
_tmp = ensemble["n_clusters"].value_counts().unique()
assert _tmp.shape[0] == 1
assert _tmp[0] == int(
    CLUSTERING_OPTIONS["N_REPS_PER_K"] * len(CLUSTERING_OPTIONS["GAMMAS"])
)

# %% tags=[]
ensemble_stats = ensemble["n_clusters"].describe()
display(ensemble_stats)

# %% [markdown] tags=[]
# ## Testing

# %% tags=[]
assert ensemble_stats["min"] > 1

# %% tags=[]
assert not ensemble["n_clusters"].isna().any()

# %% tags=[]
assert ensemble.shape[0] == len(CLUSTERERS)

# %% tags=[]
# all partitions have the right size
assert np.all(
    [part["partition"].shape[0] == data.shape[0] for idx, part in ensemble.iterrows()]
)

# %% tags=[]
# no partition has negative clusters (noisy points)
assert not np.any([(part["partition"] < 0).any() for idx, part in ensemble.iterrows()])
assert not np.any(
    [pd.Series(part["partition"]).isna().any() for idx, part in ensemble.iterrows()]
)

# %% tags=[]
# check that the number of clusters in the partitions are the expected ones
_real_k_values = ensemble["partition"].apply(lambda x: np.unique(x).shape[0])
display(_real_k_values)
assert np.all(ensemble["n_clusters"].values == _real_k_values.values)

# %% [markdown] tags=[]
# ## Add clustering quality measures

# %% tags=[]
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

# %% tags=[]
ensemble = ensemble.assign(
    si_score=ensemble["partition"].apply(lambda x: silhouette_score(data, x)),
    ch_score=ensemble["partition"].apply(lambda x: calinski_harabasz_score(data, x)),
    db_score=ensemble["partition"].apply(lambda x: davies_bouldin_score(data, x)),
)

# %% tags=[]
ensemble.shape

# %% tags=[]
ensemble.head()

# %% [markdown] tags=[]
# # Cluster quality

# %% tags=[]
with pd.option_context("display.max_rows", None, "display.max_columns", None):
    _df = ensemble.groupby(["n_clusters", "gamma"]).mean()
    display(_df)

# %% tags=[]
with sns.plotting_context("talk", font_scale=0.75), sns.axes_style(
    "whitegrid", {"grid.linestyle": "--"}
):
    fig = plt.figure(figsize=(14, 6))
    ax = sns.pointplot(data=ensemble, x="n_clusters", y="si_score", hue="gamma")
    ax.set_ylabel("Silhouette index\n(higher is better)")
    ax.set_xlabel("Number of clusters ($k$)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.grid(True)
    plt.tight_layout()

# %% tags=[]
with sns.plotting_context("talk", font_scale=0.75), sns.axes_style(
    "whitegrid", {"grid.linestyle": "--"}
):
    fig = plt.figure(figsize=(14, 6))
    ax = sns.pointplot(data=ensemble, x="n_clusters", y="ch_score", hue="gamma")
    ax.set_ylabel("Calinski-Harabasz index\n(higher is better)")
    ax.set_xlabel("Number of clusters ($k$)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.grid(True)
    plt.tight_layout()

# %% tags=[]
with sns.plotting_context("talk", font_scale=0.75), sns.axes_style(
    "whitegrid", {"grid.linestyle": "--"}
):
    fig = plt.figure(figsize=(14, 6))
    ax = sns.pointplot(data=ensemble, x="n_clusters", y="db_score", hue="gamma")
    ax.set_ylabel("Davies-Bouldin index\n(lower is better)")
    ax.set_xlabel("Number of clusters ($k$)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.grid(True)
    plt.tight_layout()

# %% [markdown] tags=[]
# # Stability

# %% [markdown] tags=[]
# ## Group ensemble by n_clusters

# %% tags=[]
parts = ensemble.groupby(["gamma", "n_clusters"]).apply(
    lambda x: np.concatenate(x["partition"].apply(lambda x: x.reshape(1, -1)), axis=0)
)

# %% tags=[]
parts.shape

# %% tags=[]
parts.head()

# %% tags=[]
parts.iloc[0].shape

# %% tags=[]
assert np.all(
    [
        parts.loc[k].shape == (int(CLUSTERING_OPTIONS["N_REPS_PER_K"]), data.shape[0])
        for k in parts.index
    ]
)

# %% [markdown] tags=[]
# ## Compute stability

# %% tags=[]
from sklearn.metrics import adjusted_rand_score as ari
from scipy.spatial.distance import pdist

# %% tags=[]
parts_ari = pd.Series(
    {k: pdist(parts.loc[k], metric=ari) for k in parts.index}, name="n_clusters"
)

# %% tags=[]
parts_ari_stability = parts_ari.apply(lambda x: x.mean())
display(parts_ari_stability.sort_values(ascending=False).head(15))

# %% tags=[]
parts_ari_df = pd.DataFrame.from_records(parts_ari.tolist()).set_index(
    parts_ari.index.copy()
)
parts_ari_df.index.rename(["gamma", "n_clusters"], inplace=True)

# %% tags=[]
parts_ari_df.shape

# %% tags=[]
_n_total_parts = int(
    CLUSTERING_OPTIONS["N_REPS_PER_K"]
)  # * len(CLUSTERING_OPTIONS["GAMMAS"]))

assert int(_n_total_parts * (_n_total_parts - 1) / 2) == parts_ari_df.shape[1]

# %% tags=[]
parts_ari_df.head()

# %% [markdown] tags=[]
# ## Stability plot

# %% tags=[]
parts_ari_df_plot = (
    parts_ari_df.stack().reset_index().rename(columns={"level_2": "idx", 0: "ari"})
)

# %% tags=[]
parts_ari_df_plot.dtypes

# %% tags=[]
parts_ari_df_plot.head()

# %% tags=[]
with pd.option_context("display.max_rows", None, "display.max_columns", None):
    _df = parts_ari_df_plot.groupby(["n_clusters", "gamma"]).mean()
    display(_df)

# %% tags=[]
with sns.plotting_context("talk", font_scale=0.75), sns.axes_style(
    "whitegrid", {"grid.linestyle": "--"}
):
    fig = plt.figure(figsize=(14, 6))
    ax = sns.pointplot(data=parts_ari_df_plot, x="n_clusters", y="ari", hue="gamma")
    ax.set_ylabel("Averange ARI")
    ax.set_xlabel("Number of clusters ($k$)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.grid(True)
    plt.tight_layout()

# %% [markdown] tags=[]
# # Conclusions

# %% [markdown] tags=[]
# **UPDATE** We choose `1e-10` as the `gamma` parameter for this data version.

# %% tags=[]
