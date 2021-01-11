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
# This notebook runs some pre-analyses using spectral clustering to explore the best set of parameters to cluster `umap` data version.

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
import warnings

import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns

from utils import generate_result_set_name

# %% [markdown] tags=[]
# # Settings

# %% tags=[] trusted=true
INITIAL_RANDOM_STATE = 50000

# %% [markdown] tags=[]
# # UMAP

# %% tags=[] trusted=true
INPUT_SUBSET = "umap"

# %% tags=[] trusted=true
INPUT_STEM = "z_score_std-projection-smultixcan-efo_partial-mashr-zscores"

# %% tags=[] trusted=true
DR_OPTIONS = {
    "n_components": 50,
    "metric": "euclidean",
    "n_neighbors": 15,
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

# %% [markdown] tags=[]
# # Clustering

# %% trusted=true
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# %% [markdown]
# ## `gamma` parameter

# %% [markdown]
# ### Using default value (`gamma=1.0`)

# %% trusted=true
with warnings.catch_warnings():
    warnings.filterwarnings("always")

    clus = SpectralClustering(
        eigen_solver="arpack",
        #         eigen_tol=1e-3,
        n_clusters=2,
        n_init=1,
        affinity="rbf",
        gamma=1.00,
        random_state=INITIAL_RANDOM_STATE,
    )

    part = clus.fit_predict(data)

# %% trusted=true
# show number of clusters and their size
pd.Series(part).value_counts()

# %% trusted=true
# From sklearn website:
# The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters.
# Negative values generally indicate that a sample has been assigned to the wrong cluster,
# as a different cluster is more similar
silhouette_score(data, part)

# %% trusted=true
# From sklearn website:
# The score is defined as ratio between the within-cluster dispersion and the between-cluster dispersion
calinski_harabasz_score(data, part)

# %% [markdown]
# For default values of `gamma` (`1.0`), the algorithm works fine with this data version.

# %% [markdown]
# ### Using `gamma>1.0` (larger than default value)

# %% trusted=true
with warnings.catch_warnings():
    warnings.filterwarnings("always")

    clus = SpectralClustering(
        eigen_solver="arpack",
        #         eigen_tol=1e-3,
        n_clusters=2,
        n_init=1,
        affinity="rbf",
        gamma=2.00,
        random_state=INITIAL_RANDOM_STATE,
    )

    part = clus.fit_predict(data)

# %% trusted=true
# show number of clusters and their size
pd.Series(part).value_counts()

# %% trusted=true
# From sklearn website:
# The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters.
# Negative values generally indicate that a sample has been assigned to the wrong cluster,
# as a different cluster is more similar
silhouette_score(data, part)

# %% trusted=true
# From sklearn website:
# The score is defined as ratio between the within-cluster dispersion and the between-cluster dispersion
calinski_harabasz_score(data, part)

# %% [markdown]
# For `gamma` values largen than the default (`1.0`), quality measures go down.

# %% [markdown]
# ### Using `gamma<1.0` (smaller than default value)

# %% trusted=true
with warnings.catch_warnings():
    warnings.filterwarnings("always")

    clus = SpectralClustering(
        eigen_solver="arpack",
        #         eigen_tol=1e-3,
        n_clusters=2,
        n_init=1,
        affinity="rbf",
        gamma=0.01,
        random_state=INITIAL_RANDOM_STATE,
    )

    part = clus.fit_predict(data)

# %% trusted=true
# show number of clusters and their size
pd.Series(part).value_counts()

# %% trusted=true
# From sklearn website:
# The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters.
# Negative values generally indicate that a sample has been assigned to the wrong cluster,
# as a different cluster is more similar
silhouette_score(data, part)

# %% trusted=true
# From sklearn website:
# The score is defined as ratio between the within-cluster dispersion and the between-cluster dispersion
calinski_harabasz_score(data, part)

# %% [markdown]
# For `gamma` values around `0.01`, clustering quality measures change (CH improves and silhouette diminishes a little bit. I will explore more around these values.

# %% [markdown]
# ## Extended test

# %% tags=[] trusted=true
CLUSTERING_OPTIONS = {}

CLUSTERING_OPTIONS["K_RANGE"] = [2, 4, 6, 8, 10, 20, 30, 40, 50, 60]
CLUSTERING_OPTIONS["N_REPS_PER_K"] = 5
CLUSTERING_OPTIONS["KMEANS_N_INIT"] = 10
CLUSTERING_OPTIONS["GAMMAS"] = [
    1.00,
    1e-02,
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
    #     1e-17,
    1e-20,
    #     1e-30,
    #     1e-40,
    1e-50,
]
CLUSTERING_OPTIONS["AFFINITY"] = "rbf"

display(CLUSTERING_OPTIONS)

# %% tags=[] trusted=true
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

# %% tags=[] trusted=true
display(len(CLUSTERERS))

# %% tags=[] trusted=true
_iter = iter(CLUSTERERS.items())
display(next(_iter))
display(next(_iter))

# %% tags=[] trusted=true
clustering_method_name = method_name
display(clustering_method_name)

# %% [markdown] tags=[]
# ## Generate ensemble

# %% tags=[] trusted=true
import tempfile
from clustering.ensemble import generate_ensemble

# %% trusted=true
# generate a temporary folder where to store the ensemble and avoid computing it again
ensemble_folder = Path(
    tempfile.gettempdir(),
    f"pre_cluster_analysis",
    clustering_method_name,
).resolve()
ensemble_folder.mkdir(parents=True, exist_ok=True)

# %% trusted=true
ensemble_file = Path(
    ensemble_folder,
    generate_result_set_name(
        CLUSTERING_OPTIONS, prefix=f"ensemble-{INPUT_SUBSET}-", suffix=".pkl"
    ),
)
display(ensemble_file)

# %% tags=[] trusted=true
if ensemble_file.exists():
    display(f"Ensemble file exists")
    ensemble = pd.read_pickle(ensemble_file)
else:
    ensemble = generate_ensemble(
        data,
        CLUSTERERS,
        attributes=["n_clusters", "gamma"],
    )

    ensemble.to_pickle(ensemble_file)

# %% tags=[] trusted=true
ensemble.shape

# %% tags=[] trusted=true
ensemble.head()

# %% trusted=true
ensemble["gamma"] = ensemble["gamma"].apply(lambda x: f"{x:.1e}")

# %% tags=[] trusted=true
ensemble["n_clusters"].value_counts()

# %% trusted=true
_tmp = ensemble["n_clusters"].value_counts().unique()
assert _tmp.shape[0] == 1
assert _tmp[0] == int(
    CLUSTERING_OPTIONS["N_REPS_PER_K"] * len(CLUSTERING_OPTIONS["GAMMAS"])
)

# %% tags=[] trusted=true
ensemble_stats = ensemble["n_clusters"].describe()
display(ensemble_stats)

# %% [markdown] tags=[]
# ### Testing

# %% tags=[] trusted=true
assert ensemble_stats["min"] > 1

# %% tags=[] trusted=true
assert not ensemble["n_clusters"].isna().any()

# %% tags=[] trusted=true
assert ensemble.shape[0] == len(CLUSTERERS)

# %% tags=[] trusted=true
# all partitions have the right size
assert np.all(
    [part["partition"].shape[0] == data.shape[0] for idx, part in ensemble.iterrows()]
)

# %% tags=[] trusted=true
# no partition has negative clusters (noisy points)
assert not np.any([(part["partition"] < 0).any() for idx, part in ensemble.iterrows()])

# %% trusted=true
# check that the number of clusters in the partitions are the expected ones
_real_k_values = ensemble["partition"].apply(lambda x: np.unique(x).shape[0])
display(_real_k_values)
assert np.all(ensemble["n_clusters"].values == _real_k_values.values)

# %% [markdown] tags=[]
# ### Add clustering quality measures

# %% trusted=true
ensemble = ensemble.assign(
    ch_score=ensemble["partition"].apply(lambda x: calinski_harabasz_score(data, x))
)

# %% trusted=true
ensemble.shape

# %% trusted=true
ensemble.head()

# %% [markdown] tags=[]
# # Cluster quality

# %% trusted=true
with pd.option_context("display.max_rows", None, "display.max_columns", None):
    _df = ensemble.groupby(["n_clusters", "gamma"]).mean()
    display(_df)

# %% tags=[] trusted=true
# with sns.axes_style('whitegrid', {'grid.linestyle': '--'}):
with sns.plotting_context("talk", font_scale=0.75), sns.axes_style(
    "whitegrid", {"grid.linestyle": "--"}
):
    fig = plt.figure(figsize=(14, 6))
    ax = sns.pointplot(data=ensemble, x="n_clusters", y="ch_score", hue="gamma")
    ax.set_ylabel("Calinski-Harabasz index")
    ax.set_xlabel("Number of clusters ($k$)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    #     ax.set_ylim(0.0, 1.0)
    #     ax.set_xlim(CLUSTERING_OPTIONS['K_MIN'], CLUSTERING_OPTIONS['K_MAX'])
    plt.grid(True)
    plt.tight_layout()

# %% [markdown] tags=[]
# # Stability

# %% [markdown] tags=[]
# ## Group ensemble by n_clusters

# %% tags=[] trusted=true
parts = ensemble.groupby(["gamma", "n_clusters"]).apply(
    lambda x: np.concatenate(x["partition"].apply(lambda x: x.reshape(1, -1)), axis=0)
)

# %% trusted=true
parts.shape

# %% tags=[] trusted=true
parts.head()

# %% trusted=true
parts.iloc[0].shape

# %% tags=[] trusted=true
assert np.all(
    [
        parts.loc[k].shape == (int(CLUSTERING_OPTIONS["N_REPS_PER_K"]), data.shape[0])
        for k in parts.index
    ]
)

# %% [markdown] tags=[]
# ## Compute stability

# %% tags=[] trusted=true
from sklearn.metrics import adjusted_rand_score as ari
from scipy.spatial.distance import squareform, pdist

# %% tags=[] trusted=true
parts_ari = pd.Series(
    {k: pdist(parts.loc[k], metric=ari) for k in parts.index}, name="n_clusters"
)

# %% tags=[] trusted=true
parts_ari_stability = parts_ari.apply(lambda x: x.mean())
display(parts_ari_stability.sort_values(ascending=False).head(15))

# %% tags=[] trusted=true
parts_ari_df = pd.DataFrame.from_records(parts_ari.tolist()).set_index(
    parts_ari.index.copy()
)
parts_ari_df.index.rename(["gamma", "n_clusters"], inplace=True)

# %% tags=[] trusted=true
parts_ari_df.shape

# %% tags=[] trusted=true
_n_total_parts = int(
    CLUSTERING_OPTIONS["N_REPS_PER_K"]
)  # * len(CLUSTERING_OPTIONS["GAMMAS"]))

assert int(_n_total_parts * (_n_total_parts - 1) / 2) == parts_ari_df.shape[1]

# %% tags=[] trusted=true
parts_ari_df.head()

# %% [markdown] tags=[]
# ## Stability plot

# %% tags=[] trusted=true
parts_ari_df_plot = (
    parts_ari_df.stack().reset_index().rename(columns={"level_2": "idx", 0: "ari"})
)

# %% tags=[] trusted=true
parts_ari_df_plot.dtypes

# %% tags=[] trusted=true
parts_ari_df_plot.head()

# %% trusted=true
with pd.option_context("display.max_rows", None, "display.max_columns", None):
    _df = parts_ari_df_plot.groupby(["n_clusters", "gamma"]).mean()
    display(_df)

# %% tags=[] trusted=true
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

# %% tags=[]
