# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
#     formats: ipynb,py//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] tags=[]
# # Description

# %% [markdown] tags=[]
# It runs DBSCAN on the `pca` version of the data.
#
# The notebook explores different values for `min_samples` and `eps` (the main parameters of DBSCAN).

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

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

from utils import generate_result_set_name
from clustering.ensembles.utils import generate_ensemble

# %% [markdown] tags=[]
# # Global settings

# %% tags=[]
np.random.seed(0)

# %% tags=[]
CLUSTERING_ATTRIBUTES_TO_SAVE = ["n_clusters"]

# %%
NULL_DIR = conf.RESULTS["CLUSTERING_NULL_DIR"] / "shuffle_lvs"

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
# these parameter values are taken from the pre-analysis notebook for this clustering method and data version
k_values = np.arange(2, 125 + 1, 1)

eps_range_per_k = {
    k: (10, 20)
    if k < 5
    else (11, 25)
    if k < 10
    else (12, 30)
    if k < 15
    else (13, 35)
    if k < 20
    else (14, 40)
    for k in k_values
}

# %% [markdown] tags=[]
# # Data version: pca

# %% [markdown] tags=[]
# ## Settings

# %% tags=[]
INPUT_SUBSET = "pca"

# %% tags=[]
INPUT_STEM = "z_score_std-projection-smultixcan-efo_partial-mashr-zscores"

# %% tags=[]
# parameters of the dimentionality reduction steps
DR_OPTIONS = {
    "n_components": 50,
    "svd_solver": "full",
    "random_state": 0,
}

# %% tags=[]
input_filepath = Path(
    NULL_DIR,
    "data_transformations",
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
# output dir for this notebook
RESULTS_DIR = Path(
    NULL_DIR,
    "runs",
    f"{INPUT_SUBSET}-{INPUT_STEM}",
).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %% [markdown] tags=[]
# ## Load input file

# %% tags=[]
data = pd.read_pickle(input_filepath)

# %% tags=[]
data.shape

# %% tags=[]
data.head()

# %% tags=[]
assert not data.isna().any().any()

# %% [markdown] tags=[]
# ## Clustering

# %% [markdown] tags=[]
# ### Generate clusterers

# %% tags=[]
CLUSTERING_OPTIONS = {}

# K_RANGE is the min_samples parameter in DBSCAN (sklearn)
CLUSTERING_OPTIONS["K_RANGE"] = k_values
CLUSTERING_OPTIONS["EPS_RANGE_PER_K"] = eps_range_per_k
CLUSTERING_OPTIONS["EPS_STEP"] = 33
CLUSTERING_OPTIONS["METRIC"] = "euclidean"

display(CLUSTERING_OPTIONS)

# %% tags=[]
CLUSTERERS = {}

idx = 0

for k in CLUSTERING_OPTIONS["K_RANGE"]:
    eps_range = CLUSTERING_OPTIONS["EPS_RANGE_PER_K"][k]
    eps_values = np.linspace(eps_range[0], eps_range[1], CLUSTERING_OPTIONS["EPS_STEP"])

    for eps in eps_values:
        clus = DBSCAN(min_samples=k, eps=eps, metric="precomputed", n_jobs=N_JOBS)

        method_name = type(clus).__name__
        CLUSTERERS[f"{method_name} #{idx}"] = clus

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
# ### Generate ensemble

# %% tags=[]
data_dist = pairwise_distances(data, metric=CLUSTERING_OPTIONS["METRIC"])

# %% tags=[]
data_dist.shape

# %% tags=[]
pd.Series(data_dist.flatten()).describe().apply(str)

# %% tags=[]
ensemble = generate_ensemble(
    data_dist,
    CLUSTERERS,
    attributes=CLUSTERING_ATTRIBUTES_TO_SAVE,
)

# %% tags=[]
ensemble.shape

# %% tags=[]
ensemble.head()

# %% tags=[]
ensemble["n_clusters"].value_counts().head()

# %% tags=[]
ensemble_stats = ensemble["n_clusters"].describe()
display(ensemble_stats)

# %% tags=[]
assert (
    ensemble.shape[0] > 0
), "Ensemble is empty, stopping here (this is not actually an error if running null simulations)"

# %% [markdown] tags=[]
# ### Testing

# %% tags=[]
assert ensemble_stats["min"] > 1

# %% tags=[]
assert not ensemble["n_clusters"].isna().any()

# %% tags=[]
# assert ensemble.shape[0] == len(CLUSTERERS)

# %% tags=[]
# all partitions have the right size
assert np.all(
    [part["partition"].shape[0] == data.shape[0] for idx, part in ensemble.iterrows()]
)

# %% tags=[]
# no partition has negative clusters (noisy points)
assert not np.any([(part["partition"] < 0).any() for idx, part in ensemble.iterrows()])

# %% [markdown] tags=[]
# ### Save

# %% tags=[]
output_filename = Path(
    RESULTS_DIR,
    generate_result_set_name(
        {},
        #         CLUSTERING_OPTIONS,
        prefix=f"{clustering_method_name}",
        suffix=".pkl",
    ),
).resolve()
display(output_filename)

# %% tags=[]
ensemble.to_pickle(output_filename)

# %% tags=[]
