# ---
# jupyter:
#   jupytext:
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

# %% [markdown] papermill={"duration": 0.021203, "end_time": "2021-01-05T21:07:35.872424", "exception": false, "start_time": "2021-01-05T21:07:35.851221", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.015406, "end_time": "2021-01-05T21:07:35.903282", "exception": false, "start_time": "2021-01-05T21:07:35.887876", "status": "completed"} tags=[]
# Runs hierarchical clustering on the pca version of the data.

# %% [markdown] papermill={"duration": 0.015327, "end_time": "2021-01-05T21:07:35.934102", "exception": false, "start_time": "2021-01-05T21:07:35.918775", "status": "completed"} tags=[]
# # Environment variables

# %% papermill={"duration": 0.029439, "end_time": "2021-01-05T21:07:35.978937", "exception": false, "start_time": "2021-01-05T21:07:35.949498", "status": "completed"} tags=[]
from IPython.display import display

import conf

N_JOBS = conf.GENERAL["N_JOBS"]
display(N_JOBS)

# %% papermill={"duration": 0.022624, "end_time": "2021-01-05T21:07:36.017567", "exception": false, "start_time": "2021-01-05T21:07:35.994943", "status": "completed"} tags=[]
# %env MKL_NUM_THREADS=$N_JOBS
# %env OPEN_BLAS_NUM_THREADS=$N_JOBS
# %env NUMEXPR_NUM_THREADS=$N_JOBS
# %env OMP_NUM_THREADS=$N_JOBS

# %% [markdown] papermill={"duration": 0.015855, "end_time": "2021-01-05T21:07:36.050032", "exception": false, "start_time": "2021-01-05T21:07:36.034177", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.026927, "end_time": "2021-01-05T21:07:36.092839", "exception": false, "start_time": "2021-01-05T21:07:36.065912", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 1.595612, "end_time": "2021-01-05T21:07:37.704586", "exception": false, "start_time": "2021-01-05T21:07:36.108974", "status": "completed"} tags=[]
from pathlib import Path

import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns

from utils import generate_result_set_name

# %% [markdown] papermill={"duration": 0.015571, "end_time": "2021-01-05T21:07:37.737555", "exception": false, "start_time": "2021-01-05T21:07:37.721984", "status": "completed"} tags=[]
# # Settings

# %% [markdown] papermill={"duration": 0.015677, "end_time": "2021-01-05T21:07:37.769070", "exception": false, "start_time": "2021-01-05T21:07:37.753393", "status": "completed"} tags=[]
# ## Input data

# %% papermill={"duration": 0.031144, "end_time": "2021-01-05T21:07:37.817797", "exception": false, "start_time": "2021-01-05T21:07:37.786653", "status": "completed"} tags=[]
INPUT_SUBSET = "pca"

# %% papermill={"duration": 0.03017, "end_time": "2021-01-05T21:07:37.864513", "exception": false, "start_time": "2021-01-05T21:07:37.834343", "status": "completed"} tags=[]
INPUT_STEM = "z_score_std-projection-smultixcan-efo_partial-mashr-zscores"

# %% papermill={"duration": 0.029796, "end_time": "2021-01-05T21:07:37.910883", "exception": false, "start_time": "2021-01-05T21:07:37.881087", "status": "completed"} tags=[]
# parameters of the dimentionality reduction steps
DR_OPTIONS = {
    "n_components": 50,
    "svd_solver": "full",
    "random_state": 0,
}

# %% papermill={"duration": 0.032416, "end_time": "2021-01-05T21:07:37.959618", "exception": false, "start_time": "2021-01-05T21:07:37.927202", "status": "completed"} tags=[]
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

# %% [markdown] papermill={"duration": 0.015874, "end_time": "2021-01-05T21:07:37.992306", "exception": false, "start_time": "2021-01-05T21:07:37.976432", "status": "completed"} tags=[]
# ## Clustering

# %% papermill={"duration": 0.036429, "end_time": "2021-01-05T21:07:38.044685", "exception": false, "start_time": "2021-01-05T21:07:38.008256", "status": "completed"} tags=[]
from sklearn.cluster import AgglomerativeClustering

# %% papermill={"duration": 0.030223, "end_time": "2021-01-05T21:07:38.091340", "exception": false, "start_time": "2021-01-05T21:07:38.061117", "status": "completed"} tags=[]
CLUSTERING_ATTRIBUTES_TO_SAVE = ["n_clusters"]

# %% papermill={"duration": 0.032521, "end_time": "2021-01-05T21:07:38.140514", "exception": false, "start_time": "2021-01-05T21:07:38.107993", "status": "completed"} tags=[]
CLUSTERING_OPTIONS = {}

CLUSTERING_OPTIONS["K_MIN"] = 2
CLUSTERING_OPTIONS["K_MAX"] = 75  # sqrt(3749) + some more to get closer to 295
CLUSTERING_OPTIONS["LINKAGE"] = {"ward", "complete", "average", "single"}
CLUSTERING_OPTIONS["AFFINITY"] = "euclidean"

display(CLUSTERING_OPTIONS)

# %% papermill={"duration": 0.032177, "end_time": "2021-01-05T21:07:38.189457", "exception": false, "start_time": "2021-01-05T21:07:38.157280", "status": "completed"} tags=[]
CLUSTERERS = {}

idx = 0

for k in range(CLUSTERING_OPTIONS["K_MIN"], CLUSTERING_OPTIONS["K_MAX"] + 1):
    for linkage in CLUSTERING_OPTIONS["LINKAGE"]:
        if linkage == "ward":
            affinity = "euclidean"
        else:
            affinity = "precomputed"

        clus = AgglomerativeClustering(
            n_clusters=k,
            affinity=affinity,
            linkage=linkage,
        )

        method_name = type(clus).__name__
        CLUSTERERS[f"{method_name} #{idx}"] = clus

        idx = idx + 1

# %% papermill={"duration": 0.031147, "end_time": "2021-01-05T21:07:38.237316", "exception": false, "start_time": "2021-01-05T21:07:38.206169", "status": "completed"} tags=[]
display(len(CLUSTERERS))

# %% papermill={"duration": 0.033508, "end_time": "2021-01-05T21:07:38.287456", "exception": false, "start_time": "2021-01-05T21:07:38.253948", "status": "completed"} tags=[]
_iter = iter(CLUSTERERS.items())
display(next(_iter))
display(next(_iter))

# %% papermill={"duration": 0.032212, "end_time": "2021-01-05T21:07:38.337405", "exception": false, "start_time": "2021-01-05T21:07:38.305193", "status": "completed"} tags=[]
clustering_method_name = method_name
display(clustering_method_name)

# %% [markdown] papermill={"duration": 0.017213, "end_time": "2021-01-05T21:07:38.372669", "exception": false, "start_time": "2021-01-05T21:07:38.355456", "status": "completed"} tags=[]
# ## Output directory

# %% papermill={"duration": 0.032372, "end_time": "2021-01-05T21:07:38.422011", "exception": false, "start_time": "2021-01-05T21:07:38.389639", "status": "completed"} tags=[]
# output dir for this notebook
RESULTS_DIR = Path(
    conf.RESULTS["CLUSTERING_RUNS_DIR"],
    f"{INPUT_SUBSET}-{INPUT_STEM}",
).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %% [markdown] papermill={"duration": 0.017277, "end_time": "2021-01-05T21:07:38.457274", "exception": false, "start_time": "2021-01-05T21:07:38.439997", "status": "completed"} tags=[]
# # Load input file

# %% papermill={"duration": 0.033563, "end_time": "2021-01-05T21:07:38.508051", "exception": false, "start_time": "2021-01-05T21:07:38.474488", "status": "completed"} tags=[]
data = pd.read_pickle(input_filepath)

# %% papermill={"duration": 0.032278, "end_time": "2021-01-05T21:07:38.558121", "exception": false, "start_time": "2021-01-05T21:07:38.525843", "status": "completed"} tags=[]
data.shape

# %% papermill={"duration": 0.044841, "end_time": "2021-01-05T21:07:38.620937", "exception": false, "start_time": "2021-01-05T21:07:38.576096", "status": "completed"} tags=[]
data.head()

# %% papermill={"duration": 0.033214, "end_time": "2021-01-05T21:07:38.673210", "exception": false, "start_time": "2021-01-05T21:07:38.639996", "status": "completed"} tags=[]
assert not data.isna().any().any()

# %% [markdown] papermill={"duration": 0.01757, "end_time": "2021-01-05T21:07:38.709002", "exception": false, "start_time": "2021-01-05T21:07:38.691432", "status": "completed"} tags=[]
# # Clustering

# %% [markdown] papermill={"duration": 0.017699, "end_time": "2021-01-05T21:07:38.744469", "exception": false, "start_time": "2021-01-05T21:07:38.726770", "status": "completed"} tags=[]
# ## Generate ensemble

# %% papermill={"duration": 0.037787, "end_time": "2021-01-05T21:07:38.800077", "exception": false, "start_time": "2021-01-05T21:07:38.762290", "status": "completed"} tags=[]
from sklearn.metrics import pairwise_distances
from clustering.ensemble import generate_ensemble

# %% papermill={"duration": 0.19165, "end_time": "2021-01-05T21:07:39.010164", "exception": false, "start_time": "2021-01-05T21:07:38.818514", "status": "completed"} tags=[]
data_dist = pairwise_distances(data, metric=CLUSTERING_OPTIONS["AFFINITY"])

# %% papermill={"duration": 0.033052, "end_time": "2021-01-05T21:07:39.061617", "exception": false, "start_time": "2021-01-05T21:07:39.028565", "status": "completed"} tags=[]
data_dist.shape

# %% papermill={"duration": 0.624686, "end_time": "2021-01-05T21:07:39.704955", "exception": false, "start_time": "2021-01-05T21:07:39.080269", "status": "completed"} tags=[]
pd.Series(data_dist.flatten()).describe().apply(str)

# %% papermill={"duration": 1657.507895, "end_time": "2021-01-05T21:35:17.232206", "exception": false, "start_time": "2021-01-05T21:07:39.724311", "status": "completed"} tags=[]
ensemble = generate_ensemble(
    data_dist,
    CLUSTERERS,
    attributes=CLUSTERING_ATTRIBUTES_TO_SAVE,
    affinity_matrix=data_dist,
)

# %% papermill={"duration": 0.087199, "end_time": "2021-01-05T21:35:17.405133", "exception": false, "start_time": "2021-01-05T21:35:17.317934", "status": "completed"} tags=[]
# the number should be close to 295 (the number of partitions generated by k-means/spectral clustering)
ensemble.shape

# %% papermill={"duration": 0.088053, "end_time": "2021-01-05T21:35:17.559830", "exception": false, "start_time": "2021-01-05T21:35:17.471777", "status": "completed"} tags=[]
ensemble.head()

# %% papermill={"duration": 0.08374, "end_time": "2021-01-05T21:35:17.710662", "exception": false, "start_time": "2021-01-05T21:35:17.626922", "status": "completed"} tags=[]
ensemble["n_clusters"].value_counts().head()

# %% papermill={"duration": 0.085572, "end_time": "2021-01-05T21:35:17.863187", "exception": false, "start_time": "2021-01-05T21:35:17.777615", "status": "completed"} tags=[]
ensemble_stats = ensemble["n_clusters"].describe()
display(ensemble_stats)

# %% [markdown] papermill={"duration": 0.066964, "end_time": "2021-01-05T21:35:17.997629", "exception": false, "start_time": "2021-01-05T21:35:17.930665", "status": "completed"} tags=[]
# ### Testing

# %% papermill={"duration": 0.081343, "end_time": "2021-01-05T21:35:18.145468", "exception": false, "start_time": "2021-01-05T21:35:18.064125", "status": "completed"} tags=[]
assert ensemble_stats["min"] > 1

# %% papermill={"duration": 0.081259, "end_time": "2021-01-05T21:35:18.294359", "exception": false, "start_time": "2021-01-05T21:35:18.213100", "status": "completed"} tags=[]
assert not ensemble["n_clusters"].isna().any()

# %% papermill={"duration": 0.083742, "end_time": "2021-01-05T21:35:18.444404", "exception": false, "start_time": "2021-01-05T21:35:18.360662", "status": "completed"} tags=[]
assert ensemble.shape[0] == len(CLUSTERERS)

# %% papermill={"duration": 0.099681, "end_time": "2021-01-05T21:35:18.611793", "exception": false, "start_time": "2021-01-05T21:35:18.512112", "status": "completed"} tags=[]
# all partitions have the right size
assert np.all(
    [part["partition"].shape[0] == data.shape[0] for idx, part in ensemble.iterrows()]
)

# %% papermill={"duration": 0.102204, "end_time": "2021-01-05T21:35:18.781115", "exception": false, "start_time": "2021-01-05T21:35:18.678911", "status": "completed"} tags=[]
# no partition has negative clusters (noisy points)
assert not np.any([(part["partition"] < 0).any() for idx, part in ensemble.iterrows()])

# %% [markdown] papermill={"duration": 0.067145, "end_time": "2021-01-05T21:35:18.915788", "exception": false, "start_time": "2021-01-05T21:35:18.848643", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.082162, "end_time": "2021-01-05T21:35:19.064231", "exception": false, "start_time": "2021-01-05T21:35:18.982069", "status": "completed"} tags=[]
del CLUSTERING_OPTIONS["LINKAGE"]

output_filename = Path(
    RESULTS_DIR,
    generate_result_set_name(
        CLUSTERING_OPTIONS,
        prefix=f"{clustering_method_name}-",
        suffix=".pkl",
    ),
).resolve()
display(output_filename)

# %% papermill={"duration": 0.088177, "end_time": "2021-01-05T21:35:19.219941", "exception": false, "start_time": "2021-01-05T21:35:19.131764", "status": "completed"} tags=[]
ensemble.to_pickle(output_filename)

# %% papermill={"duration": 0.066448, "end_time": "2021-01-05T21:35:19.353814", "exception": false, "start_time": "2021-01-05T21:35:19.287366", "status": "completed"} tags=[]
