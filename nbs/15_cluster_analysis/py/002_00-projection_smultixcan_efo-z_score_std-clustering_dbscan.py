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

# %% [markdown] papermill={"duration": 0.017167, "end_time": "2020-12-02T17:42:55.923348", "exception": false, "start_time": "2020-12-02T17:42:55.906181", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.014318, "end_time": "2020-12-02T17:42:55.952589", "exception": false, "start_time": "2020-12-02T17:42:55.938271", "status": "completed"} tags=[]
# It runs DBSCAN on the z_score_std version of the data.
#
# The notebook explores different values for min_samples and eps (the main parameters of DBSCAN).

# %% [markdown] papermill={"duration": 0.013362, "end_time": "2020-12-02T17:42:55.979550", "exception": false, "start_time": "2020-12-02T17:42:55.966188", "status": "completed"} tags=[]
# # Environment variables

# %% papermill={"duration": 0.027188, "end_time": "2020-12-02T17:42:56.020167", "exception": false, "start_time": "2020-12-02T17:42:55.992979", "status": "completed"} tags=[]
from IPython.display import display

import conf

N_JOBS = conf.GENERAL["N_JOBS"]
display(N_JOBS)

# %% papermill={"duration": 0.020742, "end_time": "2020-12-02T17:42:56.055898", "exception": false, "start_time": "2020-12-02T17:42:56.035156", "status": "completed"} tags=[]
# %env MKL_NUM_THREADS=$N_JOBS
# %env OPEN_BLAS_NUM_THREADS=$N_JOBS
# %env NUMEXPR_NUM_THREADS=$N_JOBS
# %env OMP_NUM_THREADS=$N_JOBS

# %% [markdown] papermill={"duration": 0.013847, "end_time": "2020-12-02T17:42:56.084279", "exception": false, "start_time": "2020-12-02T17:42:56.070432", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.024515, "end_time": "2020-12-02T17:42:56.122812", "exception": false, "start_time": "2020-12-02T17:42:56.098297", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 1.612315, "end_time": "2020-12-02T17:42:57.749575", "exception": false, "start_time": "2020-12-02T17:42:56.137260", "status": "completed"} tags=[]
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.cluster import DBSCAN
import umap
import matplotlib.pyplot as plt
import seaborn as sns

from utils import generate_result_set_name
from clustering.ensemble import generate_ensemble

# %% [markdown] papermill={"duration": 0.014513, "end_time": "2020-12-02T17:42:57.780029", "exception": false, "start_time": "2020-12-02T17:42:57.765516", "status": "completed"} tags=[]
# # Global settings

# %% papermill={"duration": 0.028119, "end_time": "2020-12-02T17:42:57.822303", "exception": false, "start_time": "2020-12-02T17:42:57.794184", "status": "completed"} tags=[]
CLUSTERING_ATTRIBUTES_TO_SAVE = ["n_clusters"]

# %% [markdown] papermill={"duration": 0.014454, "end_time": "2020-12-02T17:42:57.851807", "exception": false, "start_time": "2020-12-02T17:42:57.837353", "status": "completed"} tags=[]
# # Data version: z_score_std

# %% [markdown] papermill={"duration": 0.013906, "end_time": "2020-12-02T17:42:57.879622", "exception": false, "start_time": "2020-12-02T17:42:57.865716", "status": "completed"} tags=[]
# ## Settings

# %% papermill={"duration": 0.029772, "end_time": "2020-12-02T17:42:57.923248", "exception": false, "start_time": "2020-12-02T17:42:57.893476", "status": "completed"} tags=[]
INPUT_SUBSET = "z_score_std"

# %% papermill={"duration": 0.028578, "end_time": "2020-12-02T17:42:57.966993", "exception": false, "start_time": "2020-12-02T17:42:57.938415", "status": "completed"} tags=[]
INPUT_STEM = "projection-smultixcan-efo_partial-mashr-zscores"

# %% papermill={"duration": 0.030394, "end_time": "2020-12-02T17:42:58.012539", "exception": false, "start_time": "2020-12-02T17:42:57.982145", "status": "completed"} tags=[]
input_filepath = Path(
    conf.RESULTS["DATA_TRANSFORMATIONS_DIR"],
    INPUT_SUBSET,
    f"{INPUT_SUBSET}-{INPUT_STEM}.pkl",
).resolve()
display(input_filepath)

assert input_filepath.exists(), "Input file does not exist"

input_filepath_stem = input_filepath.stem
display(input_filepath_stem)

# %% papermill={"duration": 0.030302, "end_time": "2020-12-02T17:42:58.058376", "exception": false, "start_time": "2020-12-02T17:42:58.028074", "status": "completed"} tags=[]
# output dir for this notebook
RESULTS_DIR = Path(
    conf.RESULTS["CLUSTERING_RUNS_DIR"],
    f"{INPUT_SUBSET}-{INPUT_STEM}",
).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %% [markdown] papermill={"duration": 0.015076, "end_time": "2020-12-02T17:42:58.089515", "exception": false, "start_time": "2020-12-02T17:42:58.074439", "status": "completed"} tags=[]
# ## Load input file

# %% papermill={"duration": 0.041595, "end_time": "2020-12-02T17:42:58.145969", "exception": false, "start_time": "2020-12-02T17:42:58.104374", "status": "completed"} tags=[]
data = pd.read_pickle(input_filepath)

# %% papermill={"duration": 0.031251, "end_time": "2020-12-02T17:42:58.193994", "exception": false, "start_time": "2020-12-02T17:42:58.162743", "status": "completed"} tags=[]
data.shape

# %% papermill={"duration": 0.04381, "end_time": "2020-12-02T17:42:58.254275", "exception": false, "start_time": "2020-12-02T17:42:58.210465", "status": "completed"} tags=[]
data.head()

# %% papermill={"duration": 0.037088, "end_time": "2020-12-02T17:42:58.308051", "exception": false, "start_time": "2020-12-02T17:42:58.270963", "status": "completed"} tags=[]
assert not data.isna().any().any()

# %% [markdown] papermill={"duration": 0.015778, "end_time": "2020-12-02T17:42:58.342194", "exception": false, "start_time": "2020-12-02T17:42:58.326416", "status": "completed"} tags=[]
# ## Tests different k values (k-NN)

# %% papermill={"duration": 0.030649, "end_time": "2020-12-02T17:42:58.388651", "exception": false, "start_time": "2020-12-02T17:42:58.358002", "status": "completed"} tags=[]
k_values = np.arange(2, 120 + 1, 1)
k_values_to_explore = (2, 5, 10, 15, 20, 30, 40, 50, 75, 100, 125)

# %% papermill={"duration": 92.343628, "end_time": "2020-12-02T17:44:30.749438", "exception": false, "start_time": "2020-12-02T17:42:58.405810", "status": "completed"} tags=[]
results = {}

for k in k_values_to_explore:
    nbrs = NearestNeighbors(n_neighbors=k, n_jobs=N_JOBS).fit(data)
    distances, indices = nbrs.kneighbors(data)
    results[k] = (distances, indices)

# %% papermill={"duration": 0.03185, "end_time": "2020-12-02T17:44:30.799092", "exception": false, "start_time": "2020-12-02T17:44:30.767242", "status": "completed"} tags=[]
min_max_range = (40, 100)

eps_range_per_k = {k: min_max_range for k in k_values}
eps_range_per_k_to_explore = {k: min_max_range for k in k_values_to_explore}

# %% papermill={"duration": 1.014147, "end_time": "2020-12-02T17:44:31.830300", "exception": false, "start_time": "2020-12-02T17:44:30.816153", "status": "completed"} tags=[]
for k, (distances, indices) in results.items():
    d = distances[:, 1:].mean(axis=1)
    d = np.sort(d)

    fig, ax = plt.subplots()
    plt.plot(d)

    r = eps_range_per_k_to_explore[k]
    plt.hlines(r[0], 0, data.shape[0], color="red")
    plt.hlines(r[1], 0, data.shape[0], color="red")

    plt.xlim((3000, data.shape[0]))
    plt.title(f"k={k}")
    display(fig)

    plt.close(fig)

# %% [markdown] papermill={"duration": 0.020037, "end_time": "2020-12-02T17:44:31.871166", "exception": false, "start_time": "2020-12-02T17:44:31.851129", "status": "completed"} tags=[]
# ## Clustering

# %% [markdown] papermill={"duration": 0.020262, "end_time": "2020-12-02T17:44:31.911700", "exception": false, "start_time": "2020-12-02T17:44:31.891438", "status": "completed"} tags=[]
# ### Generate clusterers

# %% papermill={"duration": 0.04042, "end_time": "2020-12-02T17:44:31.972277", "exception": false, "start_time": "2020-12-02T17:44:31.931857", "status": "completed"} tags=[]
CLUSTERING_OPTIONS = {}

# K_RANGE is the min_samples parameter in DBSCAN (sklearn)
CLUSTERING_OPTIONS["K_RANGE"] = k_values
CLUSTERING_OPTIONS["EPS_RANGE_PER_K"] = eps_range_per_k
CLUSTERING_OPTIONS["EPS_STEP"] = 33
CLUSTERING_OPTIONS["METRIC"] = "euclidean"

display(CLUSTERING_OPTIONS)

# %% papermill={"duration": 0.049147, "end_time": "2020-12-02T17:44:32.043506", "exception": false, "start_time": "2020-12-02T17:44:31.994359", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.035696, "end_time": "2020-12-02T17:44:32.100543", "exception": false, "start_time": "2020-12-02T17:44:32.064847", "status": "completed"} tags=[]
display(len(CLUSTERERS))

# %% papermill={"duration": 0.038517, "end_time": "2020-12-02T17:44:32.160987", "exception": false, "start_time": "2020-12-02T17:44:32.122470", "status": "completed"} tags=[]
_iter = iter(CLUSTERERS.items())
display(next(_iter))
display(next(_iter))

# %% papermill={"duration": 0.03753, "end_time": "2020-12-02T17:44:32.220640", "exception": false, "start_time": "2020-12-02T17:44:32.183110", "status": "completed"} tags=[]
clustering_method_name = method_name
display(clustering_method_name)

# %% [markdown] papermill={"duration": 0.022205, "end_time": "2020-12-02T17:44:32.265966", "exception": false, "start_time": "2020-12-02T17:44:32.243761", "status": "completed"} tags=[]
# ### Generate ensemble

# %% papermill={"duration": 0.30126, "end_time": "2020-12-02T17:44:32.591018", "exception": false, "start_time": "2020-12-02T17:44:32.289758", "status": "completed"} tags=[]
data_dist = pairwise_distances(data, metric=CLUSTERING_OPTIONS["METRIC"])

# %% papermill={"duration": 0.037104, "end_time": "2020-12-02T17:44:32.650729", "exception": false, "start_time": "2020-12-02T17:44:32.613625", "status": "completed"} tags=[]
data_dist.shape

# %% papermill={"duration": 0.620062, "end_time": "2020-12-02T17:44:33.294754", "exception": false, "start_time": "2020-12-02T17:44:32.674692", "status": "completed"} tags=[]
pd.Series(data_dist.flatten()).describe().apply(str)

# %% papermill={"duration": 756.176382, "end_time": "2020-12-02T17:57:09.495175", "exception": false, "start_time": "2020-12-02T17:44:33.318793", "status": "completed"} tags=[]
ensemble = generate_ensemble(
    data_dist,
    CLUSTERERS,
    attributes=CLUSTERING_ATTRIBUTES_TO_SAVE,
)

# %% papermill={"duration": 0.658594, "end_time": "2020-12-02T17:57:10.822672", "exception": false, "start_time": "2020-12-02T17:57:10.164078", "status": "completed"} tags=[]
# the number should be close to 295 (the number of partitions generated by k-means/spectral clustering)
ensemble.shape

# %% papermill={"duration": 0.658782, "end_time": "2020-12-02T17:57:12.117682", "exception": false, "start_time": "2020-12-02T17:57:11.458900", "status": "completed"} tags=[]
ensemble.head()

# %% papermill={"duration": 0.656384, "end_time": "2020-12-02T17:57:13.419335", "exception": false, "start_time": "2020-12-02T17:57:12.762951", "status": "completed"} tags=[]
ensemble["n_clusters"].value_counts().head()

# %% papermill={"duration": 0.654754, "end_time": "2020-12-02T17:57:14.738229", "exception": false, "start_time": "2020-12-02T17:57:14.083475", "status": "completed"} tags=[]
ensemble_stats = ensemble["n_clusters"].describe()
display(ensemble_stats)

# %% [markdown] papermill={"duration": 0.637073, "end_time": "2020-12-02T17:57:16.015540", "exception": false, "start_time": "2020-12-02T17:57:15.378467", "status": "completed"} tags=[]
# ### Testing

# %% papermill={"duration": 0.65377, "end_time": "2020-12-02T17:57:17.314117", "exception": false, "start_time": "2020-12-02T17:57:16.660347", "status": "completed"} tags=[]
assert ensemble_stats["min"] > 1

# %% papermill={"duration": 0.650564, "end_time": "2020-12-02T17:57:18.629410", "exception": false, "start_time": "2020-12-02T17:57:17.978846", "status": "completed"} tags=[]
assert not ensemble["n_clusters"].isna().any()

# %% papermill={"duration": 0.654343, "end_time": "2020-12-02T17:57:19.919284", "exception": false, "start_time": "2020-12-02T17:57:19.264941", "status": "completed"} tags=[]
# assert ensemble.shape[0] == len(CLUSTERERS)

# %% papermill={"duration": 0.696475, "end_time": "2020-12-02T17:57:21.258158", "exception": false, "start_time": "2020-12-02T17:57:20.561683", "status": "completed"} tags=[]
# all partitions have the right size
assert np.all(
    [part["partition"].shape[0] == data.shape[0] for idx, part in ensemble.iterrows()]
)

# %% papermill={"duration": 0.671867, "end_time": "2020-12-02T17:57:22.575050", "exception": false, "start_time": "2020-12-02T17:57:21.903183", "status": "completed"} tags=[]
# no partition has negative clusters (noisy points)
assert not np.any([(part["partition"] < 0).any() for idx, part in ensemble.iterrows()])

# %% [markdown] papermill={"duration": 0.637183, "end_time": "2020-12-02T17:57:23.844820", "exception": false, "start_time": "2020-12-02T17:57:23.207637", "status": "completed"} tags=[]
# ### Save

# %% papermill={"duration": 0.677252, "end_time": "2020-12-02T17:57:25.158782", "exception": false, "start_time": "2020-12-02T17:57:24.481530", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.679475, "end_time": "2020-12-02T17:57:26.483185", "exception": false, "start_time": "2020-12-02T17:57:25.803710", "status": "completed"} tags=[]
ensemble.to_pickle(output_filename)

# %% papermill={"duration": 0.643884, "end_time": "2020-12-02T17:57:27.774541", "exception": false, "start_time": "2020-12-02T17:57:27.130657", "status": "completed"} tags=[]
