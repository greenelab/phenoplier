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

# %% [markdown] papermill={"duration": 0.019278, "end_time": "2021-01-05T19:59:42.266103", "exception": false, "start_time": "2021-01-05T19:59:42.246825", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.015358, "end_time": "2021-01-05T19:59:42.297067", "exception": false, "start_time": "2021-01-05T19:59:42.281709", "status": "completed"} tags=[]
# It runs DBSCAN on the z_score_std version of the data.
#
# The notebook explores different values for min_samples and eps (the main parameters of DBSCAN).

# %% [markdown] papermill={"duration": 0.015469, "end_time": "2021-01-05T19:59:42.328271", "exception": false, "start_time": "2021-01-05T19:59:42.312802", "status": "completed"} tags=[]
# # Environment variables

# %% papermill={"duration": 0.02933, "end_time": "2021-01-05T19:59:42.373118", "exception": false, "start_time": "2021-01-05T19:59:42.343788", "status": "completed"} tags=[]
from IPython.display import display

import conf

N_JOBS = conf.GENERAL["N_JOBS"]
display(N_JOBS)

# %% papermill={"duration": 0.022518, "end_time": "2021-01-05T19:59:42.412125", "exception": false, "start_time": "2021-01-05T19:59:42.389607", "status": "completed"} tags=[]
# %env MKL_NUM_THREADS=$N_JOBS
# %env OPEN_BLAS_NUM_THREADS=$N_JOBS
# %env NUMEXPR_NUM_THREADS=$N_JOBS
# %env OMP_NUM_THREADS=$N_JOBS

# %% [markdown] papermill={"duration": 0.016242, "end_time": "2021-01-05T19:59:42.445338", "exception": false, "start_time": "2021-01-05T19:59:42.429096", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.026775, "end_time": "2021-01-05T19:59:42.488725", "exception": false, "start_time": "2021-01-05T19:59:42.461950", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 1.614543, "end_time": "2021-01-05T19:59:44.120402", "exception": false, "start_time": "2021-01-05T19:59:42.505859", "status": "completed"} tags=[]
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

# %% [markdown] papermill={"duration": 0.016012, "end_time": "2021-01-05T19:59:44.154073", "exception": false, "start_time": "2021-01-05T19:59:44.138061", "status": "completed"} tags=[]
# # Global settings

# %% papermill={"duration": 0.030462, "end_time": "2021-01-05T19:59:44.200539", "exception": false, "start_time": "2021-01-05T19:59:44.170077", "status": "completed"} tags=[]
CLUSTERING_ATTRIBUTES_TO_SAVE = ["n_clusters"]

# %% [markdown] papermill={"duration": 0.015825, "end_time": "2021-01-05T19:59:44.233352", "exception": false, "start_time": "2021-01-05T19:59:44.217527", "status": "completed"} tags=[]
# # Data version: z_score_std

# %% [markdown] papermill={"duration": 0.015731, "end_time": "2021-01-05T19:59:44.265291", "exception": false, "start_time": "2021-01-05T19:59:44.249560", "status": "completed"} tags=[]
# ## Settings

# %% papermill={"duration": 0.030104, "end_time": "2021-01-05T19:59:44.311054", "exception": false, "start_time": "2021-01-05T19:59:44.280950", "status": "completed"} tags=[]
INPUT_SUBSET = "z_score_std"

# %% papermill={"duration": 0.030491, "end_time": "2021-01-05T19:59:44.357609", "exception": false, "start_time": "2021-01-05T19:59:44.327118", "status": "completed"} tags=[]
INPUT_STEM = "projection-smultixcan-efo_partial-mashr-zscores"

# %% papermill={"duration": 0.032222, "end_time": "2021-01-05T19:59:44.406217", "exception": false, "start_time": "2021-01-05T19:59:44.373995", "status": "completed"} tags=[]
input_filepath = Path(
    conf.RESULTS["DATA_TRANSFORMATIONS_DIR"],
    INPUT_SUBSET,
    f"{INPUT_SUBSET}-{INPUT_STEM}.pkl",
).resolve()
display(input_filepath)

assert input_filepath.exists(), "Input file does not exist"

input_filepath_stem = input_filepath.stem
display(input_filepath_stem)

# %% papermill={"duration": 0.032435, "end_time": "2021-01-05T19:59:44.455880", "exception": false, "start_time": "2021-01-05T19:59:44.423445", "status": "completed"} tags=[]
# output dir for this notebook
RESULTS_DIR = Path(
    conf.RESULTS["CLUSTERING_RUNS_DIR"],
    f"{INPUT_SUBSET}-{INPUT_STEM}",
).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %% [markdown] papermill={"duration": 0.016841, "end_time": "2021-01-05T19:59:44.490150", "exception": false, "start_time": "2021-01-05T19:59:44.473309", "status": "completed"} tags=[]
# ## Load input file

# %% papermill={"duration": 0.044289, "end_time": "2021-01-05T19:59:44.550867", "exception": false, "start_time": "2021-01-05T19:59:44.506578", "status": "completed"} tags=[]
data = pd.read_pickle(input_filepath)

# %% papermill={"duration": 0.033878, "end_time": "2021-01-05T19:59:44.604063", "exception": false, "start_time": "2021-01-05T19:59:44.570185", "status": "completed"} tags=[]
data.shape

# %% papermill={"duration": 0.0456, "end_time": "2021-01-05T19:59:44.668449", "exception": false, "start_time": "2021-01-05T19:59:44.622849", "status": "completed"} tags=[]
data.head()

# %% papermill={"duration": 0.037571, "end_time": "2021-01-05T19:59:44.723858", "exception": false, "start_time": "2021-01-05T19:59:44.686287", "status": "completed"} tags=[]
assert not data.isna().any().any()

# %% [markdown] papermill={"duration": 0.017824, "end_time": "2021-01-05T19:59:44.760140", "exception": false, "start_time": "2021-01-05T19:59:44.742316", "status": "completed"} tags=[]
# ## Tests different k values (k-NN)

# %% papermill={"duration": 0.031834, "end_time": "2021-01-05T19:59:44.809030", "exception": false, "start_time": "2021-01-05T19:59:44.777196", "status": "completed"} tags=[]
k_values = np.arange(2, 120 + 1, 1)
k_values_to_explore = (2, 5, 10, 15, 20, 30, 40, 50, 75, 100, 125)

# %% papermill={"duration": 92.099791, "end_time": "2021-01-05T20:01:16.926026", "exception": false, "start_time": "2021-01-05T19:59:44.826235", "status": "completed"} tags=[]
results = {}

for k in k_values_to_explore:
    nbrs = NearestNeighbors(n_neighbors=k, n_jobs=N_JOBS).fit(data)
    distances, indices = nbrs.kneighbors(data)
    results[k] = (distances, indices)

# %% papermill={"duration": 0.033321, "end_time": "2021-01-05T20:01:16.978881", "exception": false, "start_time": "2021-01-05T20:01:16.945560", "status": "completed"} tags=[]
min_max_range = (40, 100)

eps_range_per_k = {k: min_max_range for k in k_values}
eps_range_per_k_to_explore = {k: min_max_range for k in k_values_to_explore}

# %% papermill={"duration": 1.019867, "end_time": "2021-01-05T20:01:18.017366", "exception": false, "start_time": "2021-01-05T20:01:16.997499", "status": "completed"} tags=[]
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

# %% [markdown] papermill={"duration": 0.021689, "end_time": "2021-01-05T20:01:18.061628", "exception": false, "start_time": "2021-01-05T20:01:18.039939", "status": "completed"} tags=[]
# ## Clustering

# %% [markdown] papermill={"duration": 0.021495, "end_time": "2021-01-05T20:01:18.104586", "exception": false, "start_time": "2021-01-05T20:01:18.083091", "status": "completed"} tags=[]
# ### Generate clusterers

# %% papermill={"duration": 0.042657, "end_time": "2021-01-05T20:01:18.168795", "exception": false, "start_time": "2021-01-05T20:01:18.126138", "status": "completed"} tags=[]
CLUSTERING_OPTIONS = {}

# K_RANGE is the min_samples parameter in DBSCAN (sklearn)
CLUSTERING_OPTIONS["K_RANGE"] = k_values
CLUSTERING_OPTIONS["EPS_RANGE_PER_K"] = eps_range_per_k
CLUSTERING_OPTIONS["EPS_STEP"] = 33
CLUSTERING_OPTIONS["METRIC"] = "euclidean"

display(CLUSTERING_OPTIONS)

# %% papermill={"duration": 0.051179, "end_time": "2021-01-05T20:01:18.242366", "exception": false, "start_time": "2021-01-05T20:01:18.191187", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.037431, "end_time": "2021-01-05T20:01:18.302287", "exception": false, "start_time": "2021-01-05T20:01:18.264856", "status": "completed"} tags=[]
display(len(CLUSTERERS))

# %% papermill={"duration": 0.03996, "end_time": "2021-01-05T20:01:18.365415", "exception": false, "start_time": "2021-01-05T20:01:18.325455", "status": "completed"} tags=[]
_iter = iter(CLUSTERERS.items())
display(next(_iter))
display(next(_iter))

# %% papermill={"duration": 0.037665, "end_time": "2021-01-05T20:01:18.427269", "exception": false, "start_time": "2021-01-05T20:01:18.389604", "status": "completed"} tags=[]
clustering_method_name = method_name
display(clustering_method_name)

# %% [markdown] papermill={"duration": 0.022669, "end_time": "2021-01-05T20:01:18.473711", "exception": false, "start_time": "2021-01-05T20:01:18.451042", "status": "completed"} tags=[]
# ### Generate ensemble

# %% papermill={"duration": 0.297542, "end_time": "2021-01-05T20:01:18.793769", "exception": false, "start_time": "2021-01-05T20:01:18.496227", "status": "completed"} tags=[]
data_dist = pairwise_distances(data, metric=CLUSTERING_OPTIONS["METRIC"])

# %% papermill={"duration": 0.03833, "end_time": "2021-01-05T20:01:18.855550", "exception": false, "start_time": "2021-01-05T20:01:18.817220", "status": "completed"} tags=[]
data_dist.shape

# %% papermill={"duration": 0.624451, "end_time": "2021-01-05T20:01:19.504025", "exception": false, "start_time": "2021-01-05T20:01:18.879574", "status": "completed"} tags=[]
pd.Series(data_dist.flatten()).describe().apply(str)

# %% papermill={"duration": 778.089035, "end_time": "2021-01-05T20:14:17.616814", "exception": false, "start_time": "2021-01-05T20:01:19.527779", "status": "completed"} tags=[]
ensemble = generate_ensemble(
    data_dist,
    CLUSTERERS,
    attributes=CLUSTERING_ATTRIBUTES_TO_SAVE,
)

# %% papermill={"duration": 0.649517, "end_time": "2021-01-05T20:14:18.900786", "exception": false, "start_time": "2021-01-05T20:14:18.251269", "status": "completed"} tags=[]
# the number should be close to 295 (the number of partitions generated by k-means/spectral clustering)
ensemble.shape

# %% papermill={"duration": 0.663909, "end_time": "2021-01-05T20:14:20.197354", "exception": false, "start_time": "2021-01-05T20:14:19.533445", "status": "completed"} tags=[]
ensemble.head()

# %% papermill={"duration": 0.681072, "end_time": "2021-01-05T20:14:21.510803", "exception": false, "start_time": "2021-01-05T20:14:20.829731", "status": "completed"} tags=[]
ensemble["n_clusters"].value_counts().head()

# %% papermill={"duration": 0.65238, "end_time": "2021-01-05T20:14:22.803477", "exception": false, "start_time": "2021-01-05T20:14:22.151097", "status": "completed"} tags=[]
ensemble_stats = ensemble["n_clusters"].describe()
display(ensemble_stats)

# %% [markdown] papermill={"duration": 0.650641, "end_time": "2021-01-05T20:14:24.099036", "exception": false, "start_time": "2021-01-05T20:14:23.448395", "status": "completed"} tags=[]
# ### Testing

# %% papermill={"duration": 0.651666, "end_time": "2021-01-05T20:14:25.417932", "exception": false, "start_time": "2021-01-05T20:14:24.766266", "status": "completed"} tags=[]
assert ensemble_stats["min"] > 1

# %% papermill={"duration": 0.653131, "end_time": "2021-01-05T20:14:26.706141", "exception": false, "start_time": "2021-01-05T20:14:26.053010", "status": "completed"} tags=[]
assert not ensemble["n_clusters"].isna().any()

# %% papermill={"duration": 0.648296, "end_time": "2021-01-05T20:14:27.986743", "exception": false, "start_time": "2021-01-05T20:14:27.338447", "status": "completed"} tags=[]
# assert ensemble.shape[0] == len(CLUSTERERS)

# %% papermill={"duration": 0.668086, "end_time": "2021-01-05T20:14:29.319404", "exception": false, "start_time": "2021-01-05T20:14:28.651318", "status": "completed"} tags=[]
# all partitions have the right size
assert np.all(
    [part["partition"].shape[0] == data.shape[0] for idx, part in ensemble.iterrows()]
)

# %% papermill={"duration": 0.66796, "end_time": "2021-01-05T20:14:30.620306", "exception": false, "start_time": "2021-01-05T20:14:29.952346", "status": "completed"} tags=[]
# no partition has negative clusters (noisy points)
assert not np.any([(part["partition"] < 0).any() for idx, part in ensemble.iterrows()])

# %% [markdown] papermill={"duration": 0.633093, "end_time": "2021-01-05T20:14:31.889498", "exception": false, "start_time": "2021-01-05T20:14:31.256405", "status": "completed"} tags=[]
# ### Save

# %% papermill={"duration": 0.64826, "end_time": "2021-01-05T20:14:33.192396", "exception": false, "start_time": "2021-01-05T20:14:32.544136", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.657614, "end_time": "2021-01-05T20:14:34.481078", "exception": false, "start_time": "2021-01-05T20:14:33.823464", "status": "completed"} tags=[]
ensemble.to_pickle(output_filename)

# %% papermill={"duration": 0.654429, "end_time": "2021-01-05T20:14:35.767089", "exception": false, "start_time": "2021-01-05T20:14:35.112660", "status": "completed"} tags=[]
