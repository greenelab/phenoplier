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

# %% [markdown] papermill={"duration": 0.0179, "end_time": "2020-12-02T18:08:42.373130", "exception": false, "start_time": "2020-12-02T18:08:42.355230", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.014662, "end_time": "2020-12-02T18:08:42.401688", "exception": false, "start_time": "2020-12-02T18:08:42.387026", "status": "completed"} tags=[]
# It runs DBSCAN on the umap version of the data.
#
# The notebook explores different values for min_samples and eps (the main parameters of DBSCAN).

# %% [markdown] papermill={"duration": 0.013792, "end_time": "2020-12-02T18:08:42.429275", "exception": false, "start_time": "2020-12-02T18:08:42.415483", "status": "completed"} tags=[]
# # Environment variables

# %% papermill={"duration": 0.027916, "end_time": "2020-12-02T18:08:42.471012", "exception": false, "start_time": "2020-12-02T18:08:42.443096", "status": "completed"} tags=[]
from IPython.display import display

import conf

N_JOBS = conf.GENERAL["N_JOBS"]
display(N_JOBS)

# %% papermill={"duration": 0.020929, "end_time": "2020-12-02T18:08:42.506818", "exception": false, "start_time": "2020-12-02T18:08:42.485889", "status": "completed"} tags=[]
# %env MKL_NUM_THREADS=$N_JOBS
# %env OPEN_BLAS_NUM_THREADS=$N_JOBS
# %env NUMEXPR_NUM_THREADS=$N_JOBS
# %env OMP_NUM_THREADS=$N_JOBS

# %% [markdown] papermill={"duration": 0.014362, "end_time": "2020-12-02T18:08:42.536088", "exception": false, "start_time": "2020-12-02T18:08:42.521726", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.024706, "end_time": "2020-12-02T18:08:42.575480", "exception": false, "start_time": "2020-12-02T18:08:42.550774", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 1.629535, "end_time": "2020-12-02T18:08:44.220516", "exception": false, "start_time": "2020-12-02T18:08:42.590981", "status": "completed"} tags=[]
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

# %% [markdown] papermill={"duration": 0.014601, "end_time": "2020-12-02T18:08:44.251342", "exception": false, "start_time": "2020-12-02T18:08:44.236741", "status": "completed"} tags=[]
# # Global settings

# %% papermill={"duration": 0.029279, "end_time": "2020-12-02T18:08:44.295492", "exception": false, "start_time": "2020-12-02T18:08:44.266213", "status": "completed"} tags=[]
CLUSTERING_ATTRIBUTES_TO_SAVE = ["n_clusters"]

# %% [markdown] papermill={"duration": 0.014952, "end_time": "2020-12-02T18:08:44.325857", "exception": false, "start_time": "2020-12-02T18:08:44.310905", "status": "completed"} tags=[]
# # Data version: umap

# %% [markdown] papermill={"duration": 0.01458, "end_time": "2020-12-02T18:08:44.355025", "exception": false, "start_time": "2020-12-02T18:08:44.340445", "status": "completed"} tags=[]
# ## Settings

# %% papermill={"duration": 0.029198, "end_time": "2020-12-02T18:08:44.399011", "exception": false, "start_time": "2020-12-02T18:08:44.369813", "status": "completed"} tags=[]
INPUT_SUBSET = "umap"

# %% papermill={"duration": 0.028966, "end_time": "2020-12-02T18:08:44.443120", "exception": false, "start_time": "2020-12-02T18:08:44.414154", "status": "completed"} tags=[]
INPUT_STEM = "z_score_std-projection-smultixcan-efo_partial-mashr-zscores"

# %% papermill={"duration": 0.029131, "end_time": "2020-12-02T18:08:44.487549", "exception": false, "start_time": "2020-12-02T18:08:44.458418", "status": "completed"} tags=[]
# parameters of the dimentionality reduction steps
DR_OPTIONS = {
    "n_components": 50,
    "metric": "euclidean",
    "n_neighbors": 15,
    "random_state": 0,
}

# %% papermill={"duration": 0.031514, "end_time": "2020-12-02T18:08:44.534488", "exception": false, "start_time": "2020-12-02T18:08:44.502974", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.03064, "end_time": "2020-12-02T18:08:44.581468", "exception": false, "start_time": "2020-12-02T18:08:44.550828", "status": "completed"} tags=[]
# output dir for this notebook
RESULTS_DIR = Path(
    conf.RESULTS["CLUSTERING_RUNS_DIR"],
    f"{INPUT_SUBSET}-{INPUT_STEM}",
).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %% [markdown] papermill={"duration": 0.015867, "end_time": "2020-12-02T18:08:44.613952", "exception": false, "start_time": "2020-12-02T18:08:44.598085", "status": "completed"} tags=[]
# ## Load input file

# %% papermill={"duration": 0.031619, "end_time": "2020-12-02T18:08:44.661376", "exception": false, "start_time": "2020-12-02T18:08:44.629757", "status": "completed"} tags=[]
data = pd.read_pickle(input_filepath)

# %% papermill={"duration": 0.030537, "end_time": "2020-12-02T18:08:44.708195", "exception": false, "start_time": "2020-12-02T18:08:44.677658", "status": "completed"} tags=[]
data.shape

# %% papermill={"duration": 0.042883, "end_time": "2020-12-02T18:08:44.767580", "exception": false, "start_time": "2020-12-02T18:08:44.724697", "status": "completed"} tags=[]
data.head()

# %% papermill={"duration": 0.031888, "end_time": "2020-12-02T18:08:44.816863", "exception": false, "start_time": "2020-12-02T18:08:44.784975", "status": "completed"} tags=[]
assert not data.isna().any().any()

# %% [markdown] papermill={"duration": 0.016281, "end_time": "2020-12-02T18:08:44.850168", "exception": false, "start_time": "2020-12-02T18:08:44.833887", "status": "completed"} tags=[]
# ## Tests different k values (k-NN)

# %% papermill={"duration": 0.03078, "end_time": "2020-12-02T18:08:44.897280", "exception": false, "start_time": "2020-12-02T18:08:44.866500", "status": "completed"} tags=[]
k_values = np.arange(10, 150 + 1, 1)
k_values_to_explore = (10, 15, 20, 30, 40, 50, 75, 100, 125, 150)  # , 175, 200)

# %% papermill={"duration": 5.028814, "end_time": "2020-12-02T18:08:49.943427", "exception": false, "start_time": "2020-12-02T18:08:44.914613", "status": "completed"} tags=[]
results = {}

for k in k_values_to_explore:
    nbrs = NearestNeighbors(n_neighbors=k, n_jobs=N_JOBS).fit(data)
    distances, indices = nbrs.kneighbors(data)
    results[k] = (distances, indices)

# %% papermill={"duration": 0.032237, "end_time": "2020-12-02T18:08:49.992684", "exception": false, "start_time": "2020-12-02T18:08:49.960447", "status": "completed"} tags=[]
# min_max_range = (1.0, 3)

eps_range_per_k = {
    k: (0.75, 1.25)
    if k == 10
    else (0.85, 1.60)
    if k == 15
    else (1.0, 2.50)
    if k < 40
    else (1.25, 3.0)
    if k < 75
    else (1.25, 3.0)
    if k < 100
    else (1.50, 3.0)
    if k < 175
    else (1.75, 3.0)
    for k in k_values
}

eps_range_per_k_to_explore = {k: eps_range_per_k[k] for k in k_values_to_explore}

# %% papermill={"duration": 0.909212, "end_time": "2020-12-02T18:08:50.918907", "exception": false, "start_time": "2020-12-02T18:08:50.009695", "status": "completed"} tags=[]
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

# %% [markdown] papermill={"duration": 0.020322, "end_time": "2020-12-02T18:08:50.960782", "exception": false, "start_time": "2020-12-02T18:08:50.940460", "status": "completed"} tags=[]
# ## Clustering

# %% [markdown] papermill={"duration": 0.020376, "end_time": "2020-12-02T18:08:51.001454", "exception": false, "start_time": "2020-12-02T18:08:50.981078", "status": "completed"} tags=[]
# ### Generate clusterers

# %% papermill={"duration": 0.040778, "end_time": "2020-12-02T18:08:51.062496", "exception": false, "start_time": "2020-12-02T18:08:51.021718", "status": "completed"} tags=[]
CLUSTERING_OPTIONS = {}

# K_RANGE is the min_samples parameter in DBSCAN (sklearn)
CLUSTERING_OPTIONS["K_RANGE"] = k_values
CLUSTERING_OPTIONS["EPS_RANGE_PER_K"] = eps_range_per_k
CLUSTERING_OPTIONS["EPS_STEP"] = 33
CLUSTERING_OPTIONS["METRIC"] = "euclidean"

display(CLUSTERING_OPTIONS)

# %% papermill={"duration": 0.053338, "end_time": "2020-12-02T18:08:51.138032", "exception": false, "start_time": "2020-12-02T18:08:51.084694", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.036518, "end_time": "2020-12-02T18:08:51.195802", "exception": false, "start_time": "2020-12-02T18:08:51.159284", "status": "completed"} tags=[]
display(len(CLUSTERERS))

# %% papermill={"duration": 0.037967, "end_time": "2020-12-02T18:08:51.255908", "exception": false, "start_time": "2020-12-02T18:08:51.217941", "status": "completed"} tags=[]
_iter = iter(CLUSTERERS.items())
display(next(_iter))
display(next(_iter))

# %% papermill={"duration": 0.036808, "end_time": "2020-12-02T18:08:51.316053", "exception": false, "start_time": "2020-12-02T18:08:51.279245", "status": "completed"} tags=[]
clustering_method_name = method_name
display(clustering_method_name)

# %% [markdown] papermill={"duration": 0.021735, "end_time": "2020-12-02T18:08:51.360634", "exception": false, "start_time": "2020-12-02T18:08:51.338899", "status": "completed"} tags=[]
# ### Generate ensemble

# %% papermill={"duration": 0.179107, "end_time": "2020-12-02T18:08:51.561883", "exception": false, "start_time": "2020-12-02T18:08:51.382776", "status": "completed"} tags=[]
data_dist = pairwise_distances(data, metric=CLUSTERING_OPTIONS["METRIC"])

# %% papermill={"duration": 0.037507, "end_time": "2020-12-02T18:08:51.622661", "exception": false, "start_time": "2020-12-02T18:08:51.585154", "status": "completed"} tags=[]
data_dist.shape

# %% papermill={"duration": 0.470303, "end_time": "2020-12-02T18:08:52.115682", "exception": false, "start_time": "2020-12-02T18:08:51.645379", "status": "completed"} tags=[]
pd.Series(data_dist.flatten()).describe().apply(str)

# %% papermill={"duration": 763.748112, "end_time": "2020-12-02T18:21:35.887000", "exception": false, "start_time": "2020-12-02T18:08:52.138888", "status": "completed"} tags=[]
ensemble = generate_ensemble(
    data_dist,
    CLUSTERERS,
    attributes=CLUSTERING_ATTRIBUTES_TO_SAVE,
)

# %% papermill={"duration": 0.766413, "end_time": "2020-12-02T18:21:37.414888", "exception": false, "start_time": "2020-12-02T18:21:36.648475", "status": "completed"} tags=[]
# the number should be close to 295 (the number of partitions generated by k-means/spectral clustering)
ensemble.shape

# %% papermill={"duration": 0.77212, "end_time": "2020-12-02T18:21:38.970926", "exception": false, "start_time": "2020-12-02T18:21:38.198806", "status": "completed"} tags=[]
ensemble.head()

# %% papermill={"duration": 0.770983, "end_time": "2020-12-02T18:21:40.487006", "exception": false, "start_time": "2020-12-02T18:21:39.716023", "status": "completed"} tags=[]
ensemble["n_clusters"].value_counts().head()

# %% papermill={"duration": 0.795647, "end_time": "2020-12-02T18:21:42.034049", "exception": false, "start_time": "2020-12-02T18:21:41.238402", "status": "completed"} tags=[]
ensemble_stats = ensemble["n_clusters"].describe()
display(ensemble_stats)

# %% [markdown] papermill={"duration": 0.759088, "end_time": "2020-12-02T18:21:43.559983", "exception": false, "start_time": "2020-12-02T18:21:42.800895", "status": "completed"} tags=[]
# ### Testing

# %% papermill={"duration": 0.767151, "end_time": "2020-12-02T18:21:45.083803", "exception": false, "start_time": "2020-12-02T18:21:44.316652", "status": "completed"} tags=[]
assert ensemble_stats["min"] > 1

# %% papermill={"duration": 0.776296, "end_time": "2020-12-02T18:21:46.645420", "exception": false, "start_time": "2020-12-02T18:21:45.869124", "status": "completed"} tags=[]
assert not ensemble["n_clusters"].isna().any()

# %% papermill={"duration": 0.767803, "end_time": "2020-12-02T18:21:48.170280", "exception": false, "start_time": "2020-12-02T18:21:47.402477", "status": "completed"} tags=[]
# assert ensemble.shape[0] == len(CLUSTERERS)

# %% papermill={"duration": 0.865819, "end_time": "2020-12-02T18:21:49.794741", "exception": false, "start_time": "2020-12-02T18:21:48.928922", "status": "completed"} tags=[]
# all partitions have the right size
assert np.all(
    [part["partition"].shape[0] == data.shape[0] for idx, part in ensemble.iterrows()]
)

# %% papermill={"duration": 0.846599, "end_time": "2020-12-02T18:21:51.403023", "exception": false, "start_time": "2020-12-02T18:21:50.556424", "status": "completed"} tags=[]
# no partition has negative clusters (noisy points)
assert not np.any([(part["partition"] < 0).any() for idx, part in ensemble.iterrows()])

# %% [markdown] papermill={"duration": 0.785646, "end_time": "2020-12-02T18:21:52.944530", "exception": false, "start_time": "2020-12-02T18:21:52.158884", "status": "completed"} tags=[]
# ### Save

# %% papermill={"duration": 0.770943, "end_time": "2020-12-02T18:21:54.475557", "exception": false, "start_time": "2020-12-02T18:21:53.704614", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.919359, "end_time": "2020-12-02T18:21:56.152091", "exception": false, "start_time": "2020-12-02T18:21:55.232732", "status": "completed"} tags=[]
ensemble.to_pickle(output_filename)

# %% papermill={"duration": 0.755391, "end_time": "2020-12-02T18:21:57.725700", "exception": false, "start_time": "2020-12-02T18:21:56.970309", "status": "completed"} tags=[]
