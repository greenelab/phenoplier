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

# %% [markdown] papermill={"duration": 0.052091, "end_time": "2021-01-05T20:14:40.445508", "exception": false, "start_time": "2021-01-05T20:14:40.393417", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.015963, "end_time": "2021-01-05T20:14:40.479480", "exception": false, "start_time": "2021-01-05T20:14:40.463517", "status": "completed"} tags=[]
# It runs DBSCAN on the pca version of the data.
#
# The notebook explores different values for min_samples and eps (the main parameters of DBSCAN).

# %% [markdown] papermill={"duration": 0.015925, "end_time": "2021-01-05T20:14:40.511484", "exception": false, "start_time": "2021-01-05T20:14:40.495559", "status": "completed"} tags=[]
# # Environment variables

# %% papermill={"duration": 0.029884, "end_time": "2021-01-05T20:14:40.557466", "exception": false, "start_time": "2021-01-05T20:14:40.527582", "status": "completed"} tags=[]
from IPython.display import display

import conf

N_JOBS = conf.GENERAL["N_JOBS"]
display(N_JOBS)

# %% papermill={"duration": 0.02364, "end_time": "2021-01-05T20:14:40.598551", "exception": false, "start_time": "2021-01-05T20:14:40.574911", "status": "completed"} tags=[]
# %env MKL_NUM_THREADS=$N_JOBS
# %env OPEN_BLAS_NUM_THREADS=$N_JOBS
# %env NUMEXPR_NUM_THREADS=$N_JOBS
# %env OMP_NUM_THREADS=$N_JOBS

# %% [markdown] papermill={"duration": 0.016533, "end_time": "2021-01-05T20:14:40.632217", "exception": false, "start_time": "2021-01-05T20:14:40.615684", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.026976, "end_time": "2021-01-05T20:14:40.675550", "exception": false, "start_time": "2021-01-05T20:14:40.648574", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 1.627249, "end_time": "2021-01-05T20:14:42.320225", "exception": false, "start_time": "2021-01-05T20:14:40.692976", "status": "completed"} tags=[]
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

# %% [markdown] papermill={"duration": 0.016276, "end_time": "2021-01-05T20:14:42.354613", "exception": false, "start_time": "2021-01-05T20:14:42.338337", "status": "completed"} tags=[]
# # Global settings

# %% papermill={"duration": 0.031298, "end_time": "2021-01-05T20:14:42.402729", "exception": false, "start_time": "2021-01-05T20:14:42.371431", "status": "completed"} tags=[]
CLUSTERING_ATTRIBUTES_TO_SAVE = ["n_clusters"]

# %% [markdown] papermill={"duration": 0.017121, "end_time": "2021-01-05T20:14:42.437635", "exception": false, "start_time": "2021-01-05T20:14:42.420514", "status": "completed"} tags=[]
# # Data version: pca

# %% [markdown] papermill={"duration": 0.016515, "end_time": "2021-01-05T20:14:42.470698", "exception": false, "start_time": "2021-01-05T20:14:42.454183", "status": "completed"} tags=[]
# ## Settings

# %% papermill={"duration": 0.030832, "end_time": "2021-01-05T20:14:42.518206", "exception": false, "start_time": "2021-01-05T20:14:42.487374", "status": "completed"} tags=[]
INPUT_SUBSET = "pca"

# %% papermill={"duration": 0.030431, "end_time": "2021-01-05T20:14:42.565165", "exception": false, "start_time": "2021-01-05T20:14:42.534734", "status": "completed"} tags=[]
INPUT_STEM = "z_score_std-projection-smultixcan-efo_partial-mashr-zscores"

# %% papermill={"duration": 0.030406, "end_time": "2021-01-05T20:14:42.612410", "exception": false, "start_time": "2021-01-05T20:14:42.582004", "status": "completed"} tags=[]
# parameters of the dimentionality reduction steps
DR_OPTIONS = {
    "n_components": 50,
    "svd_solver": "full",
    "random_state": 0,
}

# %% papermill={"duration": 0.033212, "end_time": "2021-01-05T20:14:42.662085", "exception": false, "start_time": "2021-01-05T20:14:42.628873", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.032301, "end_time": "2021-01-05T20:14:42.711953", "exception": false, "start_time": "2021-01-05T20:14:42.679652", "status": "completed"} tags=[]
# output dir for this notebook
RESULTS_DIR = Path(
    conf.RESULTS["CLUSTERING_RUNS_DIR"],
    f"{INPUT_SUBSET}-{INPUT_STEM}",
).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %% [markdown] papermill={"duration": 0.016807, "end_time": "2021-01-05T20:14:42.746124", "exception": false, "start_time": "2021-01-05T20:14:42.729317", "status": "completed"} tags=[]
# ## Load input file

# %% papermill={"duration": 0.032876, "end_time": "2021-01-05T20:14:42.795990", "exception": false, "start_time": "2021-01-05T20:14:42.763114", "status": "completed"} tags=[]
data = pd.read_pickle(input_filepath)

# %% papermill={"duration": 0.032446, "end_time": "2021-01-05T20:14:42.846563", "exception": false, "start_time": "2021-01-05T20:14:42.814117", "status": "completed"} tags=[]
data.shape

# %% papermill={"duration": 0.045272, "end_time": "2021-01-05T20:14:42.910052", "exception": false, "start_time": "2021-01-05T20:14:42.864780", "status": "completed"} tags=[]
data.head()

# %% papermill={"duration": 0.03326, "end_time": "2021-01-05T20:14:42.962306", "exception": false, "start_time": "2021-01-05T20:14:42.929046", "status": "completed"} tags=[]
assert not data.isna().any().any()

# %% [markdown] papermill={"duration": 0.017518, "end_time": "2021-01-05T20:14:42.998072", "exception": false, "start_time": "2021-01-05T20:14:42.980554", "status": "completed"} tags=[]
# ## Tests different k values (k-NN)

# %% papermill={"duration": 0.031985, "end_time": "2021-01-05T20:14:43.047752", "exception": false, "start_time": "2021-01-05T20:14:43.015767", "status": "completed"} tags=[]
k_values = np.arange(2, 100 + 1, 1)
k_values_to_explore = (2, 5, 10, 15, 20, 30, 40, 50, 75, 100)

# %% papermill={"duration": 5.232883, "end_time": "2021-01-05T20:14:48.299408", "exception": false, "start_time": "2021-01-05T20:14:43.066525", "status": "completed"} tags=[]
results = {}

for k in k_values_to_explore:
    nbrs = NearestNeighbors(n_neighbors=k, n_jobs=N_JOBS).fit(data)
    distances, indices = nbrs.kneighbors(data)
    results[k] = (distances, indices)

# %% papermill={"duration": 0.031657, "end_time": "2021-01-05T20:14:48.349028", "exception": false, "start_time": "2021-01-05T20:14:48.317371", "status": "completed"} tags=[]
min_max_range = (15, 80)

eps_range_per_k = {k: min_max_range for k in k_values}
eps_range_per_k_to_explore = {k: min_max_range for k in k_values_to_explore}

# %% papermill={"duration": 0.923973, "end_time": "2021-01-05T20:14:49.290951", "exception": false, "start_time": "2021-01-05T20:14:48.366978", "status": "completed"} tags=[]
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

# %% [markdown] papermill={"duration": 0.021581, "end_time": "2021-01-05T20:14:49.335097", "exception": false, "start_time": "2021-01-05T20:14:49.313516", "status": "completed"} tags=[]
# ## Clustering

# %% [markdown] papermill={"duration": 0.021152, "end_time": "2021-01-05T20:14:49.377573", "exception": false, "start_time": "2021-01-05T20:14:49.356421", "status": "completed"} tags=[]
# ### Generate clusterers

# %% papermill={"duration": 0.041486, "end_time": "2021-01-05T20:14:49.440653", "exception": false, "start_time": "2021-01-05T20:14:49.399167", "status": "completed"} tags=[]
CLUSTERING_OPTIONS = {}

# K_RANGE is the min_samples parameter in DBSCAN (sklearn)
CLUSTERING_OPTIONS["K_RANGE"] = k_values
CLUSTERING_OPTIONS["EPS_RANGE_PER_K"] = eps_range_per_k
CLUSTERING_OPTIONS["EPS_STEP"] = 33
CLUSTERING_OPTIONS["METRIC"] = "euclidean"

display(CLUSTERING_OPTIONS)

# %% papermill={"duration": 0.048083, "end_time": "2021-01-05T20:14:49.511397", "exception": false, "start_time": "2021-01-05T20:14:49.463314", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.036537, "end_time": "2021-01-05T20:14:49.569731", "exception": false, "start_time": "2021-01-05T20:14:49.533194", "status": "completed"} tags=[]
display(len(CLUSTERERS))

# %% papermill={"duration": 0.038863, "end_time": "2021-01-05T20:14:49.631613", "exception": false, "start_time": "2021-01-05T20:14:49.592750", "status": "completed"} tags=[]
_iter = iter(CLUSTERERS.items())
display(next(_iter))
display(next(_iter))

# %% papermill={"duration": 0.03757, "end_time": "2021-01-05T20:14:49.692419", "exception": false, "start_time": "2021-01-05T20:14:49.654849", "status": "completed"} tags=[]
clustering_method_name = method_name
display(clustering_method_name)

# %% [markdown] papermill={"duration": 0.022589, "end_time": "2021-01-05T20:14:49.739109", "exception": false, "start_time": "2021-01-05T20:14:49.716520", "status": "completed"} tags=[]
# ### Generate ensemble

# %% papermill={"duration": 0.194898, "end_time": "2021-01-05T20:14:49.956791", "exception": false, "start_time": "2021-01-05T20:14:49.761893", "status": "completed"} tags=[]
data_dist = pairwise_distances(data, metric=CLUSTERING_OPTIONS["METRIC"])

# %% papermill={"duration": 0.039013, "end_time": "2021-01-05T20:14:50.018946", "exception": false, "start_time": "2021-01-05T20:14:49.979933", "status": "completed"} tags=[]
data_dist.shape

# %% papermill={"duration": 0.632292, "end_time": "2021-01-05T20:14:50.674460", "exception": false, "start_time": "2021-01-05T20:14:50.042168", "status": "completed"} tags=[]
pd.Series(data_dist.flatten()).describe().apply(str)

# %% papermill={"duration": 656.558869, "end_time": "2021-01-05T20:25:47.257331", "exception": false, "start_time": "2021-01-05T20:14:50.698462", "status": "completed"} tags=[]
ensemble = generate_ensemble(
    data_dist,
    CLUSTERERS,
    attributes=CLUSTERING_ATTRIBUTES_TO_SAVE,
)

# %% papermill={"duration": 0.554745, "end_time": "2021-01-05T20:25:48.345266", "exception": false, "start_time": "2021-01-05T20:25:47.790521", "status": "completed"} tags=[]
# the number should be close to 295 (the number of partitions generated by k-means/spectral clustering)
ensemble.shape

# %% papermill={"duration": 0.586345, "end_time": "2021-01-05T20:25:49.471546", "exception": false, "start_time": "2021-01-05T20:25:48.885201", "status": "completed"} tags=[]
ensemble.head()

# %% papermill={"duration": 0.549154, "end_time": "2021-01-05T20:25:50.561338", "exception": false, "start_time": "2021-01-05T20:25:50.012184", "status": "completed"} tags=[]
ensemble["n_clusters"].value_counts().head()

# %% papermill={"duration": 0.549364, "end_time": "2021-01-05T20:25:51.648869", "exception": false, "start_time": "2021-01-05T20:25:51.099505", "status": "completed"} tags=[]
ensemble_stats = ensemble["n_clusters"].describe()
display(ensemble_stats)

# %% [markdown] papermill={"duration": 0.534144, "end_time": "2021-01-05T20:25:52.725465", "exception": false, "start_time": "2021-01-05T20:25:52.191321", "status": "completed"} tags=[]
# ### Testing

# %% papermill={"duration": 0.54649, "end_time": "2021-01-05T20:25:53.841880", "exception": false, "start_time": "2021-01-05T20:25:53.295390", "status": "completed"} tags=[]
assert ensemble_stats["min"] > 1

# %% papermill={"duration": 0.548432, "end_time": "2021-01-05T20:25:54.926906", "exception": false, "start_time": "2021-01-05T20:25:54.378474", "status": "completed"} tags=[]
assert not ensemble["n_clusters"].isna().any()

# %% papermill={"duration": 0.547409, "end_time": "2021-01-05T20:25:56.004296", "exception": false, "start_time": "2021-01-05T20:25:55.456887", "status": "completed"} tags=[]
# assert ensemble.shape[0] == len(CLUSTERERS)

# %% papermill={"duration": 0.597269, "end_time": "2021-01-05T20:25:57.129558", "exception": false, "start_time": "2021-01-05T20:25:56.532289", "status": "completed"} tags=[]
# all partitions have the right size
assert np.all(
    [part["partition"].shape[0] == data.shape[0] for idx, part in ensemble.iterrows()]
)

# %% papermill={"duration": 0.568428, "end_time": "2021-01-05T20:25:58.230053", "exception": false, "start_time": "2021-01-05T20:25:57.661625", "status": "completed"} tags=[]
# no partition has negative clusters (noisy points)
assert not np.any([(part["partition"] < 0).any() for idx, part in ensemble.iterrows()])

# %% [markdown] papermill={"duration": 0.540061, "end_time": "2021-01-05T20:25:59.309882", "exception": false, "start_time": "2021-01-05T20:25:58.769821", "status": "completed"} tags=[]
# ### Save

# %% papermill={"duration": 0.555437, "end_time": "2021-01-05T20:26:00.407584", "exception": false, "start_time": "2021-01-05T20:25:59.852147", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.547358, "end_time": "2021-01-05T20:26:01.520518", "exception": false, "start_time": "2021-01-05T20:26:00.973160", "status": "completed"} tags=[]
ensemble.to_pickle(output_filename)

# %% papermill={"duration": 0.535901, "end_time": "2021-01-05T20:26:02.593315", "exception": false, "start_time": "2021-01-05T20:26:02.057414", "status": "completed"} tags=[]
