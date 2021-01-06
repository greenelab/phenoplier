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

# %% [markdown] papermill={"duration": 0.017976, "end_time": "2021-01-05T20:26:06.913891", "exception": false, "start_time": "2021-01-05T20:26:06.895915", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.017233, "end_time": "2021-01-05T20:26:06.948040", "exception": false, "start_time": "2021-01-05T20:26:06.930807", "status": "completed"} tags=[]
# It runs DBSCAN on the umap version of the data.
#
# The notebook explores different values for min_samples and eps (the main parameters of DBSCAN).

# %% [markdown] papermill={"duration": 0.017266, "end_time": "2021-01-05T20:26:06.982329", "exception": false, "start_time": "2021-01-05T20:26:06.965063", "status": "completed"} tags=[]
# # Environment variables

# %% papermill={"duration": 0.031492, "end_time": "2021-01-05T20:26:07.030694", "exception": false, "start_time": "2021-01-05T20:26:06.999202", "status": "completed"} tags=[]
from IPython.display import display

import conf

N_JOBS = conf.GENERAL["N_JOBS"]
display(N_JOBS)

# %% papermill={"duration": 0.023439, "end_time": "2021-01-05T20:26:07.072327", "exception": false, "start_time": "2021-01-05T20:26:07.048888", "status": "completed"} tags=[]
# %env MKL_NUM_THREADS=$N_JOBS
# %env OPEN_BLAS_NUM_THREADS=$N_JOBS
# %env NUMEXPR_NUM_THREADS=$N_JOBS
# %env OMP_NUM_THREADS=$N_JOBS

# %% [markdown] papermill={"duration": 0.01695, "end_time": "2021-01-05T20:26:07.106607", "exception": false, "start_time": "2021-01-05T20:26:07.089657", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.028381, "end_time": "2021-01-05T20:26:07.151675", "exception": false, "start_time": "2021-01-05T20:26:07.123294", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 1.626581, "end_time": "2021-01-05T20:26:08.795899", "exception": false, "start_time": "2021-01-05T20:26:07.169318", "status": "completed"} tags=[]
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

# %% [markdown] papermill={"duration": 0.016475, "end_time": "2021-01-05T20:26:08.830258", "exception": false, "start_time": "2021-01-05T20:26:08.813783", "status": "completed"} tags=[]
# # Global settings

# %% papermill={"duration": 0.035009, "end_time": "2021-01-05T20:26:08.881878", "exception": false, "start_time": "2021-01-05T20:26:08.846869", "status": "completed"} tags=[]
CLUSTERING_ATTRIBUTES_TO_SAVE = ["n_clusters"]

# %% [markdown] papermill={"duration": 0.01688, "end_time": "2021-01-05T20:26:08.916486", "exception": false, "start_time": "2021-01-05T20:26:08.899606", "status": "completed"} tags=[]
# # Data version: umap

# %% [markdown] papermill={"duration": 0.016628, "end_time": "2021-01-05T20:26:08.949710", "exception": false, "start_time": "2021-01-05T20:26:08.933082", "status": "completed"} tags=[]
# ## Settings

# %% papermill={"duration": 0.03136, "end_time": "2021-01-05T20:26:08.997937", "exception": false, "start_time": "2021-01-05T20:26:08.966577", "status": "completed"} tags=[]
INPUT_SUBSET = "umap"

# %% papermill={"duration": 0.030819, "end_time": "2021-01-05T20:26:09.045567", "exception": false, "start_time": "2021-01-05T20:26:09.014748", "status": "completed"} tags=[]
INPUT_STEM = "z_score_std-projection-smultixcan-efo_partial-mashr-zscores"

# %% papermill={"duration": 0.030918, "end_time": "2021-01-05T20:26:09.093466", "exception": false, "start_time": "2021-01-05T20:26:09.062548", "status": "completed"} tags=[]
# parameters of the dimentionality reduction steps
DR_OPTIONS = {
    "n_components": 50,
    "metric": "euclidean",
    "n_neighbors": 15,
    "random_state": 0,
}

# %% papermill={"duration": 0.032474, "end_time": "2021-01-05T20:26:09.142822", "exception": false, "start_time": "2021-01-05T20:26:09.110348", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.032828, "end_time": "2021-01-05T20:26:09.193478", "exception": false, "start_time": "2021-01-05T20:26:09.160650", "status": "completed"} tags=[]
# output dir for this notebook
RESULTS_DIR = Path(
    conf.RESULTS["CLUSTERING_RUNS_DIR"],
    f"{INPUT_SUBSET}-{INPUT_STEM}",
).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %% [markdown] papermill={"duration": 0.017634, "end_time": "2021-01-05T20:26:09.229099", "exception": false, "start_time": "2021-01-05T20:26:09.211465", "status": "completed"} tags=[]
# ## Load input file

# %% papermill={"duration": 0.033067, "end_time": "2021-01-05T20:26:09.280100", "exception": false, "start_time": "2021-01-05T20:26:09.247033", "status": "completed"} tags=[]
data = pd.read_pickle(input_filepath)

# %% papermill={"duration": 0.033057, "end_time": "2021-01-05T20:26:09.330927", "exception": false, "start_time": "2021-01-05T20:26:09.297870", "status": "completed"} tags=[]
data.shape

# %% papermill={"duration": 0.044747, "end_time": "2021-01-05T20:26:09.393675", "exception": false, "start_time": "2021-01-05T20:26:09.348928", "status": "completed"} tags=[]
data.head()

# %% papermill={"duration": 0.032424, "end_time": "2021-01-05T20:26:09.444143", "exception": false, "start_time": "2021-01-05T20:26:09.411719", "status": "completed"} tags=[]
assert not data.isna().any().any()

# %% [markdown] papermill={"duration": 0.01817, "end_time": "2021-01-05T20:26:09.480895", "exception": false, "start_time": "2021-01-05T20:26:09.462725", "status": "completed"} tags=[]
# ## Tests different k values (k-NN)

# %% papermill={"duration": 0.03233, "end_time": "2021-01-05T20:26:09.530796", "exception": false, "start_time": "2021-01-05T20:26:09.498466", "status": "completed"} tags=[]
k_values = np.arange(10, 150 + 1, 1)
k_values_to_explore = (10, 15, 20, 30, 40, 50, 75, 100, 125, 150)  # , 175, 200)

# %% papermill={"duration": 5.063976, "end_time": "2021-01-05T20:26:14.612505", "exception": false, "start_time": "2021-01-05T20:26:09.548529", "status": "completed"} tags=[]
results = {}

for k in k_values_to_explore:
    nbrs = NearestNeighbors(n_neighbors=k, n_jobs=N_JOBS).fit(data)
    distances, indices = nbrs.kneighbors(data)
    results[k] = (distances, indices)

# %% papermill={"duration": 0.0334, "end_time": "2021-01-05T20:26:14.663939", "exception": false, "start_time": "2021-01-05T20:26:14.630539", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.909554, "end_time": "2021-01-05T20:26:15.591687", "exception": false, "start_time": "2021-01-05T20:26:14.682133", "status": "completed"} tags=[]
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

# %% [markdown] papermill={"duration": 0.02144, "end_time": "2021-01-05T20:26:15.635383", "exception": false, "start_time": "2021-01-05T20:26:15.613943", "status": "completed"} tags=[]
# ## Clustering

# %% [markdown] papermill={"duration": 0.021278, "end_time": "2021-01-05T20:26:15.678257", "exception": false, "start_time": "2021-01-05T20:26:15.656979", "status": "completed"} tags=[]
# ### Generate clusterers

# %% papermill={"duration": 0.041853, "end_time": "2021-01-05T20:26:15.741299", "exception": false, "start_time": "2021-01-05T20:26:15.699446", "status": "completed"} tags=[]
CLUSTERING_OPTIONS = {}

# K_RANGE is the min_samples parameter in DBSCAN (sklearn)
CLUSTERING_OPTIONS["K_RANGE"] = k_values
CLUSTERING_OPTIONS["EPS_RANGE_PER_K"] = eps_range_per_k
CLUSTERING_OPTIONS["EPS_STEP"] = 33
CLUSTERING_OPTIONS["METRIC"] = "euclidean"

display(CLUSTERING_OPTIONS)

# %% papermill={"duration": 0.052928, "end_time": "2021-01-05T20:26:15.816069", "exception": false, "start_time": "2021-01-05T20:26:15.763141", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.037044, "end_time": "2021-01-05T20:26:15.875427", "exception": false, "start_time": "2021-01-05T20:26:15.838383", "status": "completed"} tags=[]
display(len(CLUSTERERS))

# %% papermill={"duration": 0.040589, "end_time": "2021-01-05T20:26:15.939564", "exception": false, "start_time": "2021-01-05T20:26:15.898975", "status": "completed"} tags=[]
_iter = iter(CLUSTERERS.items())
display(next(_iter))
display(next(_iter))

# %% papermill={"duration": 0.038168, "end_time": "2021-01-05T20:26:16.001904", "exception": false, "start_time": "2021-01-05T20:26:15.963736", "status": "completed"} tags=[]
clustering_method_name = method_name
display(clustering_method_name)

# %% [markdown] papermill={"duration": 0.022658, "end_time": "2021-01-05T20:26:16.047695", "exception": false, "start_time": "2021-01-05T20:26:16.025037", "status": "completed"} tags=[]
# ### Generate ensemble

# %% papermill={"duration": 0.180828, "end_time": "2021-01-05T20:26:16.251003", "exception": false, "start_time": "2021-01-05T20:26:16.070175", "status": "completed"} tags=[]
data_dist = pairwise_distances(data, metric=CLUSTERING_OPTIONS["METRIC"])

# %% papermill={"duration": 0.038605, "end_time": "2021-01-05T20:26:16.313595", "exception": false, "start_time": "2021-01-05T20:26:16.274990", "status": "completed"} tags=[]
data_dist.shape

# %% papermill={"duration": 0.469592, "end_time": "2021-01-05T20:26:16.807289", "exception": false, "start_time": "2021-01-05T20:26:16.337697", "status": "completed"} tags=[]
pd.Series(data_dist.flatten()).describe().apply(str)

# %% papermill={"duration": 782.388032, "end_time": "2021-01-05T20:39:19.218821", "exception": false, "start_time": "2021-01-05T20:26:16.830789", "status": "completed"} tags=[]
ensemble = generate_ensemble(
    data_dist,
    CLUSTERERS,
    attributes=CLUSTERING_ATTRIBUTES_TO_SAVE,
)

# %% papermill={"duration": 0.762639, "end_time": "2021-01-05T20:39:20.730843", "exception": false, "start_time": "2021-01-05T20:39:19.968204", "status": "completed"} tags=[]
# the number should be close to 295 (the number of partitions generated by k-means/spectral clustering)
ensemble.shape

# %% papermill={"duration": 0.775838, "end_time": "2021-01-05T20:39:22.280427", "exception": false, "start_time": "2021-01-05T20:39:21.504589", "status": "completed"} tags=[]
ensemble.head()

# %% papermill={"duration": 0.765918, "end_time": "2021-01-05T20:39:23.804168", "exception": false, "start_time": "2021-01-05T20:39:23.038250", "status": "completed"} tags=[]
ensemble["n_clusters"].value_counts().head()

# %% papermill={"duration": 0.793465, "end_time": "2021-01-05T20:39:25.352953", "exception": false, "start_time": "2021-01-05T20:39:24.559488", "status": "completed"} tags=[]
ensemble_stats = ensemble["n_clusters"].describe()
display(ensemble_stats)

# %% [markdown] papermill={"duration": 0.752044, "end_time": "2021-01-05T20:39:26.858056", "exception": false, "start_time": "2021-01-05T20:39:26.106012", "status": "completed"} tags=[]
# ### Testing

# %% papermill={"duration": 0.791586, "end_time": "2021-01-05T20:39:28.398993", "exception": false, "start_time": "2021-01-05T20:39:27.607407", "status": "completed"} tags=[]
assert ensemble_stats["min"] > 1

# %% papermill={"duration": 0.754321, "end_time": "2021-01-05T20:39:29.908798", "exception": false, "start_time": "2021-01-05T20:39:29.154477", "status": "completed"} tags=[]
assert not ensemble["n_clusters"].isna().any()

# %% papermill={"duration": 0.761919, "end_time": "2021-01-05T20:39:31.418603", "exception": false, "start_time": "2021-01-05T20:39:30.656684", "status": "completed"} tags=[]
# assert ensemble.shape[0] == len(CLUSTERERS)

# %% papermill={"duration": 0.818088, "end_time": "2021-01-05T20:39:33.011421", "exception": false, "start_time": "2021-01-05T20:39:32.193333", "status": "completed"} tags=[]
# all partitions have the right size
assert np.all(
    [part["partition"].shape[0] == data.shape[0] for idx, part in ensemble.iterrows()]
)

# %% papermill={"duration": 0.837323, "end_time": "2021-01-05T20:39:34.595239", "exception": false, "start_time": "2021-01-05T20:39:33.757916", "status": "completed"} tags=[]
# no partition has negative clusters (noisy points)
assert not np.any([(part["partition"] < 0).any() for idx, part in ensemble.iterrows()])

# %% [markdown] papermill={"duration": 0.784451, "end_time": "2021-01-05T20:39:36.132800", "exception": false, "start_time": "2021-01-05T20:39:35.348349", "status": "completed"} tags=[]
# ### Save

# %% papermill={"duration": 0.771671, "end_time": "2021-01-05T20:39:37.651712", "exception": false, "start_time": "2021-01-05T20:39:36.880041", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.786839, "end_time": "2021-01-05T20:39:39.186586", "exception": false, "start_time": "2021-01-05T20:39:38.399747", "status": "completed"} tags=[]
ensemble.to_pickle(output_filename)

# %% papermill={"duration": 0.753594, "end_time": "2021-01-05T20:39:40.716580", "exception": false, "start_time": "2021-01-05T20:39:39.962986", "status": "completed"} tags=[]
