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

# %% [markdown] papermill={"duration": 0.050099, "end_time": "2020-12-03T01:22:20.867420", "exception": false, "start_time": "2020-12-03T01:22:20.817321", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.014853, "end_time": "2020-12-03T01:22:20.899928", "exception": false, "start_time": "2020-12-03T01:22:20.885075", "status": "completed"} tags=[]
# Runs hierarchical clustering on the z_score_std version of the data.

# %% [markdown] papermill={"duration": 0.014957, "end_time": "2020-12-03T01:22:20.929911", "exception": false, "start_time": "2020-12-03T01:22:20.914954", "status": "completed"} tags=[]
# # Environment variables

# %% papermill={"duration": 0.029933, "end_time": "2020-12-03T01:22:20.974724", "exception": false, "start_time": "2020-12-03T01:22:20.944791", "status": "completed"} tags=[]
from IPython.display import display

import conf

N_JOBS = conf.GENERAL["N_JOBS"]
display(N_JOBS)

# %% papermill={"duration": 0.022636, "end_time": "2020-12-03T01:22:21.013529", "exception": false, "start_time": "2020-12-03T01:22:20.990893", "status": "completed"} tags=[]
# %env MKL_NUM_THREADS=$N_JOBS
# %env OPEN_BLAS_NUM_THREADS=$N_JOBS
# %env NUMEXPR_NUM_THREADS=$N_JOBS
# %env OMP_NUM_THREADS=$N_JOBS

# %% [markdown] papermill={"duration": 0.015495, "end_time": "2020-12-03T01:22:21.044912", "exception": false, "start_time": "2020-12-03T01:22:21.029417", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.026143, "end_time": "2020-12-03T01:22:21.086285", "exception": false, "start_time": "2020-12-03T01:22:21.060142", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 1.622311, "end_time": "2020-12-03T01:22:22.724430", "exception": false, "start_time": "2020-12-03T01:22:21.102119", "status": "completed"} tags=[]
from pathlib import Path

import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns

from utils import generate_result_set_name

# %% [markdown] papermill={"duration": 0.015408, "end_time": "2020-12-03T01:22:22.757694", "exception": false, "start_time": "2020-12-03T01:22:22.742286", "status": "completed"} tags=[]
# # Settings

# %% [markdown] papermill={"duration": 0.015225, "end_time": "2020-12-03T01:22:22.788164", "exception": false, "start_time": "2020-12-03T01:22:22.772939", "status": "completed"} tags=[]
# ## Input data

# %% papermill={"duration": 0.029929, "end_time": "2020-12-03T01:22:22.833586", "exception": false, "start_time": "2020-12-03T01:22:22.803657", "status": "completed"} tags=[]
INPUT_SUBSET = "z_score_std"

# %% papermill={"duration": 0.02982, "end_time": "2020-12-03T01:22:22.879235", "exception": false, "start_time": "2020-12-03T01:22:22.849415", "status": "completed"} tags=[]
INPUT_STEM = "projection-smultixcan-efo_partial-mashr-zscores"

# %% papermill={"duration": 0.031662, "end_time": "2020-12-03T01:22:22.926688", "exception": false, "start_time": "2020-12-03T01:22:22.895026", "status": "completed"} tags=[]
input_filepath = Path(
    conf.RESULTS["DATA_TRANSFORMATIONS_DIR"],
    INPUT_SUBSET,
    f"{INPUT_SUBSET}-{INPUT_STEM}.pkl",
).resolve()
display(input_filepath)

assert input_filepath.exists(), "Input file does not exist"

input_filepath_stem = input_filepath.stem
display(input_filepath_stem)

# %% [markdown] papermill={"duration": 0.015728, "end_time": "2020-12-03T01:22:22.958898", "exception": false, "start_time": "2020-12-03T01:22:22.943170", "status": "completed"} tags=[]
# ## Clustering

# %% papermill={"duration": 0.035755, "end_time": "2020-12-03T01:22:23.010169", "exception": false, "start_time": "2020-12-03T01:22:22.974414", "status": "completed"} tags=[]
from sklearn.cluster import AgglomerativeClustering

# %% papermill={"duration": 0.030346, "end_time": "2020-12-03T01:22:23.056960", "exception": false, "start_time": "2020-12-03T01:22:23.026614", "status": "completed"} tags=[]
CLUSTERING_ATTRIBUTES_TO_SAVE = ["n_clusters"]

# %% papermill={"duration": 0.031518, "end_time": "2020-12-03T01:22:23.105159", "exception": false, "start_time": "2020-12-03T01:22:23.073641", "status": "completed"} tags=[]
CLUSTERING_OPTIONS = {}

CLUSTERING_OPTIONS["K_MIN"] = 2
CLUSTERING_OPTIONS["K_MAX"] = 75  # sqrt(3749) + some more to get closer to 295
CLUSTERING_OPTIONS["LINKAGE"] = {"ward", "complete", "average", "single"}
CLUSTERING_OPTIONS["AFFINITY"] = "euclidean"

display(CLUSTERING_OPTIONS)

# %% papermill={"duration": 0.032196, "end_time": "2020-12-03T01:22:23.154154", "exception": false, "start_time": "2020-12-03T01:22:23.121958", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.033664, "end_time": "2020-12-03T01:22:23.204344", "exception": false, "start_time": "2020-12-03T01:22:23.170680", "status": "completed"} tags=[]
display(len(CLUSTERERS))

# %% papermill={"duration": 0.033373, "end_time": "2020-12-03T01:22:23.255465", "exception": false, "start_time": "2020-12-03T01:22:23.222092", "status": "completed"} tags=[]
_iter = iter(CLUSTERERS.items())
display(next(_iter))
display(next(_iter))

# %% papermill={"duration": 0.031287, "end_time": "2020-12-03T01:22:23.304226", "exception": false, "start_time": "2020-12-03T01:22:23.272939", "status": "completed"} tags=[]
clustering_method_name = method_name
display(clustering_method_name)

# %% [markdown] papermill={"duration": 0.016895, "end_time": "2020-12-03T01:22:23.338192", "exception": false, "start_time": "2020-12-03T01:22:23.321297", "status": "completed"} tags=[]
# ## Output directory

# %% papermill={"duration": 0.032462, "end_time": "2020-12-03T01:22:23.387934", "exception": false, "start_time": "2020-12-03T01:22:23.355472", "status": "completed"} tags=[]
# output dir for this notebook
RESULTS_DIR = Path(
    conf.RESULTS["CLUSTERING_RUNS_DIR"],
    f"{INPUT_SUBSET}-{INPUT_STEM}",
).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %% [markdown] papermill={"duration": 0.017142, "end_time": "2020-12-03T01:22:23.422608", "exception": false, "start_time": "2020-12-03T01:22:23.405466", "status": "completed"} tags=[]
# # Load input file

# %% papermill={"duration": 0.044112, "end_time": "2020-12-03T01:22:23.483687", "exception": false, "start_time": "2020-12-03T01:22:23.439575", "status": "completed"} tags=[]
data = pd.read_pickle(input_filepath)

# %% papermill={"duration": 0.033322, "end_time": "2020-12-03T01:22:23.536222", "exception": false, "start_time": "2020-12-03T01:22:23.502900", "status": "completed"} tags=[]
data.shape

# %% papermill={"duration": 0.04549, "end_time": "2020-12-03T01:22:23.600132", "exception": false, "start_time": "2020-12-03T01:22:23.554642", "status": "completed"} tags=[]
data.head()

# %% papermill={"duration": 0.037288, "end_time": "2020-12-03T01:22:23.656825", "exception": false, "start_time": "2020-12-03T01:22:23.619537", "status": "completed"} tags=[]
assert not data.isna().any().any()

# %% [markdown] papermill={"duration": 0.017688, "end_time": "2020-12-03T01:22:23.692534", "exception": false, "start_time": "2020-12-03T01:22:23.674846", "status": "completed"} tags=[]
# # Clustering

# %% [markdown] papermill={"duration": 0.017728, "end_time": "2020-12-03T01:22:23.728204", "exception": false, "start_time": "2020-12-03T01:22:23.710476", "status": "completed"} tags=[]
# ## Generate ensemble

# %% papermill={"duration": 0.034498, "end_time": "2020-12-03T01:22:23.780132", "exception": false, "start_time": "2020-12-03T01:22:23.745634", "status": "completed"} tags=[]
from sklearn.metrics import pairwise_distances
from clustering.ensemble import generate_ensemble

# %% papermill={"duration": 0.294188, "end_time": "2020-12-03T01:22:24.093186", "exception": false, "start_time": "2020-12-03T01:22:23.798998", "status": "completed"} tags=[]
data_dist = pairwise_distances(data, metric=CLUSTERING_OPTIONS["AFFINITY"])

# %% papermill={"duration": 0.032949, "end_time": "2020-12-03T01:22:24.143937", "exception": false, "start_time": "2020-12-03T01:22:24.110988", "status": "completed"} tags=[]
data_dist.shape

# %% papermill={"duration": 0.626109, "end_time": "2020-12-03T01:22:24.788043", "exception": false, "start_time": "2020-12-03T01:22:24.161934", "status": "completed"} tags=[]
pd.Series(data_dist.flatten()).describe().apply(str)

# %% papermill={"duration": 1654.13474, "end_time": "2020-12-03T01:49:58.941663", "exception": false, "start_time": "2020-12-03T01:22:24.806923", "status": "completed"} tags=[]
ensemble = generate_ensemble(
    data_dist,
    CLUSTERERS,
    attributes=CLUSTERING_ATTRIBUTES_TO_SAVE,
    affinity_matrix=data_dist,
)

# %% papermill={"duration": 0.083174, "end_time": "2020-12-03T01:49:59.091644", "exception": false, "start_time": "2020-12-03T01:49:59.008470", "status": "completed"} tags=[]
# the number should be close to 295 (the number of partitions generated by k-means/spectral clustering)
ensemble.shape

# %% papermill={"duration": 0.090336, "end_time": "2020-12-03T01:49:59.250687", "exception": false, "start_time": "2020-12-03T01:49:59.160351", "status": "completed"} tags=[]
ensemble.head()

# %% papermill={"duration": 0.085887, "end_time": "2020-12-03T01:49:59.413650", "exception": false, "start_time": "2020-12-03T01:49:59.327763", "status": "completed"} tags=[]
ensemble["n_clusters"].value_counts().head()

# %% papermill={"duration": 0.087755, "end_time": "2020-12-03T01:49:59.570255", "exception": false, "start_time": "2020-12-03T01:49:59.482500", "status": "completed"} tags=[]
ensemble_stats = ensemble["n_clusters"].describe()
display(ensemble_stats)

# %% [markdown] papermill={"duration": 0.068278, "end_time": "2020-12-03T01:49:59.708371", "exception": false, "start_time": "2020-12-03T01:49:59.640093", "status": "completed"} tags=[]
# ### Testing

# %% papermill={"duration": 0.086904, "end_time": "2020-12-03T01:49:59.863866", "exception": false, "start_time": "2020-12-03T01:49:59.776962", "status": "completed"} tags=[]
assert ensemble_stats["min"] > 1

# %% papermill={"duration": 0.083446, "end_time": "2020-12-03T01:50:00.016274", "exception": false, "start_time": "2020-12-03T01:49:59.932828", "status": "completed"} tags=[]
assert not ensemble["n_clusters"].isna().any()

# %% papermill={"duration": 0.083459, "end_time": "2020-12-03T01:50:00.168216", "exception": false, "start_time": "2020-12-03T01:50:00.084757", "status": "completed"} tags=[]
assert ensemble.shape[0] == len(CLUSTERERS)

# %% papermill={"duration": 0.100901, "end_time": "2020-12-03T01:50:00.337623", "exception": false, "start_time": "2020-12-03T01:50:00.236722", "status": "completed"} tags=[]
# all partitions have the right size
assert np.all(
    [part["partition"].shape[0] == data.shape[0] for idx, part in ensemble.iterrows()]
)

# %% papermill={"duration": 0.103009, "end_time": "2020-12-03T01:50:00.509614", "exception": false, "start_time": "2020-12-03T01:50:00.406605", "status": "completed"} tags=[]
# no partition has negative clusters (noisy points)
assert not np.any([(part["partition"] < 0).any() for idx, part in ensemble.iterrows()])

# %% [markdown] papermill={"duration": 0.068033, "end_time": "2020-12-03T01:50:00.646725", "exception": false, "start_time": "2020-12-03T01:50:00.578692", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.084549, "end_time": "2020-12-03T01:50:00.798807", "exception": false, "start_time": "2020-12-03T01:50:00.714258", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.107093, "end_time": "2020-12-03T01:50:00.973907", "exception": false, "start_time": "2020-12-03T01:50:00.866814", "status": "completed"} tags=[]
ensemble.to_pickle(output_filename)

# %% papermill={"duration": 0.067876, "end_time": "2020-12-03T01:50:01.121874", "exception": false, "start_time": "2020-12-03T01:50:01.053998", "status": "completed"} tags=[]
