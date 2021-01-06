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

# %% [markdown] papermill={"duration": 0.051221, "end_time": "2021-01-05T20:39:46.508507", "exception": false, "start_time": "2021-01-05T20:39:46.457286", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.01489, "end_time": "2021-01-05T20:39:46.540350", "exception": false, "start_time": "2021-01-05T20:39:46.525460", "status": "completed"} tags=[]
# Runs hierarchical clustering on the z_score_std version of the data.

# %% [markdown] papermill={"duration": 0.014936, "end_time": "2021-01-05T20:39:46.570701", "exception": false, "start_time": "2021-01-05T20:39:46.555765", "status": "completed"} tags=[]
# # Environment variables

# %% papermill={"duration": 0.029106, "end_time": "2021-01-05T20:39:46.614752", "exception": false, "start_time": "2021-01-05T20:39:46.585646", "status": "completed"} tags=[]
from IPython.display import display

import conf

N_JOBS = conf.GENERAL["N_JOBS"]
display(N_JOBS)

# %% papermill={"duration": 0.022036, "end_time": "2021-01-05T20:39:46.652733", "exception": false, "start_time": "2021-01-05T20:39:46.630697", "status": "completed"} tags=[]
# %env MKL_NUM_THREADS=$N_JOBS
# %env OPEN_BLAS_NUM_THREADS=$N_JOBS
# %env NUMEXPR_NUM_THREADS=$N_JOBS
# %env OMP_NUM_THREADS=$N_JOBS

# %% [markdown] papermill={"duration": 0.015301, "end_time": "2021-01-05T20:39:46.684118", "exception": false, "start_time": "2021-01-05T20:39:46.668817", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.025894, "end_time": "2021-01-05T20:39:46.725573", "exception": false, "start_time": "2021-01-05T20:39:46.699679", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 1.601253, "end_time": "2021-01-05T20:39:48.342763", "exception": false, "start_time": "2021-01-05T20:39:46.741510", "status": "completed"} tags=[]
from pathlib import Path

import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns

from utils import generate_result_set_name

# %% [markdown] papermill={"duration": 0.016742, "end_time": "2021-01-05T20:39:48.377037", "exception": false, "start_time": "2021-01-05T20:39:48.360295", "status": "completed"} tags=[]
# # Settings

# %% [markdown] papermill={"duration": 0.015097, "end_time": "2021-01-05T20:39:48.407569", "exception": false, "start_time": "2021-01-05T20:39:48.392472", "status": "completed"} tags=[]
# ## Input data

# %% papermill={"duration": 0.030106, "end_time": "2021-01-05T20:39:48.453252", "exception": false, "start_time": "2021-01-05T20:39:48.423146", "status": "completed"} tags=[]
INPUT_SUBSET = "z_score_std"

# %% papermill={"duration": 0.029573, "end_time": "2021-01-05T20:39:48.498379", "exception": false, "start_time": "2021-01-05T20:39:48.468806", "status": "completed"} tags=[]
INPUT_STEM = "projection-smultixcan-efo_partial-mashr-zscores"

# %% papermill={"duration": 0.031415, "end_time": "2021-01-05T20:39:48.545770", "exception": false, "start_time": "2021-01-05T20:39:48.514355", "status": "completed"} tags=[]
input_filepath = Path(
    conf.RESULTS["DATA_TRANSFORMATIONS_DIR"],
    INPUT_SUBSET,
    f"{INPUT_SUBSET}-{INPUT_STEM}.pkl",
).resolve()
display(input_filepath)

assert input_filepath.exists(), "Input file does not exist"

input_filepath_stem = input_filepath.stem
display(input_filepath_stem)

# %% [markdown] papermill={"duration": 0.015907, "end_time": "2021-01-05T20:39:48.578286", "exception": false, "start_time": "2021-01-05T20:39:48.562379", "status": "completed"} tags=[]
# ## Clustering

# %% papermill={"duration": 0.035511, "end_time": "2021-01-05T20:39:48.629556", "exception": false, "start_time": "2021-01-05T20:39:48.594045", "status": "completed"} tags=[]
from sklearn.cluster import AgglomerativeClustering

# %% papermill={"duration": 0.03113, "end_time": "2021-01-05T20:39:48.677422", "exception": false, "start_time": "2021-01-05T20:39:48.646292", "status": "completed"} tags=[]
CLUSTERING_ATTRIBUTES_TO_SAVE = ["n_clusters"]

# %% papermill={"duration": 0.031416, "end_time": "2021-01-05T20:39:48.725491", "exception": false, "start_time": "2021-01-05T20:39:48.694075", "status": "completed"} tags=[]
CLUSTERING_OPTIONS = {}

CLUSTERING_OPTIONS["K_MIN"] = 2
CLUSTERING_OPTIONS["K_MAX"] = 75  # sqrt(3749) + some more to get closer to 295
CLUSTERING_OPTIONS["LINKAGE"] = {"ward", "complete", "average", "single"}
CLUSTERING_OPTIONS["AFFINITY"] = "euclidean"

display(CLUSTERING_OPTIONS)

# %% papermill={"duration": 0.038456, "end_time": "2021-01-05T20:39:48.781025", "exception": false, "start_time": "2021-01-05T20:39:48.742569", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.032687, "end_time": "2021-01-05T20:39:48.861401", "exception": false, "start_time": "2021-01-05T20:39:48.828714", "status": "completed"} tags=[]
display(len(CLUSTERERS))

# %% papermill={"duration": 0.032815, "end_time": "2021-01-05T20:39:48.911984", "exception": false, "start_time": "2021-01-05T20:39:48.879169", "status": "completed"} tags=[]
_iter = iter(CLUSTERERS.items())
display(next(_iter))
display(next(_iter))

# %% papermill={"duration": 0.031838, "end_time": "2021-01-05T20:39:48.961279", "exception": false, "start_time": "2021-01-05T20:39:48.929441", "status": "completed"} tags=[]
clustering_method_name = method_name
display(clustering_method_name)

# %% [markdown] papermill={"duration": 0.01705, "end_time": "2021-01-05T20:39:48.996206", "exception": false, "start_time": "2021-01-05T20:39:48.979156", "status": "completed"} tags=[]
# ## Output directory

# %% papermill={"duration": 0.032469, "end_time": "2021-01-05T20:39:49.045522", "exception": false, "start_time": "2021-01-05T20:39:49.013053", "status": "completed"} tags=[]
# output dir for this notebook
RESULTS_DIR = Path(
    conf.RESULTS["CLUSTERING_RUNS_DIR"],
    f"{INPUT_SUBSET}-{INPUT_STEM}",
).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %% [markdown] papermill={"duration": 0.016855, "end_time": "2021-01-05T20:39:49.080022", "exception": false, "start_time": "2021-01-05T20:39:49.063167", "status": "completed"} tags=[]
# # Load input file

# %% papermill={"duration": 0.045346, "end_time": "2021-01-05T20:39:49.142282", "exception": false, "start_time": "2021-01-05T20:39:49.096936", "status": "completed"} tags=[]
data = pd.read_pickle(input_filepath)

# %% papermill={"duration": 0.033303, "end_time": "2021-01-05T20:39:49.195070", "exception": false, "start_time": "2021-01-05T20:39:49.161767", "status": "completed"} tags=[]
data.shape

# %% papermill={"duration": 0.045296, "end_time": "2021-01-05T20:39:49.259282", "exception": false, "start_time": "2021-01-05T20:39:49.213986", "status": "completed"} tags=[]
data.head()

# %% papermill={"duration": 0.042824, "end_time": "2021-01-05T20:39:49.320651", "exception": false, "start_time": "2021-01-05T20:39:49.277827", "status": "completed"} tags=[]
assert not data.isna().any().any()

# %% [markdown] papermill={"duration": 0.017774, "end_time": "2021-01-05T20:39:49.377407", "exception": false, "start_time": "2021-01-05T20:39:49.359633", "status": "completed"} tags=[]
# # Clustering

# %% [markdown] papermill={"duration": 0.017478, "end_time": "2021-01-05T20:39:49.412306", "exception": false, "start_time": "2021-01-05T20:39:49.394828", "status": "completed"} tags=[]
# ## Generate ensemble

# %% papermill={"duration": 0.034939, "end_time": "2021-01-05T20:39:49.464907", "exception": false, "start_time": "2021-01-05T20:39:49.429968", "status": "completed"} tags=[]
from sklearn.metrics import pairwise_distances
from clustering.ensemble import generate_ensemble

# %% papermill={"duration": 0.294374, "end_time": "2021-01-05T20:39:49.778491", "exception": false, "start_time": "2021-01-05T20:39:49.484117", "status": "completed"} tags=[]
data_dist = pairwise_distances(data, metric=CLUSTERING_OPTIONS["AFFINITY"])

# %% papermill={"duration": 0.032921, "end_time": "2021-01-05T20:39:49.829304", "exception": false, "start_time": "2021-01-05T20:39:49.796383", "status": "completed"} tags=[]
data_dist.shape

# %% papermill={"duration": 0.617679, "end_time": "2021-01-05T20:39:50.465300", "exception": false, "start_time": "2021-01-05T20:39:49.847621", "status": "completed"} tags=[]
pd.Series(data_dist.flatten()).describe().apply(str)

# %% papermill={"duration": 1660.893496, "end_time": "2021-01-05T21:07:31.377795", "exception": false, "start_time": "2021-01-05T20:39:50.484299", "status": "completed"} tags=[]
ensemble = generate_ensemble(
    data_dist,
    CLUSTERERS,
    attributes=CLUSTERING_ATTRIBUTES_TO_SAVE,
    affinity_matrix=data_dist,
)

# %% papermill={"duration": 0.103932, "end_time": "2021-01-05T21:07:31.549146", "exception": false, "start_time": "2021-01-05T21:07:31.445214", "status": "completed"} tags=[]
# the number should be close to 295 (the number of partitions generated by k-means/spectral clustering)
ensemble.shape

# %% papermill={"duration": 0.088093, "end_time": "2021-01-05T21:07:31.706327", "exception": false, "start_time": "2021-01-05T21:07:31.618234", "status": "completed"} tags=[]
ensemble.head()

# %% papermill={"duration": 0.085453, "end_time": "2021-01-05T21:07:31.859548", "exception": false, "start_time": "2021-01-05T21:07:31.774095", "status": "completed"} tags=[]
ensemble["n_clusters"].value_counts().head()

# %% papermill={"duration": 0.085014, "end_time": "2021-01-05T21:07:32.012989", "exception": false, "start_time": "2021-01-05T21:07:31.927975", "status": "completed"} tags=[]
ensemble_stats = ensemble["n_clusters"].describe()
display(ensemble_stats)

# %% [markdown] papermill={"duration": 0.06758, "end_time": "2021-01-05T21:07:32.148402", "exception": false, "start_time": "2021-01-05T21:07:32.080822", "status": "completed"} tags=[]
# ### Testing

# %% papermill={"duration": 0.081615, "end_time": "2021-01-05T21:07:32.297935", "exception": false, "start_time": "2021-01-05T21:07:32.216320", "status": "completed"} tags=[]
assert ensemble_stats["min"] > 1

# %% papermill={"duration": 0.081668, "end_time": "2021-01-05T21:07:32.446641", "exception": false, "start_time": "2021-01-05T21:07:32.364973", "status": "completed"} tags=[]
assert not ensemble["n_clusters"].isna().any()

# %% papermill={"duration": 0.081435, "end_time": "2021-01-05T21:07:32.595362", "exception": false, "start_time": "2021-01-05T21:07:32.513927", "status": "completed"} tags=[]
assert ensemble.shape[0] == len(CLUSTERERS)

# %% papermill={"duration": 0.100382, "end_time": "2021-01-05T21:07:32.763438", "exception": false, "start_time": "2021-01-05T21:07:32.663056", "status": "completed"} tags=[]
# all partitions have the right size
assert np.all(
    [part["partition"].shape[0] == data.shape[0] for idx, part in ensemble.iterrows()]
)

# %% papermill={"duration": 0.101846, "end_time": "2021-01-05T21:07:32.934206", "exception": false, "start_time": "2021-01-05T21:07:32.832360", "status": "completed"} tags=[]
# no partition has negative clusters (noisy points)
assert not np.any([(part["partition"] < 0).any() for idx, part in ensemble.iterrows()])

# %% [markdown] papermill={"duration": 0.066139, "end_time": "2021-01-05T21:07:33.067982", "exception": false, "start_time": "2021-01-05T21:07:33.001843", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.08241, "end_time": "2021-01-05T21:07:33.216950", "exception": false, "start_time": "2021-01-05T21:07:33.134540", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.088531, "end_time": "2021-01-05T21:07:33.373026", "exception": false, "start_time": "2021-01-05T21:07:33.284495", "status": "completed"} tags=[]
ensemble.to_pickle(output_filename)

# %% papermill={"duration": 0.066559, "end_time": "2021-01-05T21:07:33.506784", "exception": false, "start_time": "2021-01-05T21:07:33.440225", "status": "completed"} tags=[]
