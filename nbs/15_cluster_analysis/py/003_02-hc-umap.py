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

# %% [markdown] papermill={"duration": 0.050085, "end_time": "2021-01-05T21:35:21.754084", "exception": false, "start_time": "2021-01-05T21:35:21.703999", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.015139, "end_time": "2021-01-05T21:35:21.786005", "exception": false, "start_time": "2021-01-05T21:35:21.770866", "status": "completed"} tags=[]
# Runs hierarchical clustering on the umap version of the data.

# %% [markdown] papermill={"duration": 0.015513, "end_time": "2021-01-05T21:35:21.817021", "exception": false, "start_time": "2021-01-05T21:35:21.801508", "status": "completed"} tags=[]
# # Environment variables

# %% papermill={"duration": 0.029982, "end_time": "2021-01-05T21:35:21.862244", "exception": false, "start_time": "2021-01-05T21:35:21.832262", "status": "completed"} tags=[]
from IPython.display import display

import conf

N_JOBS = conf.GENERAL["N_JOBS"]
display(N_JOBS)

# %% papermill={"duration": 0.022378, "end_time": "2021-01-05T21:35:21.900407", "exception": false, "start_time": "2021-01-05T21:35:21.878029", "status": "completed"} tags=[]
# %env MKL_NUM_THREADS=$N_JOBS
# %env OPEN_BLAS_NUM_THREADS=$N_JOBS
# %env NUMEXPR_NUM_THREADS=$N_JOBS
# %env OMP_NUM_THREADS=$N_JOBS

# %% [markdown] papermill={"duration": 0.015969, "end_time": "2021-01-05T21:35:21.933069", "exception": false, "start_time": "2021-01-05T21:35:21.917100", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.026412, "end_time": "2021-01-05T21:35:21.975192", "exception": false, "start_time": "2021-01-05T21:35:21.948780", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 1.598108, "end_time": "2021-01-05T21:35:23.589511", "exception": false, "start_time": "2021-01-05T21:35:21.991403", "status": "completed"} tags=[]
from pathlib import Path

import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns

from utils import generate_result_set_name

# %% [markdown] papermill={"duration": 0.016666, "end_time": "2021-01-05T21:35:23.623738", "exception": false, "start_time": "2021-01-05T21:35:23.607072", "status": "completed"} tags=[]
# # Settings

# %% [markdown] papermill={"duration": 0.01551, "end_time": "2021-01-05T21:35:23.654951", "exception": false, "start_time": "2021-01-05T21:35:23.639441", "status": "completed"} tags=[]
# ## Input data

# %% papermill={"duration": 0.029817, "end_time": "2021-01-05T21:35:23.700596", "exception": false, "start_time": "2021-01-05T21:35:23.670779", "status": "completed"} tags=[]
INPUT_SUBSET = "umap"

# %% papermill={"duration": 0.029397, "end_time": "2021-01-05T21:35:23.746354", "exception": false, "start_time": "2021-01-05T21:35:23.716957", "status": "completed"} tags=[]
INPUT_STEM = "z_score_std-projection-smultixcan-efo_partial-mashr-zscores"

# %% papermill={"duration": 0.029512, "end_time": "2021-01-05T21:35:23.792243", "exception": false, "start_time": "2021-01-05T21:35:23.762731", "status": "completed"} tags=[]
# parameters of the dimentionality reduction steps
DR_OPTIONS = {
    "n_components": 50,
    "metric": "euclidean",
    "n_neighbors": 15,
    "random_state": 0,
}

# %% papermill={"duration": 0.032932, "end_time": "2021-01-05T21:35:23.841775", "exception": false, "start_time": "2021-01-05T21:35:23.808843", "status": "completed"} tags=[]
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

# %% [markdown] papermill={"duration": 0.016267, "end_time": "2021-01-05T21:35:23.874815", "exception": false, "start_time": "2021-01-05T21:35:23.858548", "status": "completed"} tags=[]
# ## Clustering

# %% papermill={"duration": 0.03705, "end_time": "2021-01-05T21:35:23.927905", "exception": false, "start_time": "2021-01-05T21:35:23.890855", "status": "completed"} tags=[]
from sklearn.cluster import AgglomerativeClustering

# %% papermill={"duration": 0.029859, "end_time": "2021-01-05T21:35:23.974288", "exception": false, "start_time": "2021-01-05T21:35:23.944429", "status": "completed"} tags=[]
CLUSTERING_ATTRIBUTES_TO_SAVE = ["n_clusters"]

# %% papermill={"duration": 0.032354, "end_time": "2021-01-05T21:35:24.023025", "exception": false, "start_time": "2021-01-05T21:35:23.990671", "status": "completed"} tags=[]
CLUSTERING_OPTIONS = {}

CLUSTERING_OPTIONS["K_MIN"] = 2
CLUSTERING_OPTIONS["K_MAX"] = 75  # sqrt(3749) + some more to get closer to 295
CLUSTERING_OPTIONS["LINKAGE"] = {"ward", "complete", "average", "single"}
CLUSTERING_OPTIONS["AFFINITY"] = "euclidean"

display(CLUSTERING_OPTIONS)

# %% papermill={"duration": 0.033738, "end_time": "2021-01-05T21:35:24.073981", "exception": false, "start_time": "2021-01-05T21:35:24.040243", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.031517, "end_time": "2021-01-05T21:35:24.122639", "exception": false, "start_time": "2021-01-05T21:35:24.091122", "status": "completed"} tags=[]
display(len(CLUSTERERS))

# %% papermill={"duration": 0.033008, "end_time": "2021-01-05T21:35:24.172319", "exception": false, "start_time": "2021-01-05T21:35:24.139311", "status": "completed"} tags=[]
_iter = iter(CLUSTERERS.items())
display(next(_iter))
display(next(_iter))

# %% papermill={"duration": 0.031986, "end_time": "2021-01-05T21:35:24.222021", "exception": false, "start_time": "2021-01-05T21:35:24.190035", "status": "completed"} tags=[]
clustering_method_name = method_name
display(clustering_method_name)

# %% [markdown] papermill={"duration": 0.017027, "end_time": "2021-01-05T21:35:24.256307", "exception": false, "start_time": "2021-01-05T21:35:24.239280", "status": "completed"} tags=[]
# ## Output directory

# %% papermill={"duration": 0.032704, "end_time": "2021-01-05T21:35:24.306056", "exception": false, "start_time": "2021-01-05T21:35:24.273352", "status": "completed"} tags=[]
# output dir for this notebook
RESULTS_DIR = Path(
    conf.RESULTS["CLUSTERING_RUNS_DIR"],
    f"{INPUT_SUBSET}-{INPUT_STEM}",
).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %% [markdown] papermill={"duration": 0.017321, "end_time": "2021-01-05T21:35:24.341245", "exception": false, "start_time": "2021-01-05T21:35:24.323924", "status": "completed"} tags=[]
# # Load input file

# %% papermill={"duration": 0.033246, "end_time": "2021-01-05T21:35:24.391651", "exception": false, "start_time": "2021-01-05T21:35:24.358405", "status": "completed"} tags=[]
data = pd.read_pickle(input_filepath)

# %% papermill={"duration": 0.034562, "end_time": "2021-01-05T21:35:24.444401", "exception": false, "start_time": "2021-01-05T21:35:24.409839", "status": "completed"} tags=[]
data.shape

# %% papermill={"duration": 0.045319, "end_time": "2021-01-05T21:35:24.508236", "exception": false, "start_time": "2021-01-05T21:35:24.462917", "status": "completed"} tags=[]
data.head()

# %% papermill={"duration": 0.032419, "end_time": "2021-01-05T21:35:24.559473", "exception": false, "start_time": "2021-01-05T21:35:24.527054", "status": "completed"} tags=[]
assert not data.isna().any().any()

# %% [markdown] papermill={"duration": 0.018147, "end_time": "2021-01-05T21:35:24.596394", "exception": false, "start_time": "2021-01-05T21:35:24.578247", "status": "completed"} tags=[]
# # Clustering

# %% [markdown] papermill={"duration": 0.017616, "end_time": "2021-01-05T21:35:24.632003", "exception": false, "start_time": "2021-01-05T21:35:24.614387", "status": "completed"} tags=[]
# ## Generate ensemble

# %% papermill={"duration": 0.03457, "end_time": "2021-01-05T21:35:24.684323", "exception": false, "start_time": "2021-01-05T21:35:24.649753", "status": "completed"} tags=[]
from sklearn.metrics import pairwise_distances
from clustering.ensemble import generate_ensemble

# %% papermill={"duration": 0.178101, "end_time": "2021-01-05T21:35:24.880813", "exception": false, "start_time": "2021-01-05T21:35:24.702712", "status": "completed"} tags=[]
data_dist = pairwise_distances(data, metric=CLUSTERING_OPTIONS["AFFINITY"])

# %% papermill={"duration": 0.033516, "end_time": "2021-01-05T21:35:24.933270", "exception": false, "start_time": "2021-01-05T21:35:24.899754", "status": "completed"} tags=[]
data_dist.shape

# %% papermill={"duration": 0.473837, "end_time": "2021-01-05T21:35:25.425468", "exception": false, "start_time": "2021-01-05T21:35:24.951631", "status": "completed"} tags=[]
pd.Series(data_dist.flatten()).describe().apply(str)

# %% papermill={"duration": 1658.998471, "end_time": "2021-01-05T22:03:04.444211", "exception": false, "start_time": "2021-01-05T21:35:25.445740", "status": "completed"} tags=[]
ensemble = generate_ensemble(
    data_dist,
    CLUSTERERS,
    attributes=CLUSTERING_ATTRIBUTES_TO_SAVE,
    affinity_matrix=data_dist,
)

# %% papermill={"duration": 0.103225, "end_time": "2021-01-05T22:03:04.612447", "exception": false, "start_time": "2021-01-05T22:03:04.509222", "status": "completed"} tags=[]
# the number should be close to 295 (the number of partitions generated by k-means/spectral clustering)
ensemble.shape

# %% papermill={"duration": 0.088016, "end_time": "2021-01-05T22:03:04.769619", "exception": false, "start_time": "2021-01-05T22:03:04.681603", "status": "completed"} tags=[]
ensemble.head()

# %% papermill={"duration": 0.081496, "end_time": "2021-01-05T22:03:04.918329", "exception": false, "start_time": "2021-01-05T22:03:04.836833", "status": "completed"} tags=[]
ensemble["n_clusters"].value_counts().head()

# %% papermill={"duration": 0.087058, "end_time": "2021-01-05T22:03:05.073488", "exception": false, "start_time": "2021-01-05T22:03:04.986430", "status": "completed"} tags=[]
ensemble_stats = ensemble["n_clusters"].describe()
display(ensemble_stats)

# %% [markdown] papermill={"duration": 0.067222, "end_time": "2021-01-05T22:03:05.209217", "exception": false, "start_time": "2021-01-05T22:03:05.141995", "status": "completed"} tags=[]
# ### Testing

# %% papermill={"duration": 0.081025, "end_time": "2021-01-05T22:03:05.356928", "exception": false, "start_time": "2021-01-05T22:03:05.275903", "status": "completed"} tags=[]
assert ensemble_stats["min"] > 1

# %% papermill={"duration": 0.081244, "end_time": "2021-01-05T22:03:05.505224", "exception": false, "start_time": "2021-01-05T22:03:05.423980", "status": "completed"} tags=[]
assert not ensemble["n_clusters"].isna().any()

# %% papermill={"duration": 0.08051, "end_time": "2021-01-05T22:03:05.652241", "exception": false, "start_time": "2021-01-05T22:03:05.571731", "status": "completed"} tags=[]
assert ensemble.shape[0] == len(CLUSTERERS)

# %% papermill={"duration": 0.098642, "end_time": "2021-01-05T22:03:05.817418", "exception": false, "start_time": "2021-01-05T22:03:05.718776", "status": "completed"} tags=[]
# all partitions have the right size
assert np.all(
    [part["partition"].shape[0] == data.shape[0] for idx, part in ensemble.iterrows()]
)

# %% papermill={"duration": 0.102904, "end_time": "2021-01-05T22:03:05.986926", "exception": false, "start_time": "2021-01-05T22:03:05.884022", "status": "completed"} tags=[]
# no partition has negative clusters (noisy points)
assert not np.any([(part["partition"] < 0).any() for idx, part in ensemble.iterrows()])

# %% [markdown] papermill={"duration": 0.065966, "end_time": "2021-01-05T22:03:06.122924", "exception": false, "start_time": "2021-01-05T22:03:06.056958", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.081342, "end_time": "2021-01-05T22:03:06.270193", "exception": false, "start_time": "2021-01-05T22:03:06.188851", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.087691, "end_time": "2021-01-05T22:03:06.424355", "exception": false, "start_time": "2021-01-05T22:03:06.336664", "status": "completed"} tags=[]
ensemble.to_pickle(output_filename)

# %% papermill={"duration": 0.066472, "end_time": "2021-01-05T22:03:06.557830", "exception": false, "start_time": "2021-01-05T22:03:06.491358", "status": "completed"} tags=[]
