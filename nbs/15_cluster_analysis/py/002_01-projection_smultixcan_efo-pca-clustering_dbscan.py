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

# %% [markdown] papermill={"duration": 0.044329, "end_time": "2020-12-02T17:57:32.521429", "exception": false, "start_time": "2020-12-02T17:57:32.477100", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.014408, "end_time": "2020-12-02T17:57:32.552152", "exception": false, "start_time": "2020-12-02T17:57:32.537744", "status": "completed"} tags=[]
# It runs DBSCAN on the pca version of the data.
#
# The notebook explores different values for min_samples and eps (the main parameters of DBSCAN).

# %% [markdown] papermill={"duration": 0.013395, "end_time": "2020-12-02T17:57:32.579311", "exception": false, "start_time": "2020-12-02T17:57:32.565916", "status": "completed"} tags=[]
# # Environment variables

# %% papermill={"duration": 0.027677, "end_time": "2020-12-02T17:57:32.620688", "exception": false, "start_time": "2020-12-02T17:57:32.593011", "status": "completed"} tags=[]
from IPython.display import display

import conf

N_JOBS = conf.GENERAL['N_JOBS']
display(N_JOBS)

# %% papermill={"duration": 0.020341, "end_time": "2020-12-02T17:57:32.655854", "exception": false, "start_time": "2020-12-02T17:57:32.635513", "status": "completed"} tags=[]
# %env MKL_NUM_THREADS=$N_JOBS
# %env OPEN_BLAS_NUM_THREADS=$N_JOBS
# %env NUMEXPR_NUM_THREADS=$N_JOBS
# %env OMP_NUM_THREADS=$N_JOBS

# %% [markdown] papermill={"duration": 0.014096, "end_time": "2020-12-02T17:57:32.684659", "exception": false, "start_time": "2020-12-02T17:57:32.670563", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.024641, "end_time": "2020-12-02T17:57:32.723473", "exception": false, "start_time": "2020-12-02T17:57:32.698832", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 1.59924, "end_time": "2020-12-02T17:57:34.337888", "exception": false, "start_time": "2020-12-02T17:57:32.738648", "status": "completed"} tags=[]
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

# %% [markdown] papermill={"duration": 0.014267, "end_time": "2020-12-02T17:57:34.368041", "exception": false, "start_time": "2020-12-02T17:57:34.353774", "status": "completed"} tags=[]
# # Global settings

# %% papermill={"duration": 0.030894, "end_time": "2020-12-02T17:57:34.413118", "exception": false, "start_time": "2020-12-02T17:57:34.382224", "status": "completed"} tags=[]
CLUSTERING_ATTRIBUTES_TO_SAVE = ['n_clusters']

# %% [markdown] papermill={"duration": 0.014372, "end_time": "2020-12-02T17:57:34.443001", "exception": false, "start_time": "2020-12-02T17:57:34.428629", "status": "completed"} tags=[]
# # Data version: pca

# %% [markdown] papermill={"duration": 0.014142, "end_time": "2020-12-02T17:57:34.471325", "exception": false, "start_time": "2020-12-02T17:57:34.457183", "status": "completed"} tags=[]
# ## Settings

# %% papermill={"duration": 0.028426, "end_time": "2020-12-02T17:57:34.513962", "exception": false, "start_time": "2020-12-02T17:57:34.485536", "status": "completed"} tags=[]
INPUT_SUBSET = 'pca'

# %% papermill={"duration": 0.02825, "end_time": "2020-12-02T17:57:34.557155", "exception": false, "start_time": "2020-12-02T17:57:34.528905", "status": "completed"} tags=[]
INPUT_STEM = 'z_score_std-projection-smultixcan-efo_partial-mashr-zscores'

# %% papermill={"duration": 0.028488, "end_time": "2020-12-02T17:57:34.600748", "exception": false, "start_time": "2020-12-02T17:57:34.572260", "status": "completed"} tags=[]
# parameters of the dimentionality reduction steps
DR_OPTIONS = {
    'n_components': 50,
    'svd_solver': 'full',
    'random_state': 0,
}

# %% papermill={"duration": 0.031104, "end_time": "2020-12-02T17:57:34.646862", "exception": false, "start_time": "2020-12-02T17:57:34.615758", "status": "completed"} tags=[]
input_filepath = Path(
    conf.RESULTS["DATA_TRANSFORMATIONS_DIR"],
    INPUT_SUBSET,
    generate_result_set_name(
        DR_OPTIONS,
        prefix=f'{INPUT_SUBSET}-{INPUT_STEM}-',
        suffix='.pkl'
    )
).resolve()
display(input_filepath)

assert input_filepath.exists(), 'Input file does not exist'

input_filepath_stem = input_filepath.stem
display(input_filepath_stem)

# %% papermill={"duration": 0.030503, "end_time": "2020-12-02T17:57:34.692873", "exception": false, "start_time": "2020-12-02T17:57:34.662370", "status": "completed"} tags=[]
# output dir for this notebook
RESULTS_DIR = Path(
    conf.RESULTS["CLUSTERING_RUNS_DIR"],
    f'{INPUT_SUBSET}-{INPUT_STEM}',
).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %% [markdown] papermill={"duration": 0.015814, "end_time": "2020-12-02T17:57:34.724857", "exception": false, "start_time": "2020-12-02T17:57:34.709043", "status": "completed"} tags=[]
# ## Load input file

# %% papermill={"duration": 0.030964, "end_time": "2020-12-02T17:57:34.770974", "exception": false, "start_time": "2020-12-02T17:57:34.740010", "status": "completed"} tags=[]
data = pd.read_pickle(input_filepath)

# %% papermill={"duration": 0.030704, "end_time": "2020-12-02T17:57:34.817931", "exception": false, "start_time": "2020-12-02T17:57:34.787227", "status": "completed"} tags=[]
data.shape

# %% papermill={"duration": 0.043547, "end_time": "2020-12-02T17:57:34.877658", "exception": false, "start_time": "2020-12-02T17:57:34.834111", "status": "completed"} tags=[]
data.head()

# %% papermill={"duration": 0.030763, "end_time": "2020-12-02T17:57:34.926002", "exception": false, "start_time": "2020-12-02T17:57:34.895239", "status": "completed"} tags=[]
assert not data.isna().any().any()

# %% [markdown] papermill={"duration": 0.016169, "end_time": "2020-12-02T17:57:34.958817", "exception": false, "start_time": "2020-12-02T17:57:34.942648", "status": "completed"} tags=[]
# ## Tests different k values (k-NN)

# %% papermill={"duration": 0.030639, "end_time": "2020-12-02T17:57:35.005560", "exception": false, "start_time": "2020-12-02T17:57:34.974921", "status": "completed"} tags=[]
k_values = np.arange(2, 100+1, 1)
k_values_to_explore = (2, 5, 10, 15, 20, 30, 40, 50, 75, 100)

# %% papermill={"duration": 5.091421, "end_time": "2020-12-02T17:57:40.113877", "exception": false, "start_time": "2020-12-02T17:57:35.022456", "status": "completed"} tags=[]
results = {}

for k in k_values_to_explore:
    nbrs = NearestNeighbors(n_neighbors=k, n_jobs=N_JOBS).fit(data)
    distances, indices = nbrs.kneighbors(data)
    results[k] = (distances, indices)

# %% papermill={"duration": 0.030808, "end_time": "2020-12-02T17:57:40.161481", "exception": false, "start_time": "2020-12-02T17:57:40.130673", "status": "completed"} tags=[]
min_max_range = (15, 80)

eps_range_per_k = {k: min_max_range for k in k_values}
eps_range_per_k_to_explore = {k: min_max_range for k in k_values_to_explore}

# %% papermill={"duration": 0.927149, "end_time": "2020-12-02T17:57:41.105329", "exception": false, "start_time": "2020-12-02T17:57:40.178180", "status": "completed"} tags=[]
for k, (distances, indices) in results.items():
    d = distances[:,1:].mean(axis=1)
    d = np.sort(d)
    
    fig, ax = plt.subplots()
    plt.plot(d)
    
    r = eps_range_per_k_to_explore[k]
    plt.hlines(r[0], 0, data.shape[0], color='red')
    plt.hlines(r[1], 0, data.shape[0], color='red')
    
    plt.xlim((3000, data.shape[0]))
    plt.title(f'k={k}')
    display(fig)
    
    plt.close(fig)

# %% [markdown] papermill={"duration": 0.020024, "end_time": "2020-12-02T17:57:41.146134", "exception": false, "start_time": "2020-12-02T17:57:41.126110", "status": "completed"} tags=[]
# ## Clustering

# %% [markdown] papermill={"duration": 0.020223, "end_time": "2020-12-02T17:57:41.186598", "exception": false, "start_time": "2020-12-02T17:57:41.166375", "status": "completed"} tags=[]
# ### Generate clusterers

# %% papermill={"duration": 0.039767, "end_time": "2020-12-02T17:57:41.246564", "exception": false, "start_time": "2020-12-02T17:57:41.206797", "status": "completed"} tags=[]
CLUSTERING_OPTIONS = {}

# K_RANGE is the min_samples parameter in DBSCAN (sklearn)
CLUSTERING_OPTIONS['K_RANGE'] = k_values
CLUSTERING_OPTIONS['EPS_RANGE_PER_K'] = eps_range_per_k
CLUSTERING_OPTIONS['EPS_STEP'] = 33
CLUSTERING_OPTIONS['METRIC'] = 'euclidean'

display(CLUSTERING_OPTIONS)

# %% papermill={"duration": 0.047035, "end_time": "2020-12-02T17:57:41.315581", "exception": false, "start_time": "2020-12-02T17:57:41.268546", "status": "completed"} tags=[]
CLUSTERERS = {}

idx = 0

for k in CLUSTERING_OPTIONS['K_RANGE']:
    eps_range = CLUSTERING_OPTIONS['EPS_RANGE_PER_K'][k]
    eps_values = np.linspace(
        eps_range[0],
        eps_range[1],
        CLUSTERING_OPTIONS['EPS_STEP']
    )
    
    for eps in eps_values:
        clus = DBSCAN(min_samples=k, eps=eps, metric='precomputed', n_jobs=N_JOBS)
        
        method_name = type(clus).__name__
        CLUSTERERS[f'{method_name} #{idx}'] = clus
        
        idx = idx + 1

# %% papermill={"duration": 0.035584, "end_time": "2020-12-02T17:57:41.373506", "exception": false, "start_time": "2020-12-02T17:57:41.337922", "status": "completed"} tags=[]
display(len(CLUSTERERS))

# %% papermill={"duration": 0.039081, "end_time": "2020-12-02T17:57:41.433854", "exception": false, "start_time": "2020-12-02T17:57:41.394773", "status": "completed"} tags=[]
_iter = iter(CLUSTERERS.items())
display(next(_iter))
display(next(_iter))

# %% papermill={"duration": 0.037169, "end_time": "2020-12-02T17:57:41.493028", "exception": false, "start_time": "2020-12-02T17:57:41.455859", "status": "completed"} tags=[]
clustering_method_name = method_name
display(clustering_method_name)

# %% [markdown] papermill={"duration": 0.021613, "end_time": "2020-12-02T17:57:41.537364", "exception": false, "start_time": "2020-12-02T17:57:41.515751", "status": "completed"} tags=[]
# ### Generate ensemble

# %% papermill={"duration": 0.194685, "end_time": "2020-12-02T17:57:41.753539", "exception": false, "start_time": "2020-12-02T17:57:41.558854", "status": "completed"} tags=[]
data_dist = pairwise_distances(data, metric=CLUSTERING_OPTIONS['METRIC'])

# %% papermill={"duration": 0.036712, "end_time": "2020-12-02T17:57:41.812368", "exception": false, "start_time": "2020-12-02T17:57:41.775656", "status": "completed"} tags=[]
data_dist.shape

# %% papermill={"duration": 0.623866, "end_time": "2020-12-02T17:57:42.459238", "exception": false, "start_time": "2020-12-02T17:57:41.835372", "status": "completed"} tags=[]
pd.Series(data_dist.flatten()).describe().apply(str)

# %% papermill={"duration": 640.397744, "end_time": "2020-12-02T18:08:22.881402", "exception": false, "start_time": "2020-12-02T17:57:42.483658", "status": "completed"} tags=[]
ensemble = generate_ensemble(
    data_dist,
    CLUSTERERS,
    attributes=CLUSTERING_ATTRIBUTES_TO_SAVE,
)

# %% papermill={"duration": 0.545944, "end_time": "2020-12-02T18:08:23.965446", "exception": false, "start_time": "2020-12-02T18:08:23.419502", "status": "completed"} tags=[]
# the number should be close to 295 (the number of partitions generated by k-means/spectral clustering)
ensemble.shape

# %% papermill={"duration": 0.554332, "end_time": "2020-12-02T18:08:25.055871", "exception": false, "start_time": "2020-12-02T18:08:24.501539", "status": "completed"} tags=[]
ensemble.head()

# %% papermill={"duration": 0.551893, "end_time": "2020-12-02T18:08:26.163407", "exception": false, "start_time": "2020-12-02T18:08:25.611514", "status": "completed"} tags=[]
ensemble['n_clusters'].value_counts().head()

# %% papermill={"duration": 0.551258, "end_time": "2020-12-02T18:08:27.246288", "exception": false, "start_time": "2020-12-02T18:08:26.695030", "status": "completed"} tags=[]
ensemble_stats = ensemble['n_clusters'].describe()
display(ensemble_stats)

# %% [markdown] papermill={"duration": 0.537921, "end_time": "2020-12-02T18:08:28.320680", "exception": false, "start_time": "2020-12-02T18:08:27.782759", "status": "completed"} tags=[]
# ### Testing

# %% papermill={"duration": 0.577933, "end_time": "2020-12-02T18:08:29.437542", "exception": false, "start_time": "2020-12-02T18:08:28.859609", "status": "completed"} tags=[]
assert ensemble_stats['min'] > 1

# %% papermill={"duration": 0.554278, "end_time": "2020-12-02T18:08:30.526282", "exception": false, "start_time": "2020-12-02T18:08:29.972004", "status": "completed"} tags=[]
assert not ensemble['n_clusters'].isna().any()

# %% papermill={"duration": 0.554341, "end_time": "2020-12-02T18:08:31.614158", "exception": false, "start_time": "2020-12-02T18:08:31.059817", "status": "completed"} tags=[]
# assert ensemble.shape[0] == len(CLUSTERERS)

# %% papermill={"duration": 0.563803, "end_time": "2020-12-02T18:08:32.719255", "exception": false, "start_time": "2020-12-02T18:08:32.155452", "status": "completed"} tags=[]
# all partitions have the right size
assert np.all([
    part['partition'].shape[0] == data.shape[0]
    for idx, part in ensemble.iterrows()
])

# %% papermill={"duration": 0.566354, "end_time": "2020-12-02T18:08:33.853665", "exception": false, "start_time": "2020-12-02T18:08:33.287311", "status": "completed"} tags=[]
# no partition has negative clusters (noisy points)
assert not np.any([
    (part['partition'] < 0).any()
    for idx, part in ensemble.iterrows()
])

# %% [markdown] papermill={"duration": 0.534006, "end_time": "2020-12-02T18:08:34.924297", "exception": false, "start_time": "2020-12-02T18:08:34.390291", "status": "completed"} tags=[]
# ### Save

# %% papermill={"duration": 0.549171, "end_time": "2020-12-02T18:08:36.010900", "exception": false, "start_time": "2020-12-02T18:08:35.461729", "status": "completed"} tags=[]
output_filename = Path(
    RESULTS_DIR,
    generate_result_set_name(
        {},
#         CLUSTERING_OPTIONS,
        prefix=f'{clustering_method_name}',
        suffix='.pkl',
    )
).resolve()
display(output_filename)

# %% papermill={"duration": 0.586163, "end_time": "2020-12-02T18:08:37.138678", "exception": false, "start_time": "2020-12-02T18:08:36.552515", "status": "completed"} tags=[]
ensemble.to_pickle(output_filename)

# %% papermill={"duration": 0.540578, "end_time": "2020-12-02T18:08:38.221636", "exception": false, "start_time": "2020-12-02T18:08:37.681058", "status": "completed"} tags=[]
