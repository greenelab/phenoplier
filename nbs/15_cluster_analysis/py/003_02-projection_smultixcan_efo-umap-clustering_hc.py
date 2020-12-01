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

# %% [markdown] papermill={"duration": 0.019267, "end_time": "2020-12-03T02:17:52.366294", "exception": false, "start_time": "2020-12-03T02:17:52.347027", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.015019, "end_time": "2020-12-03T02:17:52.396750", "exception": false, "start_time": "2020-12-03T02:17:52.381731", "status": "completed"} tags=[]
# Runs hierarchical clustering on the umap version of the data.

# %% [markdown] papermill={"duration": 0.014975, "end_time": "2020-12-03T02:17:52.426692", "exception": false, "start_time": "2020-12-03T02:17:52.411717", "status": "completed"} tags=[]
# # Environment variables

# %% papermill={"duration": 0.031777, "end_time": "2020-12-03T02:17:52.474863", "exception": false, "start_time": "2020-12-03T02:17:52.443086", "status": "completed"} tags=[]
from IPython.display import display

import conf

N_JOBS = conf.GENERAL['N_JOBS']
display(N_JOBS)

# %% papermill={"duration": 0.022543, "end_time": "2020-12-03T02:17:52.513497", "exception": false, "start_time": "2020-12-03T02:17:52.490954", "status": "completed"} tags=[]
# %env MKL_NUM_THREADS=$N_JOBS
# %env OPEN_BLAS_NUM_THREADS=$N_JOBS
# %env NUMEXPR_NUM_THREADS=$N_JOBS
# %env OMP_NUM_THREADS=$N_JOBS

# %% [markdown] papermill={"duration": 0.015998, "end_time": "2020-12-03T02:17:52.545959", "exception": false, "start_time": "2020-12-03T02:17:52.529961", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.025917, "end_time": "2020-12-03T02:17:52.587655", "exception": false, "start_time": "2020-12-03T02:17:52.561738", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 1.579861, "end_time": "2020-12-03T02:17:54.183969", "exception": false, "start_time": "2020-12-03T02:17:52.604108", "status": "completed"} tags=[]
from pathlib import Path

import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns

from utils import generate_result_set_name

# %% [markdown] papermill={"duration": 0.016019, "end_time": "2020-12-03T02:17:54.218240", "exception": false, "start_time": "2020-12-03T02:17:54.202221", "status": "completed"} tags=[]
# # Settings

# %% [markdown] papermill={"duration": 0.015529, "end_time": "2020-12-03T02:17:54.249333", "exception": false, "start_time": "2020-12-03T02:17:54.233804", "status": "completed"} tags=[]
# ## Input data

# %% papermill={"duration": 0.029912, "end_time": "2020-12-03T02:17:54.294843", "exception": false, "start_time": "2020-12-03T02:17:54.264931", "status": "completed"} tags=[]
INPUT_SUBSET = 'umap'

# %% papermill={"duration": 0.029359, "end_time": "2020-12-03T02:17:54.340076", "exception": false, "start_time": "2020-12-03T02:17:54.310717", "status": "completed"} tags=[]
INPUT_STEM = 'z_score_std-projection-smultixcan-efo_partial-mashr-zscores'

# %% papermill={"duration": 0.029488, "end_time": "2020-12-03T02:17:54.385580", "exception": false, "start_time": "2020-12-03T02:17:54.356092", "status": "completed"} tags=[]
# parameters of the dimentionality reduction steps
DR_OPTIONS = {
    'n_components': 50,
    'metric': 'euclidean',
    'n_neighbors': 15,
    'random_state': 0,
}

# %% papermill={"duration": 0.031802, "end_time": "2020-12-03T02:17:54.433398", "exception": false, "start_time": "2020-12-03T02:17:54.401596", "status": "completed"} tags=[]
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

# %% [markdown] papermill={"duration": 0.016842, "end_time": "2020-12-03T02:17:54.467937", "exception": false, "start_time": "2020-12-03T02:17:54.451095", "status": "completed"} tags=[]
# ## Clustering

# %% papermill={"duration": 0.036098, "end_time": "2020-12-03T02:17:54.519974", "exception": false, "start_time": "2020-12-03T02:17:54.483876", "status": "completed"} tags=[]
from sklearn.cluster import AgglomerativeClustering

# %% papermill={"duration": 0.030281, "end_time": "2020-12-03T02:17:54.566961", "exception": false, "start_time": "2020-12-03T02:17:54.536680", "status": "completed"} tags=[]
CLUSTERING_ATTRIBUTES_TO_SAVE = ['n_clusters']

# %% papermill={"duration": 0.031305, "end_time": "2020-12-03T02:17:54.614481", "exception": false, "start_time": "2020-12-03T02:17:54.583176", "status": "completed"} tags=[]
CLUSTERING_OPTIONS = {}

CLUSTERING_OPTIONS['K_MIN'] = 2
CLUSTERING_OPTIONS['K_MAX'] = 75 # sqrt(3749) + some more to get closer to 295
CLUSTERING_OPTIONS['LINKAGE'] = {'ward', 'complete', 'average', 'single'}
CLUSTERING_OPTIONS['AFFINITY'] = 'euclidean'

display(CLUSTERING_OPTIONS)

# %% papermill={"duration": 0.032621, "end_time": "2020-12-03T02:17:54.664030", "exception": false, "start_time": "2020-12-03T02:17:54.631409", "status": "completed"} tags=[]
CLUSTERERS = {}

idx = 0

for k in range(CLUSTERING_OPTIONS['K_MIN'], CLUSTERING_OPTIONS['K_MAX']+1):
    for linkage in CLUSTERING_OPTIONS['LINKAGE']:
        if linkage == 'ward':
            affinity = 'euclidean'
        else:
            affinity = 'precomputed'
        
        clus = AgglomerativeClustering(
                n_clusters=k,
                affinity=affinity,
                linkage=linkage,
            )
        
        method_name = type(clus).__name__
        CLUSTERERS[f'{method_name} #{idx}'] = clus
        
        idx = idx + 1

# %% papermill={"duration": 0.03075, "end_time": "2020-12-03T02:17:54.711697", "exception": false, "start_time": "2020-12-03T02:17:54.680947", "status": "completed"} tags=[]
display(len(CLUSTERERS))

# %% papermill={"duration": 0.033917, "end_time": "2020-12-03T02:17:54.762867", "exception": false, "start_time": "2020-12-03T02:17:54.728950", "status": "completed"} tags=[]
_iter = iter(CLUSTERERS.items())
display(next(_iter))
display(next(_iter))

# %% papermill={"duration": 0.03236, "end_time": "2020-12-03T02:17:54.812892", "exception": false, "start_time": "2020-12-03T02:17:54.780532", "status": "completed"} tags=[]
clustering_method_name = method_name
display(clustering_method_name)

# %% [markdown] papermill={"duration": 0.017027, "end_time": "2020-12-03T02:17:54.847431", "exception": false, "start_time": "2020-12-03T02:17:54.830404", "status": "completed"} tags=[]
# ## Output directory

# %% papermill={"duration": 0.032233, "end_time": "2020-12-03T02:17:54.896620", "exception": false, "start_time": "2020-12-03T02:17:54.864387", "status": "completed"} tags=[]
# output dir for this notebook
RESULTS_DIR = Path(
    conf.RESULTS["CLUSTERING_RUNS_DIR"],
    f'{INPUT_SUBSET}-{INPUT_STEM}',
).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %% [markdown] papermill={"duration": 0.01708, "end_time": "2020-12-03T02:17:54.931448", "exception": false, "start_time": "2020-12-03T02:17:54.914368", "status": "completed"} tags=[]
# # Load input file

# %% papermill={"duration": 0.033162, "end_time": "2020-12-03T02:17:54.981941", "exception": false, "start_time": "2020-12-03T02:17:54.948779", "status": "completed"} tags=[]
data = pd.read_pickle(input_filepath)

# %% papermill={"duration": 0.033061, "end_time": "2020-12-03T02:17:55.033390", "exception": false, "start_time": "2020-12-03T02:17:55.000329", "status": "completed"} tags=[]
data.shape

# %% papermill={"duration": 0.045205, "end_time": "2020-12-03T02:17:55.097020", "exception": false, "start_time": "2020-12-03T02:17:55.051815", "status": "completed"} tags=[]
data.head()

# %% papermill={"duration": 0.033144, "end_time": "2020-12-03T02:17:55.148383", "exception": false, "start_time": "2020-12-03T02:17:55.115239", "status": "completed"} tags=[]
assert not data.isna().any().any()

# %% [markdown] papermill={"duration": 0.018101, "end_time": "2020-12-03T02:17:55.185403", "exception": false, "start_time": "2020-12-03T02:17:55.167302", "status": "completed"} tags=[]
# # Clustering

# %% [markdown] papermill={"duration": 0.017588, "end_time": "2020-12-03T02:17:55.220857", "exception": false, "start_time": "2020-12-03T02:17:55.203269", "status": "completed"} tags=[]
# ## Generate ensemble

# %% papermill={"duration": 0.034598, "end_time": "2020-12-03T02:17:55.273281", "exception": false, "start_time": "2020-12-03T02:17:55.238683", "status": "completed"} tags=[]
from sklearn.metrics import pairwise_distances
from clustering.ensemble import generate_ensemble

# %% papermill={"duration": 0.176116, "end_time": "2020-12-03T02:17:55.467606", "exception": false, "start_time": "2020-12-03T02:17:55.291490", "status": "completed"} tags=[]
data_dist = pairwise_distances(data, metric=CLUSTERING_OPTIONS['AFFINITY'])

# %% papermill={"duration": 0.033045, "end_time": "2020-12-03T02:17:55.519393", "exception": false, "start_time": "2020-12-03T02:17:55.486348", "status": "completed"} tags=[]
data_dist.shape

# %% papermill={"duration": 0.467322, "end_time": "2020-12-03T02:17:56.005399", "exception": false, "start_time": "2020-12-03T02:17:55.538077", "status": "completed"} tags=[]
pd.Series(data_dist.flatten()).describe().apply(str)

# %% papermill={"duration": 1648.815495, "end_time": "2020-12-03T02:45:24.839171", "exception": false, "start_time": "2020-12-03T02:17:56.023676", "status": "completed"} tags=[]
ensemble = generate_ensemble(
    data_dist,
    CLUSTERERS,
    attributes=CLUSTERING_ATTRIBUTES_TO_SAVE,
    affinity_matrix=data_dist,
)

# %% papermill={"duration": 0.080488, "end_time": "2020-12-03T02:45:24.984941", "exception": false, "start_time": "2020-12-03T02:45:24.904453", "status": "completed"} tags=[]
# the number should be close to 295 (the number of partitions generated by k-means/spectral clustering)
ensemble.shape

# %% papermill={"duration": 0.087998, "end_time": "2020-12-03T02:45:25.138987", "exception": false, "start_time": "2020-12-03T02:45:25.050989", "status": "completed"} tags=[]
ensemble.head()

# %% papermill={"duration": 0.082424, "end_time": "2020-12-03T02:45:25.287808", "exception": false, "start_time": "2020-12-03T02:45:25.205384", "status": "completed"} tags=[]
ensemble['n_clusters'].value_counts().head()

# %% papermill={"duration": 0.083507, "end_time": "2020-12-03T02:45:25.438790", "exception": false, "start_time": "2020-12-03T02:45:25.355283", "status": "completed"} tags=[]
ensemble_stats = ensemble['n_clusters'].describe()
display(ensemble_stats)

# %% [markdown] papermill={"duration": 0.065573, "end_time": "2020-12-03T02:45:25.571629", "exception": false, "start_time": "2020-12-03T02:45:25.506056", "status": "completed"} tags=[]
# ### Testing

# %% papermill={"duration": 0.080149, "end_time": "2020-12-03T02:45:25.717547", "exception": false, "start_time": "2020-12-03T02:45:25.637398", "status": "completed"} tags=[]
assert ensemble_stats['min'] > 1

# %% papermill={"duration": 0.081478, "end_time": "2020-12-03T02:45:25.865672", "exception": false, "start_time": "2020-12-03T02:45:25.784194", "status": "completed"} tags=[]
assert not ensemble['n_clusters'].isna().any()

# %% papermill={"duration": 0.079979, "end_time": "2020-12-03T02:45:26.012481", "exception": false, "start_time": "2020-12-03T02:45:25.932502", "status": "completed"} tags=[]
assert ensemble.shape[0] == len(CLUSTERERS)

# %% papermill={"duration": 0.098273, "end_time": "2020-12-03T02:45:26.177409", "exception": false, "start_time": "2020-12-03T02:45:26.079136", "status": "completed"} tags=[]
# all partitions have the right size
assert np.all([
    part['partition'].shape[0] == data.shape[0]
    for idx, part in ensemble.iterrows()
])

# %% papermill={"duration": 0.101418, "end_time": "2020-12-03T02:45:26.345005", "exception": false, "start_time": "2020-12-03T02:45:26.243587", "status": "completed"} tags=[]
# no partition has negative clusters (noisy points)
assert not np.any([
    (part['partition'] < 0).any()
    for idx, part in ensemble.iterrows()
])

# %% [markdown] papermill={"duration": 0.065269, "end_time": "2020-12-03T02:45:26.477137", "exception": false, "start_time": "2020-12-03T02:45:26.411868", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.08065, "end_time": "2020-12-03T02:45:26.622922", "exception": false, "start_time": "2020-12-03T02:45:26.542272", "status": "completed"} tags=[]
del CLUSTERING_OPTIONS['LINKAGE']

output_filename = Path(
    RESULTS_DIR,
    generate_result_set_name(
        CLUSTERING_OPTIONS,
        prefix=f'{clustering_method_name}-',
        suffix='.pkl',
    )
).resolve()
display(output_filename)

# %% papermill={"duration": 0.111116, "end_time": "2020-12-03T02:45:26.801490", "exception": false, "start_time": "2020-12-03T02:45:26.690374", "status": "completed"} tags=[]
ensemble.to_pickle(output_filename)

# %% papermill={"duration": 0.06702, "end_time": "2020-12-03T02:45:26.963657", "exception": false, "start_time": "2020-12-03T02:45:26.896637", "status": "completed"} tags=[]
