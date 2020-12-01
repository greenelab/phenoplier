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

# %% [markdown] papermill={"duration": 0.048626, "end_time": "2020-12-03T01:50:03.921560", "exception": false, "start_time": "2020-12-03T01:50:03.872934", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.014679, "end_time": "2020-12-03T01:50:03.953555", "exception": false, "start_time": "2020-12-03T01:50:03.938876", "status": "completed"} tags=[]
# Runs hierarchical clustering on the pca version of the data.

# %% [markdown] papermill={"duration": 0.014887, "end_time": "2020-12-03T01:50:03.983355", "exception": false, "start_time": "2020-12-03T01:50:03.968468", "status": "completed"} tags=[]
# # Environment variables

# %% papermill={"duration": 0.029956, "end_time": "2020-12-03T01:50:04.028364", "exception": false, "start_time": "2020-12-03T01:50:03.998408", "status": "completed"} tags=[]
from IPython.display import display

import conf

N_JOBS = conf.GENERAL['N_JOBS']
display(N_JOBS)

# %% papermill={"duration": 0.022867, "end_time": "2020-12-03T01:50:04.067468", "exception": false, "start_time": "2020-12-03T01:50:04.044601", "status": "completed"} tags=[]
# %env MKL_NUM_THREADS=$N_JOBS
# %env OPEN_BLAS_NUM_THREADS=$N_JOBS
# %env NUMEXPR_NUM_THREADS=$N_JOBS
# %env OMP_NUM_THREADS=$N_JOBS

# %% [markdown] papermill={"duration": 0.015796, "end_time": "2020-12-03T01:50:04.099762", "exception": false, "start_time": "2020-12-03T01:50:04.083966", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.025873, "end_time": "2020-12-03T01:50:04.141491", "exception": false, "start_time": "2020-12-03T01:50:04.115618", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 1.614215, "end_time": "2020-12-03T01:50:05.771795", "exception": false, "start_time": "2020-12-03T01:50:04.157580", "status": "completed"} tags=[]
from pathlib import Path

import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns

from utils import generate_result_set_name

# %% [markdown] papermill={"duration": 0.016, "end_time": "2020-12-03T01:50:05.805223", "exception": false, "start_time": "2020-12-03T01:50:05.789223", "status": "completed"} tags=[]
# # Settings

# %% [markdown] papermill={"duration": 0.015496, "end_time": "2020-12-03T01:50:05.837659", "exception": false, "start_time": "2020-12-03T01:50:05.822163", "status": "completed"} tags=[]
# ## Input data

# %% papermill={"duration": 0.029533, "end_time": "2020-12-03T01:50:05.883008", "exception": false, "start_time": "2020-12-03T01:50:05.853475", "status": "completed"} tags=[]
INPUT_SUBSET = 'pca'

# %% papermill={"duration": 0.029571, "end_time": "2020-12-03T01:50:05.928466", "exception": false, "start_time": "2020-12-03T01:50:05.898895", "status": "completed"} tags=[]
INPUT_STEM = 'z_score_std-projection-smultixcan-efo_partial-mashr-zscores'

# %% papermill={"duration": 0.030005, "end_time": "2020-12-03T01:50:05.975123", "exception": false, "start_time": "2020-12-03T01:50:05.945118", "status": "completed"} tags=[]
# parameters of the dimentionality reduction steps
DR_OPTIONS = {
    'n_components': 50,
    'svd_solver': 'full',
    'random_state': 0,
}

# %% papermill={"duration": 0.032144, "end_time": "2020-12-03T01:50:06.023180", "exception": false, "start_time": "2020-12-03T01:50:05.991036", "status": "completed"} tags=[]
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

# %% [markdown] papermill={"duration": 0.015974, "end_time": "2020-12-03T01:50:06.055225", "exception": false, "start_time": "2020-12-03T01:50:06.039251", "status": "completed"} tags=[]
# ## Clustering

# %% papermill={"duration": 0.036409, "end_time": "2020-12-03T01:50:06.107776", "exception": false, "start_time": "2020-12-03T01:50:06.071367", "status": "completed"} tags=[]
from sklearn.cluster import AgglomerativeClustering

# %% papermill={"duration": 0.03114, "end_time": "2020-12-03T01:50:06.156070", "exception": false, "start_time": "2020-12-03T01:50:06.124930", "status": "completed"} tags=[]
CLUSTERING_ATTRIBUTES_TO_SAVE = ['n_clusters']

# %% papermill={"duration": 0.031859, "end_time": "2020-12-03T01:50:06.204567", "exception": false, "start_time": "2020-12-03T01:50:06.172708", "status": "completed"} tags=[]
CLUSTERING_OPTIONS = {}

CLUSTERING_OPTIONS['K_MIN'] = 2
CLUSTERING_OPTIONS['K_MAX'] = 75 # sqrt(3749) + some more to get closer to 295
CLUSTERING_OPTIONS['LINKAGE'] = {'ward', 'complete', 'average', 'single'}
CLUSTERING_OPTIONS['AFFINITY'] = 'euclidean'

display(CLUSTERING_OPTIONS)

# %% papermill={"duration": 0.031741, "end_time": "2020-12-03T01:50:06.253411", "exception": false, "start_time": "2020-12-03T01:50:06.221670", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.032565, "end_time": "2020-12-03T01:50:06.303387", "exception": false, "start_time": "2020-12-03T01:50:06.270822", "status": "completed"} tags=[]
display(len(CLUSTERERS))

# %% papermill={"duration": 0.033443, "end_time": "2020-12-03T01:50:06.354284", "exception": false, "start_time": "2020-12-03T01:50:06.320841", "status": "completed"} tags=[]
_iter = iter(CLUSTERERS.items())
display(next(_iter))
display(next(_iter))

# %% papermill={"duration": 0.032183, "end_time": "2020-12-03T01:50:06.403860", "exception": false, "start_time": "2020-12-03T01:50:06.371677", "status": "completed"} tags=[]
clustering_method_name = method_name
display(clustering_method_name)

# %% [markdown] papermill={"duration": 0.016904, "end_time": "2020-12-03T01:50:06.438446", "exception": false, "start_time": "2020-12-03T01:50:06.421542", "status": "completed"} tags=[]
# ## Output directory

# %% papermill={"duration": 0.03247, "end_time": "2020-12-03T01:50:06.487753", "exception": false, "start_time": "2020-12-03T01:50:06.455283", "status": "completed"} tags=[]
# output dir for this notebook
RESULTS_DIR = Path(
    conf.RESULTS["CLUSTERING_RUNS_DIR"],
    f'{INPUT_SUBSET}-{INPUT_STEM}',
).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %% [markdown] papermill={"duration": 0.01732, "end_time": "2020-12-03T01:50:06.523050", "exception": false, "start_time": "2020-12-03T01:50:06.505730", "status": "completed"} tags=[]
# # Load input file

# %% papermill={"duration": 0.032962, "end_time": "2020-12-03T01:50:06.572919", "exception": false, "start_time": "2020-12-03T01:50:06.539957", "status": "completed"} tags=[]
data = pd.read_pickle(input_filepath)

# %% papermill={"duration": 0.032924, "end_time": "2020-12-03T01:50:06.623965", "exception": false, "start_time": "2020-12-03T01:50:06.591041", "status": "completed"} tags=[]
data.shape

# %% papermill={"duration": 0.044754, "end_time": "2020-12-03T01:50:06.686317", "exception": false, "start_time": "2020-12-03T01:50:06.641563", "status": "completed"} tags=[]
data.head()

# %% papermill={"duration": 0.033244, "end_time": "2020-12-03T01:50:06.742939", "exception": false, "start_time": "2020-12-03T01:50:06.709695", "status": "completed"} tags=[]
assert not data.isna().any().any()

# %% [markdown] papermill={"duration": 0.017522, "end_time": "2020-12-03T01:50:06.778971", "exception": false, "start_time": "2020-12-03T01:50:06.761449", "status": "completed"} tags=[]
# # Clustering

# %% [markdown] papermill={"duration": 0.01964, "end_time": "2020-12-03T01:50:06.816504", "exception": false, "start_time": "2020-12-03T01:50:06.796864", "status": "completed"} tags=[]
# ## Generate ensemble

# %% papermill={"duration": 0.035103, "end_time": "2020-12-03T01:50:06.870029", "exception": false, "start_time": "2020-12-03T01:50:06.834926", "status": "completed"} tags=[]
from sklearn.metrics import pairwise_distances
from clustering.ensemble import generate_ensemble

# %% papermill={"duration": 0.18922, "end_time": "2020-12-03T01:50:07.077854", "exception": false, "start_time": "2020-12-03T01:50:06.888634", "status": "completed"} tags=[]
data_dist = pairwise_distances(data, metric=CLUSTERING_OPTIONS['AFFINITY'])

# %% papermill={"duration": 0.033065, "end_time": "2020-12-03T01:50:07.128396", "exception": false, "start_time": "2020-12-03T01:50:07.095331", "status": "completed"} tags=[]
data_dist.shape

# %% papermill={"duration": 0.619414, "end_time": "2020-12-03T01:50:07.766714", "exception": false, "start_time": "2020-12-03T01:50:07.147300", "status": "completed"} tags=[]
pd.Series(data_dist.flatten()).describe().apply(str)

# %% papermill={"duration": 1660.582721, "end_time": "2020-12-03T02:17:48.368219", "exception": false, "start_time": "2020-12-03T01:50:07.785498", "status": "completed"} tags=[]
ensemble = generate_ensemble(
    data_dist,
    CLUSTERERS,
    attributes=CLUSTERING_ATTRIBUTES_TO_SAVE,
    affinity_matrix=data_dist,
)

# %% papermill={"duration": 0.079568, "end_time": "2020-12-03T02:17:48.513410", "exception": false, "start_time": "2020-12-03T02:17:48.433842", "status": "completed"} tags=[]
# the number should be close to 295 (the number of partitions generated by k-means/spectral clustering)
ensemble.shape

# %% papermill={"duration": 0.086706, "end_time": "2020-12-03T02:17:48.665744", "exception": false, "start_time": "2020-12-03T02:17:48.579038", "status": "completed"} tags=[]
ensemble.head()

# %% papermill={"duration": 0.082031, "end_time": "2020-12-03T02:17:48.812975", "exception": false, "start_time": "2020-12-03T02:17:48.730944", "status": "completed"} tags=[]
ensemble['n_clusters'].value_counts().head()

# %% papermill={"duration": 0.08316, "end_time": "2020-12-03T02:17:48.961576", "exception": false, "start_time": "2020-12-03T02:17:48.878416", "status": "completed"} tags=[]
ensemble_stats = ensemble['n_clusters'].describe()
display(ensemble_stats)

# %% [markdown] papermill={"duration": 0.066517, "end_time": "2020-12-03T02:17:49.094130", "exception": false, "start_time": "2020-12-03T02:17:49.027613", "status": "completed"} tags=[]
# ### Testing

# %% papermill={"duration": 0.080053, "end_time": "2020-12-03T02:17:49.240188", "exception": false, "start_time": "2020-12-03T02:17:49.160135", "status": "completed"} tags=[]
assert ensemble_stats['min'] > 1

# %% papermill={"duration": 0.0801, "end_time": "2020-12-03T02:17:49.385981", "exception": false, "start_time": "2020-12-03T02:17:49.305881", "status": "completed"} tags=[]
assert not ensemble['n_clusters'].isna().any()

# %% papermill={"duration": 0.08207, "end_time": "2020-12-03T02:17:49.535878", "exception": false, "start_time": "2020-12-03T02:17:49.453808", "status": "completed"} tags=[]
assert ensemble.shape[0] == len(CLUSTERERS)

# %% papermill={"duration": 0.099202, "end_time": "2020-12-03T02:17:49.704399", "exception": false, "start_time": "2020-12-03T02:17:49.605197", "status": "completed"} tags=[]
# all partitions have the right size
assert np.all([
    part['partition'].shape[0] == data.shape[0]
    for idx, part in ensemble.iterrows()
])

# %% papermill={"duration": 0.100275, "end_time": "2020-12-03T02:17:49.869739", "exception": false, "start_time": "2020-12-03T02:17:49.769464", "status": "completed"} tags=[]
# no partition has negative clusters (noisy points)
assert not np.any([
    (part['partition'] < 0).any()
    for idx, part in ensemble.iterrows()
])

# %% [markdown] papermill={"duration": 0.064474, "end_time": "2020-12-03T02:17:50.000811", "exception": false, "start_time": "2020-12-03T02:17:49.936337", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.082313, "end_time": "2020-12-03T02:17:50.149007", "exception": false, "start_time": "2020-12-03T02:17:50.066694", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.109519, "end_time": "2020-12-03T02:17:50.323919", "exception": false, "start_time": "2020-12-03T02:17:50.214400", "status": "completed"} tags=[]
ensemble.to_pickle(output_filename)

# %% papermill={"duration": 0.068518, "end_time": "2020-12-03T02:17:50.460504", "exception": false, "start_time": "2020-12-03T02:17:50.391986", "status": "completed"} tags=[]
