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

# %% [markdown] papermill={"duration": 0.046422, "end_time": "2020-12-04T19:11:05.972007", "exception": false, "start_time": "2020-12-04T19:11:05.925585", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.013498, "end_time": "2020-12-04T19:11:06.001425", "exception": false, "start_time": "2020-12-04T19:11:05.987927", "status": "completed"} tags=[]
# TODO

# %% [markdown] papermill={"duration": 0.01261, "end_time": "2020-12-04T19:11:06.026871", "exception": false, "start_time": "2020-12-04T19:11:06.014261", "status": "completed"} tags=[]
# # Environment variables

# %% papermill={"duration": 0.026973, "end_time": "2020-12-04T19:11:06.066494", "exception": false, "start_time": "2020-12-04T19:11:06.039521", "status": "completed"} tags=[]
from IPython.display import display

import conf

N_JOBS = conf.GENERAL['N_JOBS']
display(N_JOBS)

# %% papermill={"duration": 0.019624, "end_time": "2020-12-04T19:11:06.100027", "exception": false, "start_time": "2020-12-04T19:11:06.080403", "status": "completed"} tags=[]
# %env MKL_NUM_THREADS=$N_JOBS
# %env OPEN_BLAS_NUM_THREADS=$N_JOBS
# %env NUMEXPR_NUM_THREADS=$N_JOBS
# %env OMP_NUM_THREADS=$N_JOBS

# %% [markdown] papermill={"duration": 0.014595, "end_time": "2020-12-04T19:11:06.128906", "exception": false, "start_time": "2020-12-04T19:11:06.114311", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.024329, "end_time": "2020-12-04T19:11:06.166814", "exception": false, "start_time": "2020-12-04T19:11:06.142485", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 0.576633, "end_time": "2020-12-04T19:11:06.757106", "exception": false, "start_time": "2020-12-04T19:11:06.180473", "status": "completed"} tags=[]
from pathlib import Path

import numpy as np
import pandas as pd
# from sklearn.cluster import SpectralClustering
# import umap
import matplotlib.pyplot as plt
import seaborn as sns

from utils import generate_result_set_name

# %% [markdown] papermill={"duration": 0.016598, "end_time": "2020-12-04T19:11:06.791136", "exception": false, "start_time": "2020-12-04T19:11:06.774538", "status": "completed"} tags=[]
# # Settings

# %% papermill={"duration": 0.026353, "end_time": "2020-12-04T19:11:06.832709", "exception": false, "start_time": "2020-12-04T19:11:06.806356", "status": "completed"} tags=[]
RANDOM_GENERATOR = np.random.default_rng(12345)

# %% [markdown] papermill={"duration": 0.013677, "end_time": "2020-12-04T19:11:06.860604", "exception": false, "start_time": "2020-12-04T19:11:06.846927", "status": "completed"} tags=[]
# ## Ensemble size

# %% papermill={"duration": 0.025145, "end_time": "2020-12-04T19:11:06.899085", "exception": false, "start_time": "2020-12-04T19:11:06.873940", "status": "completed"} tags=[]
EXPECTED_ENSEMBLE_SIZE = 295

MIN_ENSEMBLE_SIZE = 290
MAX_ENSEMBLE_SIZE = 300

# %% [markdown] papermill={"duration": 0.013399, "end_time": "2020-12-04T19:11:06.926241", "exception": false, "start_time": "2020-12-04T19:11:06.912842", "status": "completed"} tags=[]
# ## Data subsets included

# %% papermill={"duration": 0.02521, "end_time": "2020-12-04T19:11:06.964786", "exception": false, "start_time": "2020-12-04T19:11:06.939576", "status": "completed"} tags=[]
# DATA_SUBSETS = [
#     'z_score_std',
#     'pca',
#     'umap'
# ]

# %% [markdown] papermill={"duration": 0.013249, "end_time": "2020-12-04T19:11:06.991555", "exception": false, "start_time": "2020-12-04T19:11:06.978306", "status": "completed"} tags=[]
# ## Algorithms included

# %% papermill={"duration": 0.025766, "end_time": "2020-12-04T19:11:07.030984", "exception": false, "start_time": "2020-12-04T19:11:07.005218", "status": "completed"} tags=[]
# ALGORITHMS = [
#     'KMeans',
#     COMPLETE
# ]

# %% [markdown] papermill={"duration": 0.01399, "end_time": "2020-12-04T19:11:07.059229", "exception": false, "start_time": "2020-12-04T19:11:07.045239", "status": "completed"} tags=[]
# ## Input data

# %% papermill={"duration": 0.025069, "end_time": "2020-12-04T19:11:07.098384", "exception": false, "start_time": "2020-12-04T19:11:07.073315", "status": "completed"} tags=[]
# DATA_SUBSET = 'z_score_std'

# INPUT_DIR = Path(
#     conf.RESULTS["DATA_TRANSFORMATIONS_DIR"],
#     DATA_SUBSET,
# ).resolve()
# display(INPUT_DIR)

# %% papermill={"duration": 0.027319, "end_time": "2020-12-04T19:11:07.139548", "exception": false, "start_time": "2020-12-04T19:11:07.112229", "status": "completed"} tags=[]
# INPUT_STEM = 'projection-smultixcan-efo_partial-mashr-zscores'

# %% papermill={"duration": 0.02651, "end_time": "2020-12-04T19:11:07.181402", "exception": false, "start_time": "2020-12-04T19:11:07.154892", "status": "completed"} tags=[]
# INPUT_FILEPATH = Path(
#     INPUT_DIR,
#     generate_result_set_name(
#         {},
#         prefix=f'{DATA_SUBSET}-{INPUT_STEM}',
#         suffix='.pkl'
#     )
# ).resolve()
# display(INPUT_FILEPATH)

# assert INPUT_FILEPATH.exists(), 'Input file does not exist'

# %% [markdown] papermill={"duration": 0.01436, "end_time": "2020-12-04T19:11:07.210655", "exception": false, "start_time": "2020-12-04T19:11:07.196295", "status": "completed"} tags=[]
# ## Consensus clustering

# %% papermill={"duration": 0.025253, "end_time": "2020-12-04T19:11:07.249621", "exception": false, "start_time": "2020-12-04T19:11:07.224368", "status": "completed"} tags=[]
# ALL_RUNS_OPTIONS = {}

# ALL_RUNS_OPTIONS['K_MIN'] = 2
# ALL_RUNS_OPTIONS['K_MAX'] = 25
# ALL_RUNS_OPTIONS['N_REPS_PER_K'] = 100
# ALL_RUNS_OPTIONS['KMEANS_N_INIT'] = 1
# ALL_RUNS_OPTIONS['AFFINITY'] = 'nearest_neighbors'
# ALL_RUNS_OPTIONS['N_JOBS'] = N_JOBS

# display(ALL_RUNS_OPTIONS)

# %% papermill={"duration": 0.026492, "end_time": "2020-12-04T19:11:07.290307", "exception": false, "start_time": "2020-12-04T19:11:07.263815", "status": "completed"} tags=[]
# output dir for this notebook
RESULTS_DIR = Path(
    conf.RESULTS["CLUSTERING_DIR"],
    'consensus_clustering'
).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %% [markdown] papermill={"duration": 0.0146, "end_time": "2020-12-04T19:11:07.319700", "exception": false, "start_time": "2020-12-04T19:11:07.305100", "status": "completed"} tags=[]
# # Get ensemble

# %% [markdown] papermill={"duration": 0.014324, "end_time": "2020-12-04T19:11:07.348367", "exception": false, "start_time": "2020-12-04T19:11:07.334043", "status": "completed"} tags=[]
# ## Load partition files

# %% papermill={"duration": 0.026972, "end_time": "2020-12-04T19:11:07.389696", "exception": false, "start_time": "2020-12-04T19:11:07.362724", "status": "completed"} tags=[]
input_dir = Path(
    conf.RESULTS["CLUSTERING_RUNS_DIR"],
).resolve()
display(input_dir)

# %% papermill={"duration": 0.027369, "end_time": "2020-12-04T19:11:07.431762", "exception": false, "start_time": "2020-12-04T19:11:07.404393", "status": "completed"} tags=[]
included_pkl_files = []

for pkl_file in input_dir.rglob('*.pkl'):
    pkl_file_str = str(pkl_file)
    
    if '-stability-' in pkl_file_str:
        continue
    
    included_pkl_files.append(pkl_file)

# %% papermill={"duration": 0.027422, "end_time": "2020-12-04T19:11:07.474668", "exception": false, "start_time": "2020-12-04T19:11:07.447246", "status": "completed"} tags=[]
display(len(included_pkl_files))
# 5 algorithms, 3 dataset versions + 1 extra run of spectral clustering with rbf in umap
assert len(included_pkl_files) == (5 * 3) + 1

# %% [markdown] papermill={"duration": 0.015115, "end_time": "2020-12-04T19:11:07.505274", "exception": false, "start_time": "2020-12-04T19:11:07.490159", "status": "completed"} tags=[]
# ## Combine partition files to get final ensemble

# %% papermill={"duration": 0.02644, "end_time": "2020-12-04T19:11:07.546405", "exception": false, "start_time": "2020-12-04T19:11:07.519965", "status": "completed"} tags=[]
n_partitions = 0

# %% papermill={"duration": 0.026565, "end_time": "2020-12-04T19:11:07.587774", "exception": false, "start_time": "2020-12-04T19:11:07.561209", "status": "completed"} tags=[]
ensembles_list = []

# %% papermill={"duration": 0.165073, "end_time": "2020-12-04T19:11:07.768183", "exception": false, "start_time": "2020-12-04T19:11:07.603110", "status": "completed"} tags=[]
for ens_file in included_pkl_files:
    ens = pd.read_pickle(ens_file)
#     ens_by_k = ens.groupby('n_clusters').apply(
#         lambda x: np.concatenate(x['partition'].apply(lambda x: x.reshape(1, -1)), axis=0)
#     )
    short_file_path = Path(*ens_file.parts[-2:])
    
    if ens.shape[0] < MIN_ENSEMBLE_SIZE:
        print(f'Less partitions than expected in {short_file_path}')
        
        ens = ens.sample(n=EXPECTED_ENSEMBLE_SIZE, replace=True, random_state=RANDOM_GENERATOR.bit_generator)
        assert ens.shape[0] == EXPECTED_ENSEMBLE_SIZE
    
    elif ens.shape[0] > MAX_ENSEMBLE_SIZE:
        print(f'More partitions than expected in {short_file_path}')
        
        ens = ens.sample(n=EXPECTED_ENSEMBLE_SIZE, random_state=RANDOM_GENERATOR.bit_generator)
        assert ens.shape[0] == EXPECTED_ENSEMBLE_SIZE

    ens_full_format = np.concatenate(ens['partition'].apply(lambda x: x.reshape(1, -1)), axis=0)
    
    # check ensemble size
#     n_parts = ensemble_full_format.shape[0]
#     if n_parts > MAX_ENSEMBLE_SIZE:
    
    n_partitions += ens_full_format.shape[0]
    
    ensembles_list.append(ens_full_format)

# %% papermill={"duration": 0.029031, "end_time": "2020-12-04T19:11:07.815621", "exception": false, "start_time": "2020-12-04T19:11:07.786590", "status": "completed"} tags=[]
display(len(ensembles_list))
assert len(ensembles_list) == len(included_pkl_files)

# %% papermill={"duration": 0.029523, "end_time": "2020-12-04T19:11:07.861861", "exception": false, "start_time": "2020-12-04T19:11:07.832338", "status": "completed"} tags=[]
n_data_objects = ensembles_list[0].shape[1]
display(n_data_objects)

# %% papermill={"duration": 0.028121, "end_time": "2020-12-04T19:11:07.906334", "exception": false, "start_time": "2020-12-04T19:11:07.878213", "status": "completed"} tags=[]
display(n_partitions)

# %% papermill={"duration": 0.256397, "end_time": "2020-12-04T19:11:08.179345", "exception": false, "start_time": "2020-12-04T19:11:07.922948", "status": "completed"} tags=[]
full_ensemble = ensembles_list[0]
for ens in ensembles_list[1:]:
    full_ensemble = np.concatenate((full_ensemble, ens), axis=0)

# %% papermill={"duration": 0.029678, "end_time": "2020-12-04T19:11:08.226584", "exception": false, "start_time": "2020-12-04T19:11:08.196906", "status": "completed"} tags=[]
display(full_ensemble.shape)
assert full_ensemble.shape == (n_partitions, n_data_objects)

# %% [markdown] papermill={"duration": 0.016499, "end_time": "2020-12-04T19:11:08.260041", "exception": false, "start_time": "2020-12-04T19:11:08.243542", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.029261, "end_time": "2020-12-04T19:11:08.305298", "exception": false, "start_time": "2020-12-04T19:11:08.276037", "status": "completed"} tags=[]
output_file = Path(
    RESULTS_DIR,
    'ensemble.npy'
).resolve()
display(output_file)

# %% papermill={"duration": 0.030471, "end_time": "2020-12-04T19:11:08.353081", "exception": false, "start_time": "2020-12-04T19:11:08.322610", "status": "completed"} tags=[]
full_ensemble

# %% papermill={"duration": 0.094293, "end_time": "2020-12-04T19:11:08.464603", "exception": false, "start_time": "2020-12-04T19:11:08.370310", "status": "completed"} tags=[]
np.save(output_file, full_ensemble)

# %% [markdown] papermill={"duration": 0.017146, "end_time": "2020-12-04T19:11:08.500021", "exception": false, "start_time": "2020-12-04T19:11:08.482875", "status": "completed"} tags=[]
# # Get coassociation matrix from ensemble

# %% papermill={"duration": 0.152649, "end_time": "2020-12-04T19:11:08.669270", "exception": false, "start_time": "2020-12-04T19:11:08.516621", "status": "completed"} tags=[]
from clustering.ensemble import get_ensemble_distance_matrix

# %% papermill={"duration": 902.205653, "end_time": "2020-12-04T19:26:10.892030", "exception": false, "start_time": "2020-12-04T19:11:08.686377", "status": "completed"} tags=[]
ensemble_coassoc_matrix = get_ensemble_distance_matrix(
    full_ensemble,
    n_jobs=conf.GENERAL["N_JOBS"],
)

# %% papermill={"duration": 0.955856, "end_time": "2020-12-04T19:26:11.869645", "exception": false, "start_time": "2020-12-04T19:26:10.913789", "status": "completed"} tags=[]
ensemble_coassoc_matrix.shape

# %% papermill={"duration": 0.03171, "end_time": "2020-12-04T19:26:11.922108", "exception": false, "start_time": "2020-12-04T19:26:11.890398", "status": "completed"} tags=[]
ensemble_coassoc_matrix

# %% [markdown] papermill={"duration": 0.017374, "end_time": "2020-12-04T19:26:11.957721", "exception": false, "start_time": "2020-12-04T19:26:11.940347", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.031667, "end_time": "2020-12-04T19:26:12.006893", "exception": false, "start_time": "2020-12-04T19:26:11.975226", "status": "completed"} tags=[]
output_file = Path(
    RESULTS_DIR,
    'ensemble_coassoc_matrix.npy'
).resolve()
display(output_file)

# %% papermill={"duration": 0.082496, "end_time": "2020-12-04T19:26:12.107274", "exception": false, "start_time": "2020-12-04T19:26:12.024778", "status": "completed"} tags=[]
np.save(output_file, ensemble_coassoc_matrix)

# %% papermill={"duration": 0.018754, "end_time": "2020-12-04T19:26:12.146014", "exception": false, "start_time": "2020-12-04T19:26:12.127260", "status": "completed"} tags=[]
