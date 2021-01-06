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

# %% [markdown] papermill={"duration": 0.050434, "end_time": "2021-01-06T17:53:28.189744", "exception": false, "start_time": "2021-01-06T17:53:28.139310", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.01479, "end_time": "2021-01-06T17:53:28.221280", "exception": false, "start_time": "2021-01-06T17:53:28.206490", "status": "completed"} tags=[]
# TODO

# %% [markdown] papermill={"duration": 0.014745, "end_time": "2021-01-06T17:53:28.250830", "exception": false, "start_time": "2021-01-06T17:53:28.236085", "status": "completed"} tags=[]
# # Environment variables

# %% papermill={"duration": 0.028465, "end_time": "2021-01-06T17:53:28.294088", "exception": false, "start_time": "2021-01-06T17:53:28.265623", "status": "completed"} tags=[]
from IPython.display import display

import conf

N_JOBS = conf.GENERAL["N_JOBS"]
display(N_JOBS)

# %% papermill={"duration": 0.021636, "end_time": "2021-01-06T17:53:28.331359", "exception": false, "start_time": "2021-01-06T17:53:28.309723", "status": "completed"} tags=[]
# %env MKL_NUM_THREADS=$N_JOBS
# %env OPEN_BLAS_NUM_THREADS=$N_JOBS
# %env NUMEXPR_NUM_THREADS=$N_JOBS
# %env OMP_NUM_THREADS=$N_JOBS

# %% [markdown] papermill={"duration": 0.015769, "end_time": "2021-01-06T17:53:28.363086", "exception": false, "start_time": "2021-01-06T17:53:28.347317", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.025588, "end_time": "2021-01-06T17:53:28.403918", "exception": false, "start_time": "2021-01-06T17:53:28.378330", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 0.583418, "end_time": "2021-01-06T17:53:29.003421", "exception": false, "start_time": "2021-01-06T17:53:28.420003", "status": "completed"} tags=[]
from pathlib import Path

import numpy as np
import pandas as pd

# from sklearn.cluster import SpectralClustering
# import umap
import matplotlib.pyplot as plt
import seaborn as sns

from utils import generate_result_set_name

# %% [markdown] papermill={"duration": 0.018459, "end_time": "2021-01-06T17:53:29.041746", "exception": false, "start_time": "2021-01-06T17:53:29.023287", "status": "completed"} tags=[]
# # Settings

# %% papermill={"duration": 0.027679, "end_time": "2021-01-06T17:53:29.085604", "exception": false, "start_time": "2021-01-06T17:53:29.057925", "status": "completed"} tags=[]
RANDOM_GENERATOR = np.random.default_rng(12345)

# %% [markdown] papermill={"duration": 0.014927, "end_time": "2021-01-06T17:53:29.116263", "exception": false, "start_time": "2021-01-06T17:53:29.101336", "status": "completed"} tags=[]
# ## Ensemble size

# %% papermill={"duration": 0.027076, "end_time": "2021-01-06T17:53:29.158513", "exception": false, "start_time": "2021-01-06T17:53:29.131437", "status": "completed"} tags=[]
EXPECTED_ENSEMBLE_SIZE = 295

MIN_ENSEMBLE_SIZE = 290
MAX_ENSEMBLE_SIZE = 300

# %% [markdown] papermill={"duration": 0.014985, "end_time": "2021-01-06T17:53:29.188956", "exception": false, "start_time": "2021-01-06T17:53:29.173971", "status": "completed"} tags=[]
# ## Data subsets included

# %% papermill={"duration": 0.026658, "end_time": "2021-01-06T17:53:29.230827", "exception": false, "start_time": "2021-01-06T17:53:29.204169", "status": "completed"} tags=[]
# DATA_SUBSETS = [
#     'z_score_std',
#     'pca',
#     'umap'
# ]

# %% [markdown] papermill={"duration": 0.01544, "end_time": "2021-01-06T17:53:29.262073", "exception": false, "start_time": "2021-01-06T17:53:29.246633", "status": "completed"} tags=[]
# ## Algorithms included

# %% papermill={"duration": 0.026901, "end_time": "2021-01-06T17:53:29.304024", "exception": false, "start_time": "2021-01-06T17:53:29.277123", "status": "completed"} tags=[]
# ALGORITHMS = [
#     'KMeans',
#     COMPLETE
# ]

# %% [markdown] papermill={"duration": 0.015304, "end_time": "2021-01-06T17:53:29.334672", "exception": false, "start_time": "2021-01-06T17:53:29.319368", "status": "completed"} tags=[]
# ## Input data

# %% papermill={"duration": 0.026845, "end_time": "2021-01-06T17:53:29.377088", "exception": false, "start_time": "2021-01-06T17:53:29.350243", "status": "completed"} tags=[]
# DATA_SUBSET = 'z_score_std'

# INPUT_DIR = Path(
#     conf.RESULTS["DATA_TRANSFORMATIONS_DIR"],
#     DATA_SUBSET,
# ).resolve()
# display(INPUT_DIR)

# %% papermill={"duration": 0.027078, "end_time": "2021-01-06T17:53:29.419743", "exception": false, "start_time": "2021-01-06T17:53:29.392665", "status": "completed"} tags=[]
# INPUT_STEM = 'projection-smultixcan-efo_partial-mashr-zscores'

# %% papermill={"duration": 0.027245, "end_time": "2021-01-06T17:53:29.462967", "exception": false, "start_time": "2021-01-06T17:53:29.435722", "status": "completed"} tags=[]
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

# %% [markdown] papermill={"duration": 0.015336, "end_time": "2021-01-06T17:53:29.494186", "exception": false, "start_time": "2021-01-06T17:53:29.478850", "status": "completed"} tags=[]
# ## Consensus clustering

# %% papermill={"duration": 0.02692, "end_time": "2021-01-06T17:53:29.536256", "exception": false, "start_time": "2021-01-06T17:53:29.509336", "status": "completed"} tags=[]
# ALL_RUNS_OPTIONS = {}

# ALL_RUNS_OPTIONS['K_MIN'] = 2
# ALL_RUNS_OPTIONS['K_MAX'] = 25
# ALL_RUNS_OPTIONS['N_REPS_PER_K'] = 100
# ALL_RUNS_OPTIONS['KMEANS_N_INIT'] = 1
# ALL_RUNS_OPTIONS['AFFINITY'] = 'nearest_neighbors'
# ALL_RUNS_OPTIONS['N_JOBS'] = N_JOBS

# display(ALL_RUNS_OPTIONS)

# %% papermill={"duration": 0.028877, "end_time": "2021-01-06T17:53:29.581467", "exception": false, "start_time": "2021-01-06T17:53:29.552590", "status": "completed"} tags=[]
# output dir for this notebook
RESULTS_DIR = Path(conf.RESULTS["CLUSTERING_DIR"], "consensus_clustering").resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %% [markdown] papermill={"duration": 0.016274, "end_time": "2021-01-06T17:53:29.614634", "exception": false, "start_time": "2021-01-06T17:53:29.598360", "status": "completed"} tags=[]
# # Get ensemble

# %% [markdown] papermill={"duration": 0.015328, "end_time": "2021-01-06T17:53:29.645229", "exception": false, "start_time": "2021-01-06T17:53:29.629901", "status": "completed"} tags=[]
# ## Load partition files

# %% papermill={"duration": 0.028451, "end_time": "2021-01-06T17:53:29.689136", "exception": false, "start_time": "2021-01-06T17:53:29.660685", "status": "completed"} tags=[]
input_dir = Path(
    conf.RESULTS["CLUSTERING_RUNS_DIR"],
).resolve()
display(input_dir)

# %% papermill={"duration": 0.028297, "end_time": "2021-01-06T17:53:29.733740", "exception": false, "start_time": "2021-01-06T17:53:29.705443", "status": "completed"} tags=[]
included_pkl_files = []

for pkl_file in input_dir.rglob("*.pkl"):
    pkl_file_str = str(pkl_file)

    if "-stability-" in pkl_file_str:
        continue

    included_pkl_files.append(pkl_file)

# %% papermill={"duration": 0.028363, "end_time": "2021-01-06T17:53:29.778476", "exception": false, "start_time": "2021-01-06T17:53:29.750113", "status": "completed"} tags=[]
display(len(included_pkl_files))
# 5 algorithms, 3 dataset versions
assert len(included_pkl_files) == 5 * 3

# %% [markdown] papermill={"duration": 0.016148, "end_time": "2021-01-06T17:53:29.811249", "exception": false, "start_time": "2021-01-06T17:53:29.795101", "status": "completed"} tags=[]
# ## Combine partition files to get final ensemble

# %% papermill={"duration": 0.027655, "end_time": "2021-01-06T17:53:29.854806", "exception": false, "start_time": "2021-01-06T17:53:29.827151", "status": "completed"} tags=[]
n_partitions = 0

# %% papermill={"duration": 0.027765, "end_time": "2021-01-06T17:53:29.899075", "exception": false, "start_time": "2021-01-06T17:53:29.871310", "status": "completed"} tags=[]
ensembles_list = []

# %% papermill={"duration": 0.171881, "end_time": "2021-01-06T17:53:30.087234", "exception": false, "start_time": "2021-01-06T17:53:29.915353", "status": "completed"} tags=[]
for ens_file in included_pkl_files:
    ens = pd.read_pickle(ens_file)
    #     ens_by_k = ens.groupby('n_clusters').apply(
    #         lambda x: np.concatenate(x['partition'].apply(lambda x: x.reshape(1, -1)), axis=0)
    #     )
    short_file_path = Path(*ens_file.parts[-2:])

    if ens.shape[0] < MIN_ENSEMBLE_SIZE:
        print(f"Less partitions than expected in {short_file_path}")

        ens = ens.sample(
            n=EXPECTED_ENSEMBLE_SIZE,
            replace=True,
            random_state=RANDOM_GENERATOR.bit_generator,
        )
        assert ens.shape[0] == EXPECTED_ENSEMBLE_SIZE

    elif ens.shape[0] > MAX_ENSEMBLE_SIZE:
        print(f"More partitions than expected in {short_file_path}")

        ens = ens.sample(
            n=EXPECTED_ENSEMBLE_SIZE, random_state=RANDOM_GENERATOR.bit_generator
        )
        assert ens.shape[0] == EXPECTED_ENSEMBLE_SIZE

    ens_full_format = np.concatenate(
        ens["partition"].apply(lambda x: x.reshape(1, -1)), axis=0
    )

    # check ensemble size
    #     n_parts = ensemble_full_format.shape[0]
    #     if n_parts > MAX_ENSEMBLE_SIZE:

    n_partitions += ens_full_format.shape[0]

    ensembles_list.append(ens_full_format)

# %% papermill={"duration": 0.030393, "end_time": "2021-01-06T17:53:30.137599", "exception": false, "start_time": "2021-01-06T17:53:30.107206", "status": "completed"} tags=[]
display(len(ensembles_list))
assert len(ensembles_list) == len(included_pkl_files)

# %% papermill={"duration": 0.030218, "end_time": "2021-01-06T17:53:30.185269", "exception": false, "start_time": "2021-01-06T17:53:30.155051", "status": "completed"} tags=[]
n_data_objects = ensembles_list[0].shape[1]
display(n_data_objects)

# %% papermill={"duration": 0.029575, "end_time": "2021-01-06T17:53:30.231911", "exception": false, "start_time": "2021-01-06T17:53:30.202336", "status": "completed"} tags=[]
display(n_partitions)

# %% papermill={"duration": 0.227121, "end_time": "2021-01-06T17:53:30.476068", "exception": false, "start_time": "2021-01-06T17:53:30.248947", "status": "completed"} tags=[]
full_ensemble = ensembles_list[0]
for ens in ensembles_list[1:]:
    full_ensemble = np.concatenate((full_ensemble, ens), axis=0)

# %% papermill={"duration": 0.029772, "end_time": "2021-01-06T17:53:30.523712", "exception": false, "start_time": "2021-01-06T17:53:30.493940", "status": "completed"} tags=[]
display(full_ensemble.shape)
assert full_ensemble.shape == (n_partitions, n_data_objects)

# %% [markdown] papermill={"duration": 0.01754, "end_time": "2021-01-06T17:53:30.558695", "exception": false, "start_time": "2021-01-06T17:53:30.541155", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.029813, "end_time": "2021-01-06T17:53:30.605467", "exception": false, "start_time": "2021-01-06T17:53:30.575654", "status": "completed"} tags=[]
output_file = Path(RESULTS_DIR, "ensemble.npy").resolve()
display(output_file)

# %% papermill={"duration": 0.030568, "end_time": "2021-01-06T17:53:30.653918", "exception": false, "start_time": "2021-01-06T17:53:30.623350", "status": "completed"} tags=[]
full_ensemble

# %% papermill={"duration": 0.104906, "end_time": "2021-01-06T17:53:30.776649", "exception": false, "start_time": "2021-01-06T17:53:30.671743", "status": "completed"} tags=[]
np.save(output_file, full_ensemble)

# %% [markdown] papermill={"duration": 0.017585, "end_time": "2021-01-06T17:53:30.813311", "exception": false, "start_time": "2021-01-06T17:53:30.795726", "status": "completed"} tags=[]
# # Get coassociation matrix from ensemble

# %% papermill={"duration": 0.150097, "end_time": "2021-01-06T17:53:30.980531", "exception": false, "start_time": "2021-01-06T17:53:30.830434", "status": "completed"} tags=[]
from clustering.ensemble import get_ensemble_distance_matrix

# %% papermill={"duration": 898.609421, "end_time": "2021-01-06T18:08:29.607438", "exception": false, "start_time": "2021-01-06T17:53:30.998017", "status": "completed"} tags=[]
ensemble_coassoc_matrix = get_ensemble_distance_matrix(
    full_ensemble,
    n_jobs=conf.GENERAL["N_JOBS"],
)

# %% papermill={"duration": 0.934727, "end_time": "2021-01-06T18:08:30.563137", "exception": false, "start_time": "2021-01-06T18:08:29.628410", "status": "completed"} tags=[]
ensemble_coassoc_matrix.shape

# %% papermill={"duration": 0.033353, "end_time": "2021-01-06T18:08:30.616718", "exception": false, "start_time": "2021-01-06T18:08:30.583365", "status": "completed"} tags=[]
ensemble_coassoc_matrix

# %% [markdown] papermill={"duration": 0.018322, "end_time": "2021-01-06T18:08:30.654669", "exception": false, "start_time": "2021-01-06T18:08:30.636347", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.031682, "end_time": "2021-01-06T18:08:30.704184", "exception": false, "start_time": "2021-01-06T18:08:30.672502", "status": "completed"} tags=[]
output_file = Path(RESULTS_DIR, "ensemble_coassoc_matrix.npy").resolve()
display(output_file)

# %% papermill={"duration": 0.08571, "end_time": "2021-01-06T18:08:30.808667", "exception": false, "start_time": "2021-01-06T18:08:30.722957", "status": "completed"} tags=[]
np.save(output_file, ensemble_coassoc_matrix)

# %% papermill={"duration": 0.019656, "end_time": "2021-01-06T18:08:30.849242", "exception": false, "start_time": "2021-01-06T18:08:30.829586", "status": "completed"} tags=[]
