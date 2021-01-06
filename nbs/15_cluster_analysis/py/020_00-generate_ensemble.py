# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill
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

# %% [markdown] tags=[]
# # Description

# %% [markdown] tags=[]
# TODO

# %% [markdown] tags=[]
# # Environment variables

# %% tags=[]
from IPython.display import display

import conf

N_JOBS = conf.GENERAL["N_JOBS"]
display(N_JOBS)

# %% tags=[]
# %env MKL_NUM_THREADS=$N_JOBS
# %env OPEN_BLAS_NUM_THREADS=$N_JOBS
# %env NUMEXPR_NUM_THREADS=$N_JOBS
# %env OMP_NUM_THREADS=$N_JOBS

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
from pathlib import Path

import numpy as np
import pandas as pd

# from sklearn.cluster import SpectralClustering
# import umap
import matplotlib.pyplot as plt
import seaborn as sns

from utils import generate_result_set_name

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
RANDOM_GENERATOR = np.random.default_rng(12345)

# %% [markdown] tags=[]
# ## Ensemble size

# %% tags=[]
EXPECTED_ENSEMBLE_SIZE = 295

MIN_ENSEMBLE_SIZE = 290
MAX_ENSEMBLE_SIZE = 300

# %% [markdown] tags=[]
# ## Data subsets included

# %% tags=[]
# DATA_SUBSETS = [
#     'z_score_std',
#     'pca',
#     'umap'
# ]

# %% [markdown] tags=[]
# ## Algorithms included

# %% tags=[]
# ALGORITHMS = [
#     'KMeans',
#     COMPLETE
# ]

# %% [markdown] tags=[]
# ## Input data

# %% tags=[]
# DATA_SUBSET = 'z_score_std'

# INPUT_DIR = Path(
#     conf.RESULTS["DATA_TRANSFORMATIONS_DIR"],
#     DATA_SUBSET,
# ).resolve()
# display(INPUT_DIR)

# %% tags=[]
# INPUT_STEM = 'projection-smultixcan-efo_partial-mashr-zscores'

# %% tags=[]
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

# %% [markdown] tags=[]
# ## Consensus clustering

# %% tags=[]
# ALL_RUNS_OPTIONS = {}

# ALL_RUNS_OPTIONS['K_MIN'] = 2
# ALL_RUNS_OPTIONS['K_MAX'] = 25
# ALL_RUNS_OPTIONS['N_REPS_PER_K'] = 100
# ALL_RUNS_OPTIONS['KMEANS_N_INIT'] = 1
# ALL_RUNS_OPTIONS['AFFINITY'] = 'nearest_neighbors'
# ALL_RUNS_OPTIONS['N_JOBS'] = N_JOBS

# display(ALL_RUNS_OPTIONS)

# %% tags=[]
# output dir for this notebook
RESULTS_DIR = Path(conf.RESULTS["CLUSTERING_DIR"], "consensus_clustering").resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %% [markdown] tags=[]
# # Get ensemble

# %% [markdown] tags=[]
# ## Load partition files

# %% tags=[]
input_dir = Path(
    conf.RESULTS["CLUSTERING_RUNS_DIR"],
).resolve()
display(input_dir)

# %% tags=[]
included_pkl_files = []

for pkl_file in input_dir.rglob("*.pkl"):
    pkl_file_str = str(pkl_file)

    if "-stability-" in pkl_file_str:
        continue

    included_pkl_files.append(pkl_file)

# %% tags=[]
display(len(included_pkl_files))
# 5 algorithms, 3 dataset versions
assert len(included_pkl_files) == 5 * 3

# %% [markdown] tags=[]
# ## Combine partition files to get final ensemble

# %% tags=[]
n_partitions = 0

# %% tags=[]
ensembles_list = []

# %% tags=[]
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

# %% tags=[]
display(len(ensembles_list))
assert len(ensembles_list) == len(included_pkl_files)

# %% tags=[]
n_data_objects = ensembles_list[0].shape[1]
display(n_data_objects)

# %% tags=[]
display(n_partitions)

# %% tags=[]
full_ensemble = ensembles_list[0]
for ens in ensembles_list[1:]:
    full_ensemble = np.concatenate((full_ensemble, ens), axis=0)

# %% tags=[]
display(full_ensemble.shape)
assert full_ensemble.shape == (n_partitions, n_data_objects)

# %% [markdown] tags=[]
# ## Save

# %% tags=[]
output_file = Path(RESULTS_DIR, "ensemble.npy").resolve()
display(output_file)

# %% tags=[]
full_ensemble

# %% tags=[]
np.save(output_file, full_ensemble)

# %% [markdown] tags=[]
# # Get coassociation matrix from ensemble

# %% tags=[]
from clustering.ensemble import get_ensemble_distance_matrix

# %% tags=[]
ensemble_coassoc_matrix = get_ensemble_distance_matrix(
    full_ensemble,
    n_jobs=conf.GENERAL["N_JOBS"],
)

# %% tags=[]
ensemble_coassoc_matrix.shape

# %% tags=[]
ensemble_coassoc_matrix

# %% [markdown] tags=[]
# ## Save

# %% tags=[]
output_file = Path(RESULTS_DIR, "ensemble_coassoc_matrix.npy").resolve()
display(output_file)

# %% tags=[]
np.save(output_file, ensemble_coassoc_matrix)

# %% tags=[]
