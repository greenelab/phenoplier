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
# It combines all clustering solutions generated into a single consolidated solution using consensus clustering.

# %% [markdown] tags=[]
# # Environment variables

# %% tags=[]
from IPython.display import display

# set numpy n_jobs to 1, since I'll be using n_jobs later
NUMPY_N_JOBS = 1
display(NUMPY_N_JOBS)

# %% tags=[]
# %env MKL_NUM_THREADS=$NUMPY_N_JOBS
# %env OPEN_BLAS_NUM_THREADS=$NUMPY_N_JOBS
# %env NUMEXPR_NUM_THREADS=$NUMPY_N_JOBS
# %env OMP_NUM_THREADS=$NUMPY_N_JOBS

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
from pathlib import Path

import numpy as np
import pandas as pd

import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
RANDOM_GENERATOR = np.random.default_rng(12345)

# %% [markdown] tags=[]
# ## Consensus clustering

# %% tags=[]
CLUSTERING_OPTIONS = {}

CLUSTERING_OPTIONS["K_MIN"] = 2
CLUSTERING_OPTIONS["K_MAX"] = 40

display(CLUSTERING_OPTIONS)

# %% tags=[]
# output dir for this notebook
RESULTS_DIR = Path(conf.RESULTS["CLUSTERING_DIR"], "consensus_clustering").resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %% [markdown] tags=[]
# # Load ensemble

# %% tags=[]
output_file = Path(RESULTS_DIR, "ensemble.npy").resolve()
display(output_file)

# %% tags=[]
full_ensemble = np.load(output_file)

# %% tags=[]
display(full_ensemble.shape)

# %% [markdown] tags=[]
# # Load ensemble coassociation distance matrix

# %% tags=[]
output_file = Path(RESULTS_DIR, "ensemble_coassoc_matrix.npy").resolve()
display(output_file)

# %% tags=[]
ensemble_coassoc_matrix = np.load(output_file)

# %% tags=[]
display(ensemble_coassoc_matrix.shape)

# %% tags=[]
display(ensemble_coassoc_matrix)

# %% [markdown] tags=[]
# # Consensus clustering

# %% tags=[]
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from clustering.ensemble import (
    eac_single_coassoc_matrix,
    eac_complete_coassoc_matrix,
    eac_average_coassoc_matrix,
    run_method_and_compute_agreement,
)

# %% tags=[]
all_consensus_methods = set(
    (
        eac_single_coassoc_matrix,
        eac_complete_coassoc_matrix,
        eac_average_coassoc_matrix,
    )
)
display(all_consensus_methods)

# %% tags=[]
consensus_results = []

with ProcessPoolExecutor(max_workers=conf.GENERAL["N_JOBS"]) as executor:
    tasks = {
        executor.submit(
            run_method_and_compute_agreement,
            m,
            ensemble_coassoc_matrix,
            full_ensemble,
            k,
        ): (m.__name__, k)
        for m in all_consensus_methods
        for k in range(CLUSTERING_OPTIONS["K_MIN"], CLUSTERING_OPTIONS["K_MAX"] + 1)
    }

    for future in tqdm(as_completed(tasks), total=len(tasks), disable=False, ncols=100):
        method_name, k = tasks[future]
        part, performance_values = future.result()

        method_results = {
            "method": method_name,
            "partition": part,
            "k": k,
        }
        method_results.update(performance_values)

        consensus_results.append(method_results)

# %% tags=[]
# TODO: check if each partition is really generating k clusters

# %% tags=[]
consensus_results = pd.DataFrame(consensus_results)

# %% tags=[]
display(consensus_results.shape)

# %% tags=[]
consensus_results.head()

# %% [markdown]
# ## Testing

# %%
assert not consensus_results.isna().any().any()

# %%
# check that the number of clusters in the partitions are the expected ones
_real_k_values = consensus_results["partition"].apply(lambda x: np.unique(x).shape[0])
display(_real_k_values)
assert np.all(consensus_results["k"].values == _real_k_values.values)

# %% [markdown] tags=[]
# ## Save

# %% tags=[]
output_file = Path(RESULTS_DIR, "consensus_clustering_runs.pkl").resolve()
display(output_file)

# %% tags=[]
consensus_results.to_pickle(output_file)

# %% tags=[]
