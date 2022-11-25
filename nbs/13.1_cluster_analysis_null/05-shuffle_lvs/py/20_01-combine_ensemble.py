# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
#     formats: ipynb,py//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
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
np.random.seed(0)

# %% tags=[]
RANDOM_STATES_ITER = iter(np.random.randint(0, np.iinfo(np.int32).max, size=100000))

# %% tags=[]
# n_init parameter for DeltaSpectralClustering
# a high number should produce more stable final solutions
SC_N_INIT = 50

# %%
NULL_DIR = conf.RESULTS["CLUSTERING_NULL_DIR"] / "shuffle_lvs"

# %% [markdown] tags=[]
# ## Consensus clustering

# %% tags=[]
CLUSTERING_OPTIONS = {}

CLUSTERING_OPTIONS["K_MIN"] = 2
CLUSTERING_OPTIONS["K_MAX"] = 60

display(CLUSTERING_OPTIONS)

# %% tags=[]
# output dir for this notebook
RESULTS_DIR = Path(
    NULL_DIR, "consensus_clustering"
).resolve()
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

from clustering.ensembles.utils import (
    run_method_and_compute_agreement,
)
from clustering.ensembles.eac import (
    eac_single_coassoc_matrix,
    eac_complete_coassoc_matrix,
    eac_average_coassoc_matrix,
)
from clustering.ensembles.spectral import scc


# %% [markdown] tags=[]
# Define spectral consensus clustering methods with delta values found in pre-analysis:

# %% tags=[]
def scc_020(coassoc_distance_matrix, k, **kwargs):
    return scc(
        coassoc_distance_matrix,
        k,
        delta=0.20,
        ensemble_is_coassoc_matrix=True,
        **kwargs
    )


def scc_025(coassoc_distance_matrix, k, **kwargs):
    return scc(
        coassoc_distance_matrix,
        k,
        delta=0.25,
        ensemble_is_coassoc_matrix=True,
        **kwargs
    )


def scc_030(coassoc_distance_matrix, k, **kwargs):
    return scc(
        coassoc_distance_matrix,
        k,
        delta=0.30,
        ensemble_is_coassoc_matrix=True,
        **kwargs
    )


def scc_050(coassoc_distance_matrix, k, **kwargs):
    return scc(
        coassoc_distance_matrix,
        k,
        delta=0.50,
        ensemble_is_coassoc_matrix=True,
        **kwargs
    )


# %% tags=[]
all_consensus_methods = set(
    (
        eac_single_coassoc_matrix,
        eac_complete_coassoc_matrix,
        eac_average_coassoc_matrix,
        scc_020,
        scc_025,
        scc_030,
        scc_050,
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
            n_init=SC_N_INIT,
            random_state=next(RANDOM_STATES_ITER),
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
consensus_results = pd.DataFrame(consensus_results)

# %% tags=[]
display(consensus_results.shape)

# %% tags=[]
consensus_results.head()

# %% [markdown] tags=[]
# ## Testing

# %% tags=[]
assert not consensus_results.isna().any().any()

# %% tags=[]
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
