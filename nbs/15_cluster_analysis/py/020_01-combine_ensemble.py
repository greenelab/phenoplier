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

# %% [markdown] papermill={"duration": 0.014907, "end_time": "2021-01-06T18:08:32.919010", "exception": false, "start_time": "2021-01-06T18:08:32.904103", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.011616, "end_time": "2021-01-06T18:08:32.941581", "exception": false, "start_time": "2021-01-06T18:08:32.929965", "status": "completed"} tags=[]
# It combines all clustering solutions generated into a single consolidated solution using consensus clustering.

# %% [markdown] papermill={"duration": 0.01074, "end_time": "2021-01-06T18:08:32.963119", "exception": false, "start_time": "2021-01-06T18:08:32.952379", "status": "completed"} tags=[]
# # Environment variables

# %% papermill={"duration": 0.020808, "end_time": "2021-01-06T18:08:32.995080", "exception": false, "start_time": "2021-01-06T18:08:32.974272", "status": "completed"} tags=[]
from IPython.display import display

# set numpy n_jobs to 1, since I'll be using n_jobs later
NUMPY_N_JOBS = 1
display(NUMPY_N_JOBS)

# %% papermill={"duration": 0.01801, "end_time": "2021-01-06T18:08:33.024973", "exception": false, "start_time": "2021-01-06T18:08:33.006963", "status": "completed"} tags=[]
# %env MKL_NUM_THREADS=$NUMPY_N_JOBS
# %env OPEN_BLAS_NUM_THREADS=$NUMPY_N_JOBS
# %env NUMEXPR_NUM_THREADS=$NUMPY_N_JOBS
# %env OMP_NUM_THREADS=$NUMPY_N_JOBS

# %% [markdown] papermill={"duration": 0.011154, "end_time": "2021-01-06T18:08:33.048088", "exception": false, "start_time": "2021-01-06T18:08:33.036934", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.022207, "end_time": "2021-01-06T18:08:33.081611", "exception": false, "start_time": "2021-01-06T18:08:33.059404", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 0.185905, "end_time": "2021-01-06T18:08:33.279362", "exception": false, "start_time": "2021-01-06T18:08:33.093457", "status": "completed"} tags=[]
from pathlib import Path

import numpy as np
import pandas as pd

import conf

# %% [markdown] papermill={"duration": 0.011262, "end_time": "2021-01-06T18:08:33.302648", "exception": false, "start_time": "2021-01-06T18:08:33.291386", "status": "completed"} tags=[]
# # Settings

# %% papermill={"duration": 0.021881, "end_time": "2021-01-06T18:08:33.335974", "exception": false, "start_time": "2021-01-06T18:08:33.314093", "status": "completed"} tags=[]
RANDOM_GENERATOR = np.random.default_rng(12345)

# %% [markdown] papermill={"duration": 0.011343, "end_time": "2021-01-06T18:08:33.359163", "exception": false, "start_time": "2021-01-06T18:08:33.347820", "status": "completed"} tags=[]
# ## Ensemble size

# %% papermill={"duration": 0.021333, "end_time": "2021-01-06T18:08:33.391796", "exception": false, "start_time": "2021-01-06T18:08:33.370463", "status": "completed"} tags=[]
EXPECTED_ENSEMBLE_SIZE = 295

MIN_ENSEMBLE_SIZE = 290
MAX_ENSEMBLE_SIZE = 300

# %% [markdown] papermill={"duration": 0.011342, "end_time": "2021-01-06T18:08:33.414750", "exception": false, "start_time": "2021-01-06T18:08:33.403408", "status": "completed"} tags=[]
# ## Consensus clustering

# %% papermill={"duration": 0.022254, "end_time": "2021-01-06T18:08:33.448424", "exception": false, "start_time": "2021-01-06T18:08:33.426170", "status": "completed"} tags=[]
CLUSTERING_OPTIONS = {}

CLUSTERING_OPTIONS["K_MIN"] = 2
CLUSTERING_OPTIONS["K_MAX"] = 40

display(CLUSTERING_OPTIONS)

# %% papermill={"duration": 0.023058, "end_time": "2021-01-06T18:08:33.483583", "exception": false, "start_time": "2021-01-06T18:08:33.460525", "status": "completed"} tags=[]
# output dir for this notebook
RESULTS_DIR = Path(conf.RESULTS["CLUSTERING_DIR"], "consensus_clustering").resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %% [markdown] papermill={"duration": 0.011987, "end_time": "2021-01-06T18:08:33.507797", "exception": false, "start_time": "2021-01-06T18:08:33.495810", "status": "completed"} tags=[]
# # Get ensemble

# %% papermill={"duration": 0.022423, "end_time": "2021-01-06T18:08:33.541946", "exception": false, "start_time": "2021-01-06T18:08:33.519523", "status": "completed"} tags=[]
output_file = Path(RESULTS_DIR, "ensemble.npy").resolve()
display(output_file)

# %% papermill={"duration": 0.053824, "end_time": "2021-01-06T18:08:33.608192", "exception": false, "start_time": "2021-01-06T18:08:33.554368", "status": "completed"} tags=[]
full_ensemble = np.load(output_file)

# %% papermill={"duration": 0.025221, "end_time": "2021-01-06T18:08:33.649327", "exception": false, "start_time": "2021-01-06T18:08:33.624106", "status": "completed"} tags=[]
display(full_ensemble.shape)

# %% [markdown] papermill={"duration": 0.013861, "end_time": "2021-01-06T18:08:33.677215", "exception": false, "start_time": "2021-01-06T18:08:33.663354", "status": "completed"} tags=[]
# # Get ensemble coassociation distance matrix

# %% papermill={"duration": 0.023294, "end_time": "2021-01-06T18:08:33.713148", "exception": false, "start_time": "2021-01-06T18:08:33.689854", "status": "completed"} tags=[]
output_file = Path(RESULTS_DIR, "ensemble_coassoc_matrix.npy").resolve()
display(output_file)

# %% papermill={"duration": 0.046839, "end_time": "2021-01-06T18:08:33.772795", "exception": false, "start_time": "2021-01-06T18:08:33.725956", "status": "completed"} tags=[]
ensemble_coassoc_matrix = np.load(output_file)

# %% papermill={"duration": 0.025255, "end_time": "2021-01-06T18:08:33.813783", "exception": false, "start_time": "2021-01-06T18:08:33.788528", "status": "completed"} tags=[]
display(ensemble_coassoc_matrix.shape)

# %% papermill={"duration": 0.024706, "end_time": "2021-01-06T18:08:33.852680", "exception": false, "start_time": "2021-01-06T18:08:33.827974", "status": "completed"} tags=[]
display(ensemble_coassoc_matrix)

# %% [markdown] papermill={"duration": 0.014014, "end_time": "2021-01-06T18:08:33.881034", "exception": false, "start_time": "2021-01-06T18:08:33.867020", "status": "completed"} tags=[]
# # Consensus clustering

# %% papermill={"duration": 0.339003, "end_time": "2021-01-06T18:08:34.232866", "exception": false, "start_time": "2021-01-06T18:08:33.893863", "status": "completed"} tags=[]
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from clustering.ensemble import (
    eac_single_coassoc_matrix,
    eac_complete_coassoc_matrix,
    eac_average_coassoc_matrix,
    run_method_and_compute_agreement,
)

# %% papermill={"duration": 0.026826, "end_time": "2021-01-06T18:08:34.273247", "exception": false, "start_time": "2021-01-06T18:08:34.246421", "status": "completed"} tags=[]
all_consensus_methods = set(
    (
        eac_single_coassoc_matrix,
        eac_complete_coassoc_matrix,
        eac_average_coassoc_matrix,
    )
)
display(all_consensus_methods)

# %% papermill={"duration": 1322.038163, "end_time": "2021-01-06T18:30:36.325753", "exception": false, "start_time": "2021-01-06T18:08:34.287590", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.047145, "end_time": "2021-01-06T18:30:36.405216", "exception": false, "start_time": "2021-01-06T18:30:36.358071", "status": "completed"} tags=[]
# TODO: check if each partition is really generating k clusters

# %% papermill={"duration": 0.047199, "end_time": "2021-01-06T18:30:36.485425", "exception": false, "start_time": "2021-01-06T18:30:36.438226", "status": "completed"} tags=[]
consensus_results = pd.DataFrame(consensus_results)

# %% papermill={"duration": 0.04655, "end_time": "2021-01-06T18:30:36.565098", "exception": false, "start_time": "2021-01-06T18:30:36.518548", "status": "completed"} tags=[]
display(consensus_results.shape)

# %% papermill={"duration": 0.060266, "end_time": "2021-01-06T18:30:36.659228", "exception": false, "start_time": "2021-01-06T18:30:36.598962", "status": "completed"} tags=[]
consensus_results.head()

# %% [markdown] papermill={"duration": 0.032421, "end_time": "2021-01-06T18:30:36.726371", "exception": false, "start_time": "2021-01-06T18:30:36.693950", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.046538, "end_time": "2021-01-06T18:30:36.805608", "exception": false, "start_time": "2021-01-06T18:30:36.759070", "status": "completed"} tags=[]
output_file = Path(RESULTS_DIR, "consensus_clustering_runs.pkl").resolve()
display(output_file)

# %% papermill={"duration": 0.053152, "end_time": "2021-01-06T18:30:36.893354", "exception": false, "start_time": "2021-01-06T18:30:36.840202", "status": "completed"} tags=[]
consensus_results.to_pickle(output_file)

# %% papermill={"duration": 0.033376, "end_time": "2021-01-06T18:30:36.960610", "exception": false, "start_time": "2021-01-06T18:30:36.927234", "status": "completed"} tags=[]
