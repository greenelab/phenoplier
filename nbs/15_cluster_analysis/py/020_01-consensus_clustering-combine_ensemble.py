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

# %% [markdown] papermill={"duration": 0.013988, "end_time": "2020-12-06T04:09:56.303878", "exception": false, "start_time": "2020-12-06T04:09:56.289890", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.010329, "end_time": "2020-12-06T04:09:56.324771", "exception": false, "start_time": "2020-12-06T04:09:56.314442", "status": "completed"} tags=[]
# It combines all clustering solutions generated into a single consolidated solution using consensus clustering.

# %% [markdown] papermill={"duration": 0.010325, "end_time": "2020-12-06T04:09:56.345504", "exception": false, "start_time": "2020-12-06T04:09:56.335179", "status": "completed"} tags=[]
# # Environment variables

# %% papermill={"duration": 0.021255, "end_time": "2020-12-06T04:09:56.377325", "exception": false, "start_time": "2020-12-06T04:09:56.356070", "status": "completed"} tags=[]
from IPython.display import display

# set numpy n_jobs to 1, since I'll be using n_jobs later
NUMPY_N_JOBS = 1
display(NUMPY_N_JOBS)

# %% papermill={"duration": 0.017167, "end_time": "2020-12-06T04:09:56.405721", "exception": false, "start_time": "2020-12-06T04:09:56.388554", "status": "completed"} tags=[]
# %env MKL_NUM_THREADS=$NUMPY_N_JOBS
# %env OPEN_BLAS_NUM_THREADS=$NUMPY_N_JOBS
# %env NUMEXPR_NUM_THREADS=$NUMPY_N_JOBS
# %env OMP_NUM_THREADS=$NUMPY_N_JOBS

# %% [markdown] papermill={"duration": 0.010703, "end_time": "2020-12-06T04:09:56.427422", "exception": false, "start_time": "2020-12-06T04:09:56.416719", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.021553, "end_time": "2020-12-06T04:09:56.459777", "exception": false, "start_time": "2020-12-06T04:09:56.438224", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 0.581222, "end_time": "2020-12-06T04:09:57.052113", "exception": false, "start_time": "2020-12-06T04:09:56.470891", "status": "completed"} tags=[]
from pathlib import Path

import numpy as np
import pandas as pd
# from sklearn.cluster import SpectralClustering
# import umap
import matplotlib.pyplot as plt
import seaborn as sns

import conf
from utils import generate_result_set_name

# %% [markdown] papermill={"duration": 0.013523, "end_time": "2020-12-06T04:09:57.079973", "exception": false, "start_time": "2020-12-06T04:09:57.066450", "status": "completed"} tags=[]
# # Settings

# %% papermill={"duration": 0.02374, "end_time": "2020-12-06T04:09:57.115924", "exception": false, "start_time": "2020-12-06T04:09:57.092184", "status": "completed"} tags=[]
RANDOM_GENERATOR = np.random.default_rng(12345)

# %% [markdown] papermill={"duration": 0.010712, "end_time": "2020-12-06T04:09:57.138422", "exception": false, "start_time": "2020-12-06T04:09:57.127710", "status": "completed"} tags=[]
# ## Ensemble size

# %% papermill={"duration": 0.022805, "end_time": "2020-12-06T04:09:57.172114", "exception": false, "start_time": "2020-12-06T04:09:57.149309", "status": "completed"} tags=[]
EXPECTED_ENSEMBLE_SIZE = 295

MIN_ENSEMBLE_SIZE = 290
MAX_ENSEMBLE_SIZE = 300

# %% [markdown] papermill={"duration": 0.011255, "end_time": "2020-12-06T04:09:57.194771", "exception": false, "start_time": "2020-12-06T04:09:57.183516", "status": "completed"} tags=[]
# ## Consensus clustering

# %% papermill={"duration": 0.024018, "end_time": "2020-12-06T04:09:57.229903", "exception": false, "start_time": "2020-12-06T04:09:57.205885", "status": "completed"} tags=[]
CLUSTERING_OPTIONS = {}

CLUSTERING_OPTIONS['K_MIN'] = 2
CLUSTERING_OPTIONS['K_MAX'] = 40

display(CLUSTERING_OPTIONS)

# %% papermill={"duration": 0.024395, "end_time": "2020-12-06T04:09:57.265679", "exception": false, "start_time": "2020-12-06T04:09:57.241284", "status": "completed"} tags=[]
# output dir for this notebook
RESULTS_DIR = Path(
    conf.RESULTS["CLUSTERING_DIR"],
    'consensus_clustering'
).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %% [markdown] papermill={"duration": 0.011828, "end_time": "2020-12-06T04:09:57.289616", "exception": false, "start_time": "2020-12-06T04:09:57.277788", "status": "completed"} tags=[]
# # Get ensemble

# %% papermill={"duration": 0.024186, "end_time": "2020-12-06T04:09:57.325241", "exception": false, "start_time": "2020-12-06T04:09:57.301055", "status": "completed"} tags=[]
output_file = Path(
    RESULTS_DIR,
    'ensemble.npy'
).resolve()
display(output_file)

# %% papermill={"duration": 0.05132, "end_time": "2020-12-06T04:09:57.388708", "exception": false, "start_time": "2020-12-06T04:09:57.337388", "status": "completed"} tags=[]
full_ensemble = np.load(output_file)

# %% papermill={"duration": 0.027393, "end_time": "2020-12-06T04:09:57.430809", "exception": false, "start_time": "2020-12-06T04:09:57.403416", "status": "completed"} tags=[]
full_ensemble.shape

# %% papermill={"duration": 0.024231, "end_time": "2020-12-06T04:09:57.467681", "exception": false, "start_time": "2020-12-06T04:09:57.443450", "status": "completed"} tags=[]
full_ensemble.shape

# %% [markdown] papermill={"duration": 0.012373, "end_time": "2020-12-06T04:09:57.492676", "exception": false, "start_time": "2020-12-06T04:09:57.480303", "status": "completed"} tags=[]
# # Get ensemble coassociation distance matrix

# %% papermill={"duration": 0.025787, "end_time": "2020-12-06T04:09:57.530972", "exception": false, "start_time": "2020-12-06T04:09:57.505185", "status": "completed"} tags=[]
output_file = Path(
    RESULTS_DIR,
    'ensemble_coassoc_matrix.npy'
).resolve()
display(output_file)

# %% papermill={"duration": 0.046263, "end_time": "2020-12-06T04:09:57.590143", "exception": false, "start_time": "2020-12-06T04:09:57.543880", "status": "completed"} tags=[]
ensemble_coassoc_matrix = np.load(output_file)

# %% papermill={"duration": 0.027492, "end_time": "2020-12-06T04:09:57.633231", "exception": false, "start_time": "2020-12-06T04:09:57.605739", "status": "completed"} tags=[]
ensemble_coassoc_matrix.shape

# %% papermill={"duration": 0.026987, "end_time": "2020-12-06T04:09:57.675038", "exception": false, "start_time": "2020-12-06T04:09:57.648051", "status": "completed"} tags=[]
ensemble_coassoc_matrix

# %% [markdown] papermill={"duration": 0.012931, "end_time": "2020-12-06T04:09:57.701961", "exception": false, "start_time": "2020-12-06T04:09:57.689030", "status": "completed"} tags=[]
# # Consensus clustering

# %% papermill={"duration": 0.152274, "end_time": "2020-12-06T04:09:57.867297", "exception": false, "start_time": "2020-12-06T04:09:57.715023", "status": "completed"} tags=[]
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from clustering.ensemble import supraconsensus, sc_consensus,\
    eac_single_coassoc_matrix, eac_complete_coassoc_matrix, eac_average_coassoc_matrix,\
    cspa, hgpa, mcla,\
    aami, run_method_and_compute_agreement

# %% papermill={"duration": 0.025944, "end_time": "2020-12-06T04:09:57.906791", "exception": false, "start_time": "2020-12-06T04:09:57.880847", "status": "completed"} tags=[]
consensus_graph_methods = (
    cspa, hgpa, mcla
)

# %% papermill={"duration": 0.026116, "end_time": "2020-12-06T04:09:57.946877", "exception": false, "start_time": "2020-12-06T04:09:57.920761", "status": "completed"} tags=[]
consensus_other_methods = (
    eac_single_coassoc_matrix,
    eac_complete_coassoc_matrix,
    eac_average_coassoc_matrix,
    sc_consensus,
)

# %% papermill={"duration": 0.027886, "end_time": "2020-12-06T04:09:57.988900", "exception": false, "start_time": "2020-12-06T04:09:57.961014", "status": "completed"} tags=[]
all_consensus_methods = set()
# all_consensus_methods.update(consensus_graph_methods)
all_consensus_methods.update(consensus_other_methods)
display(all_consensus_methods)

# %% papermill={"duration": 1991.770781, "end_time": "2020-12-06T04:43:09.773921", "exception": false, "start_time": "2020-12-06T04:09:58.003140", "status": "completed"} tags=[]
consensus_results = []

with ProcessPoolExecutor(max_workers=conf.GENERAL['N_JOBS']) as executor:
    tasks = {
        executor.submit(
            run_method_and_compute_agreement,
            m,
            ensemble_coassoc_matrix if m in consensus_other_methods else full_ensemble,
            full_ensemble,
            k
        ): (m.__name__, k)
        for m in all_consensus_methods
        for k in range(CLUSTERING_OPTIONS['K_MIN'], CLUSTERING_OPTIONS['K_MAX']+1)
    }

    for future in tqdm(as_completed(tasks), total=len(tasks), disable=False):
        method_name, k = tasks[future]
        part, performance_values = future.result()

        method_results = {
            'method': method_name,
            'partition': part,
            'k': k,
        }
        method_results.update(performance_values)
        
        consensus_results.append(method_results)

# %% papermill={"duration": 0.05419, "end_time": "2020-12-06T04:43:09.870056", "exception": false, "start_time": "2020-12-06T04:43:09.815866", "status": "completed"} tags=[]
# TODO: check if each partition is really generating k clusters

# %% papermill={"duration": 0.055427, "end_time": "2020-12-06T04:43:09.964973", "exception": false, "start_time": "2020-12-06T04:43:09.909546", "status": "completed"} tags=[]
consensus_results = pd.DataFrame(consensus_results)

# %% papermill={"duration": 0.052198, "end_time": "2020-12-06T04:43:10.057591", "exception": false, "start_time": "2020-12-06T04:43:10.005393", "status": "completed"} tags=[]
consensus_results.shape

# %% papermill={"duration": 0.066466, "end_time": "2020-12-06T04:43:10.165033", "exception": false, "start_time": "2020-12-06T04:43:10.098567", "status": "completed"} tags=[]
consensus_results.head()

# %% [markdown] papermill={"duration": 0.038781, "end_time": "2020-12-06T04:43:10.243414", "exception": false, "start_time": "2020-12-06T04:43:10.204633", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.053805, "end_time": "2020-12-06T04:43:10.336682", "exception": false, "start_time": "2020-12-06T04:43:10.282877", "status": "completed"} tags=[]
output_file = Path(
    RESULTS_DIR,
    'consensus_clustering_runs.pkl'
).resolve()
display(output_file)

# %% papermill={"duration": 0.05981, "end_time": "2020-12-06T04:43:10.436603", "exception": false, "start_time": "2020-12-06T04:43:10.376793", "status": "completed"} tags=[]
consensus_results.to_pickle(output_file)

# %% papermill={"duration": 0.039764, "end_time": "2020-12-06T04:43:10.516327", "exception": false, "start_time": "2020-12-06T04:43:10.476563", "status": "completed"} tags=[]
