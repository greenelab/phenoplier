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
# It reads all the clustering partitions obtained with different algorithms, and generates the ensemble by putting it into a numpy array. Then it obtains the coassociation matrix from the ensemble (see more details below).

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

# %% tags=[] trusted=true
# %load_ext autoreload
# %autoreload 2

# %% tags=[] trusted=true
from pathlib import Path

import numpy as np
import pandas as pd

# %% [markdown] tags=[]
# # Settings

# %% tags=[] trusted=true
RANDOM_GENERATOR = np.random.default_rng(12345)

# %% [markdown] tags=[]
# ## Ensemble size

# %% [markdown] tags=[]
# For some clustering algorithms it is easy to control the number of final partitions to generate: for instance, for k-means, you can generate partitions from k=2 to k=20 (19 partitions with different number of clusters). However, with algorithms such as DBSCAN this is not very easy to achieve, since for some parameter combinations (`eps` and `min_samples`) it generates partitions with one cluster (which is not an actual partition of the data) that are not included here.
#
# The parameters below specify the expected number of partitions for each clustering algorithm, and a range of allowed sizes. Then, the code below checks that each algorithm has the same representation in the ensemble. For example, if `EXPECTED_ENSEMBLE_SIZE=50`, `MIN_ENSEMBLE_SIZE=45` and `MAX_ENSEMBLE_SIZE=55`, the code below will check that k-means, spectral clustering, DBSCAN, etc, generated between 45 and 55 partitions. If not, it resamples the generated partitions to get 50 (the value specified by `EXPECTED_ENSEMBLE_SIZE`), so each algorithm has approximately the same representation in the full ensemble.

# %% tags=[] trusted=true
EXPECTED_ENSEMBLE_SIZE = 295

MIN_ENSEMBLE_SIZE = 290
MAX_ENSEMBLE_SIZE = 300

# %% [markdown] tags=[]
# ## Consensus clustering

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

# %% tags=[] trusted=true
included_pkl_files = []

for pkl_file in input_dir.rglob("*.pkl"):
    pkl_file_str = str(pkl_file)

    # skip stability pkl files
    if "-stability-" in pkl_file_str:
        continue

    included_pkl_files.append(pkl_file)

# %% tags=[]
display(len(included_pkl_files))

# 5 algorithms, 3 dataset versions
assert len(included_pkl_files) == 5 * 3

# %% [markdown] tags=[]
# ## Combine partition files to get final ensemble

# %% tags=[] trusted=true
n_partitions = 0

# %% tags=[] trusted=true
ensembles_list = []

# %% tags=[]
for ens_file in included_pkl_files:
    ens = pd.read_pickle(ens_file)

    short_file_path = Path(*ens_file.parts[-2:])

    if ens.shape[0] < MIN_ENSEMBLE_SIZE:
        print(f"Less partitions than expected in {short_file_path}")

        # if less partitions than expected, resample with replacement
        ens = ens.sample(
            n=EXPECTED_ENSEMBLE_SIZE,
            replace=True,
            random_state=RANDOM_GENERATOR.bit_generator,
        )
        assert ens.shape[0] == EXPECTED_ENSEMBLE_SIZE

    elif ens.shape[0] > MAX_ENSEMBLE_SIZE:
        print(f"More partitions than expected in {short_file_path}")

        # if more partitions than expected, take a smaller sample
        ens = ens.sample(
            n=EXPECTED_ENSEMBLE_SIZE, random_state=RANDOM_GENERATOR.bit_generator
        )
        assert ens.shape[0] == EXPECTED_ENSEMBLE_SIZE

    ens_full_format = np.concatenate(
        ens["partition"].apply(lambda x: x.reshape(1, -1)), axis=0
    )

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

# %% tags=[] trusted=true
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

# %% tags=[] trusted=true
np.save(output_file, full_ensemble)

# %% [markdown] tags=[]
# # Get coassociation matrix from ensemble

# %% [markdown] tags=[]
# The coassociation matrix is a distance matrix derived from the ensemble, where each cell represents the percentage of times a pair of objects (traits and diseases in this case) were not clustered together. It serves as an input for any consensus function (basically, another clustering algorithm) to derive a consensus partition.

# %% tags=[] trusted=true
from clustering.ensembles.utils import get_ensemble_distance_matrix

# %% tags=[] trusted=true
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

# %% tags=[] trusted=true
np.save(output_file, ensemble_coassoc_matrix)

# %% tags=[] trusted=true
