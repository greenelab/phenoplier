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
import warnings

import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns

from utils import generate_result_set_name

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
INITIAL_RANDOM_STATE = 12345

# %% [markdown] tags=[]
# ## Output directory

# %% tags=[]
# # output dir for this notebook
# RESULTS_DIR = Path(
#     conf.RESULTS["CLUSTERING_RUNS_DIR"],
#     f"{INPUT_SUBSET}-{INPUT_STEM}",
# ).resolve()
# RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# display(RESULTS_DIR)

# %% [markdown] tags=[]
# # Z-score standardized data

# %% tags=[]
INPUT_SUBSET = "z_score_std"

# %% tags=[]
INPUT_STEM = "projection-smultixcan-efo_partial-mashr-zscores"

# %% tags=[]
input_filepath = Path(
    conf.RESULTS["DATA_TRANSFORMATIONS_DIR"],
    INPUT_SUBSET,
    f"{INPUT_SUBSET}-{INPUT_STEM}.pkl",
).resolve()
display(input_filepath)

assert input_filepath.exists(), "Input file does not exist"

input_filepath_stem = input_filepath.stem
display(input_filepath_stem)

# %% tags=[]
data = pd.read_pickle(input_filepath)

# %% tags=[]
data.shape

# %% tags=[]
data.head()

# %% [markdown] tags=[]
# ## Data stats

# %% tags=[]
data.min().min(), data.max().max()

# %% tags=[]
assert not np.isinf(data).any().any()

# %% tags=[]
assert not data.isna().any().any()

# %% tags=[]
data_stats = data.describe()

# %% tags=[]
data_stats.T

# %% tags=[]
assert not np.isinf(data_stats).any().any()

# %% tags=[]
assert not data_stats.isna().any().any()

# %% [markdown] tags=[]
# ## Check duplicated values

# %% tags=[]
data_dups = data.duplicated()

# %% tags=[]
data_dups.any()

# %% tags=[]
data_dups.value_counts()

# %% tags=[]
data_dups_labels = data.index[data_dups]
display(data_dups_labels)

# %% [markdown] tags=[]
# # PCA

# %% tags=[]
INPUT_SUBSET = "pca"

# %% tags=[]
INPUT_STEM = "z_score_std-projection-smultixcan-efo_partial-mashr-zscores"

# %% tags=[]
DR_OPTIONS = {
    "n_components": 50,
    "svd_solver": "full",
    "random_state": 0,
}

# %% tags=[]
input_filepath = Path(
    conf.RESULTS["DATA_TRANSFORMATIONS_DIR"],
    INPUT_SUBSET,
    generate_result_set_name(
        DR_OPTIONS, prefix=f"{INPUT_SUBSET}-{INPUT_STEM}-", suffix=".pkl"
    ),
).resolve()
display(input_filepath)

assert input_filepath.exists(), "Input file does not exist"

input_filepath_stem = input_filepath.stem
display(input_filepath_stem)

# %% tags=[]
data = pd.read_pickle(input_filepath)

# %% tags=[]
data.shape

# %% tags=[]
data.head()

# %% [markdown] tags=[]
# ## Data stats

# %% tags=[]
data.min().min(), data.max().max()

# %% tags=[]
assert not np.isinf(data).any().any()

# %% tags=[]
assert not data.isna().any().any()

# %% tags=[]
data_stats = data.describe()

# %% tags=[]
data_stats.T

# %% tags=[]
assert not np.isinf(data_stats).any().any()

# %% tags=[]
assert not data_stats.isna().any().any()

# %% [markdown] tags=[]
# ## Check duplicated values

# %% tags=[]
data_dups = data.duplicated()

# %% tags=[]
data_dups.any()

# %% tags=[]
data_dups.value_counts()

# %% tags=[]
data.index[data_dups]

# %% tags=[]
assert data.index[data_dups].equals(data_dups_labels)

# %% [markdown] tags=[]
# # UMAP

# %% tags=[]
INPUT_SUBSET = "umap"

# %% tags=[]
INPUT_STEM = "z_score_std-projection-smultixcan-efo_partial-mashr-zscores"

# %% tags=[]
DR_OPTIONS = {
    "n_components": 50,
    "metric": "euclidean",
    "n_neighbors": 15,
    "random_state": 0,
}

# %% tags=[]
input_filepath = Path(
    conf.RESULTS["DATA_TRANSFORMATIONS_DIR"],
    INPUT_SUBSET,
    generate_result_set_name(
        DR_OPTIONS, prefix=f"{INPUT_SUBSET}-{INPUT_STEM}-", suffix=".pkl"
    ),
).resolve()
display(input_filepath)

assert input_filepath.exists(), "Input file does not exist"

input_filepath_stem = input_filepath.stem
display(input_filepath_stem)

# %% tags=[]
data = pd.read_pickle(input_filepath)

# %% tags=[]
data.shape

# %% tags=[]
data.head()

# %% [markdown] tags=[]
# ## Data stats

# %% tags=[]
data.min().min(), data.max().max()

# %% tags=[]
assert not np.isinf(data).any().any()

# %% tags=[]
assert not data.isna().any().any()

# %% tags=[]
data_stats = data.describe()

# %% tags=[]
data_stats.T

# %% tags=[]
assert not np.isinf(data_stats).any().any()

# %% tags=[]
assert not data_stats.isna().any().any()

# %% [markdown] tags=[]
# ## Check duplicated values

# %% tags=[]
data_dups = data.duplicated()

# %% tags=[]
data_dups.any()

# %% [markdown] tags=[]
# There are no duplicates with UMAP data, but the duplicates in `z_score_std` and `pca` are very close by in the UMAP representation.

# %% tags=[]
data_dups_labels

# %% tags=[]
data.loc[data_dups_labels]

# %% tags=[]
