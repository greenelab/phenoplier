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
# It gets the PCA transformation of an input file.

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
from pathlib import Path
from IPython.display import display

import numpy as np
import pandas as pd
import seaborn as sns

import conf
from utils import generate_result_set_name

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
np.random.seed(0)

# %% [markdown] tags=[]
# ## Input data

# %% tags=[]
INPUT_FILEPATH_STEM = "projection-smultixcan-efo_partial-mashr-zscores"

# %% tags=[]
INPUT_FILEPATH = Path(
    conf.RESULTS["CLUSTERING_NULL_DIR"],
    "data_transformations",
    "z_score_std",
    f"z_score_std-{INPUT_FILEPATH_STEM}.pkl",
).resolve()
display(INPUT_FILEPATH)

input_filepath_stem = INPUT_FILEPATH.stem
display(input_filepath_stem)

# %% [markdown] tags=[]
# ## Output folder

# %% tags=[]
# output dir for this notebook
RESULTS_DIR = Path(
    conf.RESULTS["CLUSTERING_NULL_DIR"], "data_transformations", "pca"
).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %% [markdown] tags=[]
# ## PCA options

# %% tags=[]
# parameters of the dimentionality reduction steps
DR_OPTIONS = {
    "n_components": 50,
    "svd_solver": "full",
    "random_state": 0,
}

# %% tags=[]
# dictionary containing all options/settings (used to generate filenames)
ALL_OPTIONS = DR_OPTIONS.copy()

display(ALL_OPTIONS)

# %% [markdown] tags=[]
# # Load input file

# %% tags=[]
data = pd.read_pickle(INPUT_FILEPATH)

# %% tags=[]
display(data.shape)

# %% tags=[]
display(data.head())

# %% [markdown] tags=[]
# # PCA

# %% tags=[]
from data.dimreduction import get_pca_proj

# %% tags=[]
dr_data = get_pca_proj(data, DR_OPTIONS)

# %% tags=[]
display(dr_data.shape)

# %% tags=[]
display(dr_data.head())

# %% [markdown] tags=[]
# ## Plot

# %% tags=[]
g = sns.pairplot(data=dr_data.iloc[:, :5])

# %% [markdown] tags=[]
# # Save

# %% tags=[]
output_file = Path(
    RESULTS_DIR,
    generate_result_set_name(
        ALL_OPTIONS, prefix=f"pca-{input_filepath_stem}-", suffix=".pkl"
    ),
).resolve()

display(output_file)

# %% tags=[]
dr_data.to_pickle(output_file)

# %% tags=[]
