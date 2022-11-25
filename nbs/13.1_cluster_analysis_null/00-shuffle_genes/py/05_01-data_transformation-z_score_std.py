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
# It standardize (z-score) S-MultiXcan results projected into the MultiPLIER latent space.

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
from sklearn.preprocessing import scale

import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
np.random.seed(0)

# %% [markdown] tags=[]
# ## Input data

# %% tags=[]
INPUT_FILEPATH = Path(
    conf.RESULTS["CLUSTERING_NULL_DIR"],
    "projections",
    "projection-smultixcan-efo_partial-mashr-zscores.pkl",
).resolve()
display(INPUT_FILEPATH)

input_filepath_stem = INPUT_FILEPATH.stem
display(input_filepath_stem)

# %% [markdown] tags=[]
# ## Output folder

# %% tags=[]
# output dir for this notebook
RESULTS_DIR = Path(
    conf.RESULTS["CLUSTERING_NULL_DIR"], "data_transformations", "z_score_std"
).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %% [markdown] tags=[]
# # Load input file

# %% tags=[]
data = pd.read_pickle(INPUT_FILEPATH).T

# %% tags=[]
display(data.shape)

# %% tags=[]
display(data.head())

# %% [markdown] tags=[]
# # z-score standardization

# %% tags=[]
data_stats = data.iloc[:, :10].describe()
display(data_stats)

# %% tags=[]
scaled_data = pd.DataFrame(
    data=scale(data), index=data.index.copy(), columns=data.columns.copy()
)

# %% tags=[]
display(scaled_data.shape)

# %% tags=[]
display(scaled_data.head())

# %% tags=[]
scaled_data_stats = scaled_data.iloc[:, :10].describe()
display(scaled_data_stats)

# %% [markdown] tags=[]
# ## Testing

# %% tags=[]
assert np.all(
    [
        np.isclose(scaled_data_stats.loc["mean", c], 0.0)
        for c in scaled_data_stats.columns
    ]
)

# %% tags=[]
assert np.all(
    [
        np.isclose(scaled_data_stats.loc["std", c], 1.0, atol=1e-03)
        for c in scaled_data_stats.columns
    ]
)

# %% [markdown] tags=[]
# # Save

# %% tags=[]
output_file = Path(
    RESULTS_DIR,
    f"z_score_std-{input_filepath_stem}.pkl",
).resolve()

display(output_file)

# %% tags=[]
scaled_data.to_pickle(output_file)

# %% tags=[]
