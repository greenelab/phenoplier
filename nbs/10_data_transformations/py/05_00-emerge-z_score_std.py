# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
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
# It standardize (z-score) eMERGE S-MultiXcan results projected into the MultiPLIER latent space.

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
from sklearn.preprocessing import scale

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
# output dir for this notebook
RESULTS_DIR = Path(conf.RESULTS["DATA_TRANSFORMATIONS_DIR"], "z_score_std").resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %% [markdown]
# # Load eMERGE data

# %% [markdown]
# ## Projection of S-MultiXcan z-scores

# %% tags=[]
input_file = Path(
    conf.RESULTS["PROJECTIONS_DIR"],
    "projection-emerge-smultixcan-mashr-zscores.pkl",
).resolve()
display(input_file)
assert input_file.exists()

input_filepath_stem = input_file.stem
display(input_filepath_stem)

# %%
pmbb_data = pd.read_pickle(input_file)

# %%
pmbb_data.shape

# %%
pmbb_data.head()

# %% [markdown] tags=[]
# # z-score standardization

# %%
data = pmbb_data.T

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
        np.isclose(scaled_data_stats.loc["std", c], 1.0, atol=1e-02)
        for c in scaled_data_stats.columns
    ]
)

# %% [markdown]
# # Save

# %% tags=[]
output_file = Path(
    RESULTS_DIR,
    f"z_score_std-{input_filepath_stem}.pkl",
).resolve()

display(output_file)

# %% tags=[]
scaled_data.to_pickle(output_file)

# %%
