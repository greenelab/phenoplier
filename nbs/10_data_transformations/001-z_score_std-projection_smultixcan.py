# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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

# %% [markdown]
# # Description

# %% [markdown]
# It standardizes the features (latent variables) of an input file.

# %% [markdown]
# # Modules loading

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from pathlib import Path
from IPython.display import display

import numpy as np
import pandas as pd
from sklearn.preprocessing import scale

import conf

# %% [markdown]
# # Settings

# %%
INPUT_FILEPATH = Path(
    conf.RESULTS['PROJECTIONS_DIR'],
    'projection-smultixcan-mashr-zscores.pkl'
).resolve()
display(INPUT_FILEPATH)

input_filepath_stem = INPUT_FILEPATH.stem
display(input_filepath_stem)

# %%
# output dir for this notebook
RESULTS_DIR = Path(
    conf.RESULTS["DATA_TRANSFORMATIONS_DIR"],
    'z_score_std'
).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %% [markdown]
# # Load input file

# %%
data = pd.read_pickle(INPUT_FILEPATH)

# %%
data.shape

# %%
data.head()

# %% [markdown]
# # Data preprocessing for clustering

# %%
data_stats = data.T.iloc[:, :10].describe()
display(data_stats)

# %% [markdown]
# ## Standardize

# %%
data_t = data.T

scaled_data = pd.DataFrame(
    data=scale(data_t),
    index=data_t.index.copy(),
    columns=data_t.columns.copy()
)

# %%
scaled_data.shape

# %%
scaled_data.head()

# %%
scaled_data_stats = scaled_data.iloc[:,:10].describe()
display(scaled_data_stats)

# %%
assert np.all([np.isclose(scaled_data_stats.loc['mean', c], 0.0) for c in scaled_data_stats.columns])

# %%
assert np.all([np.isclose(scaled_data_stats.loc['std', c], 1.0, atol=1e-03) for c in scaled_data_stats.columns])

# %% [markdown]
# # Save

# %%
output_file = Path(
    RESULTS_DIR,
    f'z_score_standardized-{input_filepath_stem}.pkl',
).resolve()

display(output_file)

# %%
scaled_data.to_pickle(output_file)

# %%
