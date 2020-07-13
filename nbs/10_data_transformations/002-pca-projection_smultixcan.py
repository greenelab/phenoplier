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
# It gets the PCA transformation of an input file.

# %% [markdown]
# # Modules loading

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from pathlib import Path
from IPython.display import display

import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

import conf
from utils import generate_result_set_name

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
# number of components to use in the dimensionality reduction step
DR_OPTIONS = {
    'n_components': 50,
    'svd_solver': 'full',
    'random_state': 0,
}

# %%
# output dir for this notebook
RESULTS_DIR = Path(
    conf.RESULTS["DATA_TRANSFORMATIONS_DIR"],
    'pca'
).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %%
# dictionary containing all options/settings (used to generate filenames)
ALL_OPTIONS = DR_OPTIONS.copy()
# ALL_OPTIONS['proj_percentile'] = PERCENTILE_NAME

display(ALL_OPTIONS)

# %% [markdown]
# # Load input file

# %%
data = pd.read_pickle(INPUT_FILEPATH).T

# %%
data.shape

# %%
data.head()

# %% [markdown]
# # PCA

# %%
dr_obj = PCA(**DR_OPTIONS)
display(dr_obj)

# %%
dr_obj = dr_obj.fit(data)

# %%
dr_data = dr_obj.transform(data)

# %%
dr_data = pd.DataFrame(
    data=dr_data,
    index=data.index.copy(),
    columns=[f'PCA{i+1}' for i in range(dr_data.shape[1])]
)

# %%
dr_data.shape

# %%
dr_data.head()

# %%
g = sns.pairplot(data=dr_data.iloc[:,:5])

# %% [markdown]
# # Save

# %%
output_file = Path(
    RESULTS_DIR,
    generate_result_set_name(
        ALL_OPTIONS,
        prefix=f'pca-{input_filepath_stem}-',
        suffix='.pkl'
    )
).resolve()

display(output_file)

# %%
dr_data.to_pickle(output_file)

# %%
