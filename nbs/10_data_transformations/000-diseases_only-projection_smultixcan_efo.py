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
# It selects diseases only from traits.

# %% [markdown]
# # Modules loading

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from pathlib import Path
from IPython.display import display

import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns

import conf
from utils import generate_result_set_name
from data.cache import read_data

# %% [markdown]
# # Settings

# %%
INPUT_FILEPATH = Path(
    conf.RESULTS['PROJECTIONS_DIR'],
    'projection-smultixcan-efo_partial-mashr-zscores.pkl'
).resolve()
display(INPUT_FILEPATH)

input_filepath_stem = INPUT_FILEPATH.stem
display(input_filepath_stem)

# %%
# output dir for this notebook
RESULTS_DIR = Path(
    conf.RESULTS["DATA_TRANSFORMATIONS_DIR"],
    'traits_selections'
).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %% [markdown]
# # Load input file

# %%
data = pd.read_pickle(INPUT_FILEPATH).T

# %%
data.shape

# %%
data.head()

# %% [markdown]
# # Select diseases only

# %%
input_file = conf.PHENOMEXCAN["TRAITS_FULLCODE_TO_EFO_MAP_FILE"]
display(input_file)

# %%
ukb_to_efo_map = read_data(input_file)

# %%
ukb_to_efo_map.shape

# %%
ukb_to_efo_map.head()

# %%
efo_diseases = ukb_to_efo_map[ukb_to_efo_map['category'] == 'disease']['current_term_label'].unique()

# %%
efo_diseases.shape

# %%
data = data.loc[efo_diseases]

# %%
data.shape

# %%
data.head()

# %%
assert not data.isna().any().any()

# %% [markdown]
# # Save

# %%
output_file = Path(
    RESULTS_DIR,
    f'diseases_only-{input_filepath_stem}.pkl',
).resolve()

display(output_file)

# %%
data.to_pickle(output_file)

# %%
