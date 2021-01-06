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
# It projects input data into a UMAP representation.

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
from pathlib import Path
from IPython.display import display

import pandas as pd

import conf
from utils import generate_result_set_name

# %% [markdown] tags=[]
# # Settings

# %% [markdown] tags=[]
# ## Input data

# %% tags=[]
INPUT_FILEPATH_STEM = "projection-smultixcan-efo_partial-mashr-zscores"

# %% tags=[]
INPUT_FILEPATH = Path(
    conf.RESULTS["DATA_TRANSFORMATIONS_DIR"],
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
RESULTS_DIR = Path(conf.RESULTS["DATA_TRANSFORMATIONS_DIR"], "umap").resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %% [markdown] tags=[]
# ## UMAP options

# %% tags=[]
# parameters of the dimentionality reduction steps
# note that these are the default parameters of UMAP (metric and n_neighbors)
DR_OPTIONS = {
    "n_components": [5, 10, 20, 30, 40, 50],
    "metric": "euclidean",
    "n_neighbors": 15,
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
# # UMAP

# %% [markdown] tags=[]
# **Reproducibility problem**: there seems to be a bug with UMAP in which it produces different results in different operating systems or machines: https://github.com/lmcinnes/umap/issues/153

# %% tags=[]
from data.dimreduction import get_umap_proj

# %% tags=[]
# Get a UMAP representation for all n_components configurations
for n_comp in DR_OPTIONS["n_components"]:
    print(f"# components: {n_comp}")

    # prepare options of n_comp
    options = ALL_OPTIONS.copy()
    options["n_components"] = n_comp
    options = {k: v for k, v in options.items() if k in DR_OPTIONS}

    # get projection
    dr_data = get_umap_proj(data, options)

    # check data dimensionality
    display(dr_data.shape)
    assert dr_data.shape == (data.shape[0], n_comp)

    display(dr_data.iloc[:, 0:5].describe())

    # save
    output_file = Path(
        RESULTS_DIR,
        generate_result_set_name(
            options, prefix=f"umap-{input_filepath_stem}-", suffix=".pkl"
        ),
    ).resolve()
    display(output_file)

    dr_data.to_pickle(output_file)

    print("\n")

# %% [markdown] tags=[] papermill={"duration": 0.010926, "end_time": "2020-11-30T18:31:25.610594", "exception": false, "start_time": "2020-11-30T18:31:25.599668", "status": "completed"}
# ## Plots

# %%
import seaborn as sns

# %% [markdown]
# Plot the data from the UMAP version with 5 components.

# %%
# prepare options of 5 components
options = ALL_OPTIONS.copy()
options["n_components"] = 5
options = {k: v for k, v in options.items() if k in DR_OPTIONS}

# load
input_file = Path(
    RESULTS_DIR,
    generate_result_set_name(
        options, prefix=f"umap-{input_filepath_stem}-", suffix=".pkl"
    ),
).resolve()

dr_data = pd.read_pickle(input_file)

# %% [markdown]
# ## Full plot

# %% tags=[] papermill={"duration": 5.586547, "end_time": "2020-11-30T18:31:31.208070", "exception": false, "start_time": "2020-11-30T18:31:25.621523", "status": "completed"}
g = sns.pairplot(data=dr_data)

# %% [markdown]
# ## Plot without "outliers"

# %%
# remove "outliers" just to take a look at the big cluster
dr_data_thin = dr_data[(dr_data["UMAP1"] < -1)]

# %%
g = sns.PairGrid(data=dr_data_thin)
g.map_upper(sns.histplot)
g.map_lower(sns.kdeplot, fill=False)
g.map_diag(sns.histplot, kde=True)

# %% tags=[] papermill={"duration": 0.015327, "end_time": "2020-11-30T18:33:30.644774", "exception": false, "start_time": "2020-11-30T18:33:30.629447", "status": "completed"}
