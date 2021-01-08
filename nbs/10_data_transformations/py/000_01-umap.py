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

# %% [markdown] papermill={"duration": 0.010137, "end_time": "2020-11-30T18:31:33.121838", "exception": false, "start_time": "2020-11-30T18:31:33.111701", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.006066, "end_time": "2020-11-30T18:31:33.134218", "exception": false, "start_time": "2020-11-30T18:31:33.128152", "status": "completed"} tags=[]
# It projects input data into a UMAP representation.

# %% [markdown] papermill={"duration": 0.006102, "end_time": "2020-11-30T18:31:33.146515", "exception": false, "start_time": "2020-11-30T18:31:33.140413", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.017413, "end_time": "2020-11-30T18:31:33.170086", "exception": false, "start_time": "2020-11-30T18:31:33.152673", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 0.57397, "end_time": "2020-11-30T18:31:33.750950", "exception": false, "start_time": "2020-11-30T18:31:33.176980", "status": "completed"} tags=[]
from pathlib import Path
from IPython.display import display

import pandas as pd

import conf
from utils import generate_result_set_name

# %% [markdown] papermill={"duration": 0.008085, "end_time": "2020-11-30T18:31:33.767559", "exception": false, "start_time": "2020-11-30T18:31:33.759474", "status": "completed"} tags=[]
# # Settings

# %% [markdown] papermill={"duration": 0.007456, "end_time": "2020-11-30T18:31:33.783918", "exception": false, "start_time": "2020-11-30T18:31:33.776462", "status": "completed"} tags=[]
# ## Input data

# %% papermill={"duration": 0.019311, "end_time": "2020-11-30T18:31:33.810267", "exception": false, "start_time": "2020-11-30T18:31:33.790956", "status": "completed"} tags=[]
INPUT_FILEPATH_STEM = "projection-smultixcan-efo_partial-mashr-zscores"

# %% papermill={"duration": 0.022739, "end_time": "2020-11-30T18:31:33.840220", "exception": false, "start_time": "2020-11-30T18:31:33.817481", "status": "completed"} tags=[]
INPUT_FILEPATH = Path(
    conf.RESULTS["DATA_TRANSFORMATIONS_DIR"],
    "z_score_std",
    f"z_score_std-{INPUT_FILEPATH_STEM}.pkl",
).resolve()
display(INPUT_FILEPATH)

input_filepath_stem = INPUT_FILEPATH.stem
display(input_filepath_stem)

# %% [markdown] papermill={"duration": 0.008102, "end_time": "2020-11-30T18:31:33.856013", "exception": false, "start_time": "2020-11-30T18:31:33.847911", "status": "completed"} tags=[]
# ## Output folder

# %% papermill={"duration": 0.02057, "end_time": "2020-11-30T18:31:33.884349", "exception": false, "start_time": "2020-11-30T18:31:33.863779", "status": "completed"} tags=[]
# output dir for this notebook
RESULTS_DIR = Path(conf.RESULTS["DATA_TRANSFORMATIONS_DIR"], "umap").resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %% [markdown] papermill={"duration": 0.007595, "end_time": "2020-11-30T18:31:33.899752", "exception": false, "start_time": "2020-11-30T18:31:33.892157", "status": "completed"} tags=[]
# ## UMAP options

# %% papermill={"duration": 0.020365, "end_time": "2020-11-30T18:31:33.927880", "exception": false, "start_time": "2020-11-30T18:31:33.907515", "status": "completed"} tags=[]
# parameters of the dimentionality reduction steps
# note that these are the default parameters of UMAP (metric and n_neighbors)
DR_OPTIONS = {
    "n_components": [5, 10, 20, 30, 40, 50],
    "metric": "euclidean",
    "n_neighbors": 15,
    "random_state": 0,
}

# %% papermill={"duration": 0.020501, "end_time": "2020-11-30T18:31:33.956090", "exception": false, "start_time": "2020-11-30T18:31:33.935589", "status": "completed"} tags=[]
# dictionary containing all options/settings (used to generate filenames)
ALL_OPTIONS = DR_OPTIONS.copy()

display(ALL_OPTIONS)

# %% [markdown] papermill={"duration": 0.007431, "end_time": "2020-11-30T18:31:33.971366", "exception": false, "start_time": "2020-11-30T18:31:33.963935", "status": "completed"} tags=[]
# # Load input file

# %% papermill={"duration": 0.032456, "end_time": "2020-11-30T18:31:34.011262", "exception": false, "start_time": "2020-11-30T18:31:33.978806", "status": "completed"} tags=[]
data = pd.read_pickle(INPUT_FILEPATH)

# %% papermill={"duration": 0.021387, "end_time": "2020-11-30T18:31:34.041193", "exception": false, "start_time": "2020-11-30T18:31:34.019806", "status": "completed"} tags=[]
display(data.shape)

# %% papermill={"duration": 0.035424, "end_time": "2020-11-30T18:31:34.085538", "exception": false, "start_time": "2020-11-30T18:31:34.050114", "status": "completed"} tags=[]
display(data.head())

# %% [markdown] papermill={"duration": 0.009136, "end_time": "2020-11-30T18:31:34.104038", "exception": false, "start_time": "2020-11-30T18:31:34.094902", "status": "completed"} tags=[]
# # UMAP

# %% [markdown] papermill={"duration": 0.008271, "end_time": "2020-11-30T18:31:34.121574", "exception": false, "start_time": "2020-11-30T18:31:34.113303", "status": "completed"} tags=[]
# **Reproducibility problem**: there seems to be a bug with UMAP in which it produces different results in different operating systems or machines: https://github.com/lmcinnes/umap/issues/153

# %% papermill={"duration": 0.020807, "end_time": "2020-11-30T18:31:34.150762", "exception": false, "start_time": "2020-11-30T18:31:34.129955", "status": "completed"} tags=[]
from data.dimreduction import get_umap_proj

# %% papermill={"duration": 116.453553, "end_time": "2020-11-30T18:33:30.613431", "exception": false, "start_time": "2020-11-30T18:31:34.159878", "status": "completed"} tags=[]
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
