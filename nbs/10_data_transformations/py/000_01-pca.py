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

# %% [markdown] papermill={"duration": 0.011683, "end_time": "2020-11-30T18:31:23.723335", "exception": false, "start_time": "2020-11-30T18:31:23.711652", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.007375, "end_time": "2020-11-30T18:31:23.738363", "exception": false, "start_time": "2020-11-30T18:31:23.730988", "status": "completed"} tags=[]
# It gets the PCA transformation of an input file.

# %% [markdown] papermill={"duration": 0.007433, "end_time": "2020-11-30T18:31:23.753677", "exception": false, "start_time": "2020-11-30T18:31:23.746244", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.018492, "end_time": "2020-11-30T18:31:23.779689", "exception": false, "start_time": "2020-11-30T18:31:23.761197", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 0.584415, "end_time": "2020-11-30T18:31:24.372370", "exception": false, "start_time": "2020-11-30T18:31:23.787955", "status": "completed"} tags=[]
from pathlib import Path
from IPython.display import display

import pandas as pd
import seaborn as sns

import conf
from utils import generate_result_set_name

# %% [markdown] papermill={"duration": 0.009842, "end_time": "2020-11-30T18:31:24.392652", "exception": false, "start_time": "2020-11-30T18:31:24.382810", "status": "completed"} tags=[]
# # Settings

# %% [markdown] papermill={"duration": 0.00862, "end_time": "2020-11-30T18:31:24.410732", "exception": false, "start_time": "2020-11-30T18:31:24.402112", "status": "completed"} tags=[]
# ## Input data

# %% papermill={"duration": 0.020189, "end_time": "2020-11-30T18:31:24.439057", "exception": false, "start_time": "2020-11-30T18:31:24.418868", "status": "completed"} tags=[]
INPUT_FILEPATH_STEM = "projection-smultixcan-efo_partial-mashr-zscores"

# %% papermill={"duration": 0.023337, "end_time": "2020-11-30T18:31:24.470745", "exception": false, "start_time": "2020-11-30T18:31:24.447408", "status": "completed"} tags=[]
INPUT_FILEPATH = Path(
    conf.RESULTS["DATA_TRANSFORMATIONS_DIR"],
    "z_score_std",
    f"z_score_std-{INPUT_FILEPATH_STEM}.pkl",
).resolve()
display(INPUT_FILEPATH)

input_filepath_stem = INPUT_FILEPATH.stem
display(input_filepath_stem)

# %% [markdown] papermill={"duration": 0.008226, "end_time": "2020-11-30T18:31:24.488089", "exception": false, "start_time": "2020-11-30T18:31:24.479863", "status": "completed"} tags=[]
# ## Output folder

# %% papermill={"duration": 0.021432, "end_time": "2020-11-30T18:31:24.517997", "exception": false, "start_time": "2020-11-30T18:31:24.496565", "status": "completed"} tags=[]
# output dir for this notebook
RESULTS_DIR = Path(conf.RESULTS["DATA_TRANSFORMATIONS_DIR"], "pca").resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %% [markdown] papermill={"duration": 0.0088, "end_time": "2020-11-30T18:31:24.535866", "exception": false, "start_time": "2020-11-30T18:31:24.527066", "status": "completed"} tags=[]
# ## PCA options

# %% papermill={"duration": 0.021201, "end_time": "2020-11-30T18:31:24.565989", "exception": false, "start_time": "2020-11-30T18:31:24.544788", "status": "completed"} tags=[]
# parameters of the dimentionality reduction steps
DR_OPTIONS = {
    "n_components": 50,
    "svd_solver": "full",
    "random_state": 0,
}

# %% papermill={"duration": 0.021208, "end_time": "2020-11-30T18:31:24.596221", "exception": false, "start_time": "2020-11-30T18:31:24.575013", "status": "completed"} tags=[]
# dictionary containing all options/settings (used to generate filenames)
ALL_OPTIONS = DR_OPTIONS.copy()

display(ALL_OPTIONS)

# %% [markdown] papermill={"duration": 0.00903, "end_time": "2020-11-30T18:31:24.614354", "exception": false, "start_time": "2020-11-30T18:31:24.605324", "status": "completed"} tags=[]
# # Load input file

# %% papermill={"duration": 0.033326, "end_time": "2020-11-30T18:31:24.656646", "exception": false, "start_time": "2020-11-30T18:31:24.623320", "status": "completed"} tags=[]
data = pd.read_pickle(INPUT_FILEPATH)

# %% papermill={"duration": 0.023498, "end_time": "2020-11-30T18:31:24.690475", "exception": false, "start_time": "2020-11-30T18:31:24.666977", "status": "completed"} tags=[]
display(data.shape)

# %% papermill={"duration": 0.037456, "end_time": "2020-11-30T18:31:24.738706", "exception": false, "start_time": "2020-11-30T18:31:24.701250", "status": "completed"} tags=[]
display(data.head())

# %% [markdown] papermill={"duration": 0.010934, "end_time": "2020-11-30T18:31:24.760894", "exception": false, "start_time": "2020-11-30T18:31:24.749960", "status": "completed"} tags=[]
# # PCA

# %% papermill={"duration": 0.021928, "end_time": "2020-11-30T18:31:24.793479", "exception": false, "start_time": "2020-11-30T18:31:24.771551", "status": "completed"} tags=[]
from data.dimreduction import get_pca_proj

# %% papermill={"duration": 0.698385, "end_time": "2020-11-30T18:31:25.502170", "exception": false, "start_time": "2020-11-30T18:31:24.803785", "status": "completed"} tags=[]
dr_data = get_pca_proj(data, DR_OPTIONS)

# %% papermill={"duration": 0.02372, "end_time": "2020-11-30T18:31:25.537039", "exception": false, "start_time": "2020-11-30T18:31:25.513319", "status": "completed"} tags=[]
display(dr_data.shape)

# %% papermill={"duration": 0.039962, "end_time": "2020-11-30T18:31:25.588307", "exception": false, "start_time": "2020-11-30T18:31:25.548345", "status": "completed"} tags=[]
display(dr_data.head())

# %% [markdown] papermill={"duration": 0.010926, "end_time": "2020-11-30T18:31:25.610594", "exception": false, "start_time": "2020-11-30T18:31:25.599668", "status": "completed"} tags=[]
# ## Plot

# %% papermill={"duration": 5.586547, "end_time": "2020-11-30T18:31:31.208070", "exception": false, "start_time": "2020-11-30T18:31:25.621523", "status": "completed"} tags=[]
g = sns.pairplot(data=dr_data.iloc[:, :5])

# %% [markdown] papermill={"duration": 0.016597, "end_time": "2020-11-30T18:31:31.243103", "exception": false, "start_time": "2020-11-30T18:31:31.226506", "status": "completed"} tags=[]
# # Save

# %% papermill={"duration": 0.028239, "end_time": "2020-11-30T18:31:31.286166", "exception": false, "start_time": "2020-11-30T18:31:31.257927", "status": "completed"} tags=[]
output_file = Path(
    RESULTS_DIR,
    generate_result_set_name(
        ALL_OPTIONS, prefix=f"pca-{input_filepath_stem}-", suffix=".pkl"
    ),
).resolve()

display(output_file)

# %% papermill={"duration": 0.029247, "end_time": "2020-11-30T18:31:31.330604", "exception": false, "start_time": "2020-11-30T18:31:31.301357", "status": "completed"} tags=[]
dr_data.to_pickle(output_file)

# %% papermill={"duration": 0.014751, "end_time": "2020-11-30T18:31:31.360662", "exception": false, "start_time": "2020-11-30T18:31:31.345911", "status": "completed"} tags=[]
