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

# %% [markdown] papermill={"duration": 0.012236, "end_time": "2020-12-11T20:14:35.703942", "exception": false, "start_time": "2020-12-11T20:14:35.691706", "status": "completed"} tags=[]
# # Description

# %% [markdown]
# It projects the PhenomeXcan results into the MultiPLIER latent space.

# %% [markdown] papermill={"duration": 0.011233, "end_time": "2020-12-11T20:14:35.746265", "exception": false, "start_time": "2020-12-11T20:14:35.735032", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.021981, "end_time": "2020-12-11T20:14:35.778647", "exception": false, "start_time": "2020-12-11T20:14:35.756666", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 0.297222, "end_time": "2020-12-11T20:14:36.087015", "exception": false, "start_time": "2020-12-11T20:14:35.789793", "status": "completed"} tags=[]
from pathlib import Path

from IPython.display import display
import pandas as pd

import conf
from data.cache import read_data
from multiplier import MultiplierProjection

# %% [markdown] papermill={"duration": 0.018339, "end_time": "2020-12-11T20:14:36.124493", "exception": false, "start_time": "2020-12-11T20:14:36.106154", "status": "completed"} tags=[]
# # Settings

# %% papermill={"duration": 0.025769, "end_time": "2020-12-11T20:14:36.162744", "exception": false, "start_time": "2020-12-11T20:14:36.136975", "status": "completed"} tags=[]
RESULTS_PROJ_OUTPUT_DIR = Path(conf.RESULTS["PROJECTIONS_DIR"])

RESULTS_PROJ_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_PROJ_OUTPUT_DIR)

# %% [markdown] papermill={"duration": 0.010915, "end_time": "2020-12-11T20:14:36.185218", "exception": false, "start_time": "2020-12-11T20:14:36.174303", "status": "completed"} tags=[]
# # Read gene mappings

# %% papermill={"duration": 0.050299, "end_time": "2020-12-11T20:14:36.246450", "exception": false, "start_time": "2020-12-11T20:14:36.196151", "status": "completed"} tags=[]
GENE_ID_TO_NAME_MAP = read_data(conf.PHENOMEXCAN["GENE_MAP_ID_TO_NAME"])
GENE_NAME_TO_ID_MAP = read_data(conf.PHENOMEXCAN["GENE_MAP_NAME_TO_ID"])

# %% [markdown] papermill={"duration": 0.011497, "end_time": "2020-12-11T20:14:36.270117", "exception": false, "start_time": "2020-12-11T20:14:36.258620", "status": "completed"} tags=[]
# # Load PhenomeXcan data (S-MultiXcan)

# %% papermill={"duration": 0.021511, "end_time": "2020-12-11T20:14:36.302451", "exception": false, "start_time": "2020-12-11T20:14:36.280940", "status": "completed"} tags=[]
smultixcan_results_filename = conf.PHENOMEXCAN["SMULTIXCAN_MASHR_ZSCORES_FILE"]

display(smultixcan_results_filename)

# %% papermill={"duration": 0.021438, "end_time": "2020-12-11T20:14:36.335405", "exception": false, "start_time": "2020-12-11T20:14:36.313967", "status": "completed"} tags=[]
results_filename_stem = smultixcan_results_filename.stem
display(results_filename_stem)

# %% papermill={"duration": 3.204958, "end_time": "2020-12-11T20:14:39.552338", "exception": false, "start_time": "2020-12-11T20:14:36.347380", "status": "completed"} tags=[]
smultixcan_results = pd.read_pickle(smultixcan_results_filename)

# %% papermill={"duration": 0.025563, "end_time": "2020-12-11T20:14:39.617872", "exception": false, "start_time": "2020-12-11T20:14:39.592309", "status": "completed"} tags=[]
smultixcan_results.shape

# %% papermill={"duration": 0.036748, "end_time": "2020-12-11T20:14:39.666881", "exception": false, "start_time": "2020-12-11T20:14:39.630133", "status": "completed"} tags=[]
smultixcan_results.head()

# %% [markdown] papermill={"duration": 0.012846, "end_time": "2020-12-11T20:14:39.693058", "exception": false, "start_time": "2020-12-11T20:14:39.680212", "status": "completed"} tags=[]
# ## Gene IDs to Gene names

# %% papermill={"duration": 0.363321, "end_time": "2020-12-11T20:14:40.069084", "exception": false, "start_time": "2020-12-11T20:14:39.705763", "status": "completed"} tags=[]
smultixcan_results = smultixcan_results.rename(index=GENE_ID_TO_NAME_MAP)

# %% papermill={"duration": 0.025159, "end_time": "2020-12-11T20:14:40.111439", "exception": false, "start_time": "2020-12-11T20:14:40.086280", "status": "completed"} tags=[]
smultixcan_results.shape

# %% papermill={"duration": 0.037808, "end_time": "2020-12-11T20:14:40.163197", "exception": false, "start_time": "2020-12-11T20:14:40.125389", "status": "completed"} tags=[]
smultixcan_results.head()

# %% [markdown] papermill={"duration": 0.013698, "end_time": "2020-12-11T20:14:40.191604", "exception": false, "start_time": "2020-12-11T20:14:40.177906", "status": "completed"} tags=[]
# ## Remove duplicated gene entries

# %% papermill={"duration": 0.026447, "end_time": "2020-12-11T20:14:40.231569", "exception": false, "start_time": "2020-12-11T20:14:40.205122", "status": "completed"} tags=[]
smultixcan_results.index[smultixcan_results.index.duplicated(keep="first")]

# %% papermill={"duration": 0.433325, "end_time": "2020-12-11T20:14:40.678778", "exception": false, "start_time": "2020-12-11T20:14:40.245453", "status": "completed"} tags=[]
smultixcan_results = smultixcan_results.loc[
    ~smultixcan_results.index.duplicated(keep="first")
]

# %% papermill={"duration": 0.026498, "end_time": "2020-12-11T20:14:40.722900", "exception": false, "start_time": "2020-12-11T20:14:40.696402", "status": "completed"} tags=[]
smultixcan_results.shape

# %% [markdown] papermill={"duration": 0.013811, "end_time": "2020-12-11T20:14:40.750934", "exception": false, "start_time": "2020-12-11T20:14:40.737123", "status": "completed"} tags=[]
# ## Remove NaN values

# %% [markdown] papermill={"duration": 0.01352, "end_time": "2020-12-11T20:14:40.779475", "exception": false, "start_time": "2020-12-11T20:14:40.765955", "status": "completed"} tags=[]
# **TODO**: it might be better to try to impute this values

# %% papermill={"duration": 0.591332, "end_time": "2020-12-11T20:14:41.384591", "exception": false, "start_time": "2020-12-11T20:14:40.793259", "status": "completed"} tags=[]
smultixcan_results = smultixcan_results.dropna(how="any")

# %% papermill={"duration": 0.024689, "end_time": "2020-12-11T20:14:41.424853", "exception": false, "start_time": "2020-12-11T20:14:41.400164", "status": "completed"} tags=[]
smultixcan_results.shape

# %% [markdown] papermill={"duration": 0.014189, "end_time": "2020-12-11T20:14:41.453568", "exception": false, "start_time": "2020-12-11T20:14:41.439379", "status": "completed"} tags=[]
# # Project S-MultiXcan data into MultiPLIER latent space

# %% papermill={"duration": 0.023353, "end_time": "2020-12-11T20:14:41.533964", "exception": false, "start_time": "2020-12-11T20:14:41.510611", "status": "completed"} tags=[]
mproj = MultiplierProjection()

# %% papermill={"duration": 4.661228, "end_time": "2020-12-11T20:14:46.210082", "exception": false, "start_time": "2020-12-11T20:14:41.548854", "status": "completed"} tags=[]
smultixcan_into_multiplier = mproj.transform(smultixcan_results)

# %% papermill={"duration": 0.024727, "end_time": "2020-12-11T20:14:46.250193", "exception": false, "start_time": "2020-12-11T20:14:46.225466", "status": "completed"} tags=[]
smultixcan_into_multiplier.shape

# %% papermill={"duration": 0.038282, "end_time": "2020-12-11T20:14:46.303982", "exception": false, "start_time": "2020-12-11T20:14:46.265700", "status": "completed"} tags=[]
smultixcan_into_multiplier.head()

# %% [markdown] papermill={"duration": 0.015054, "end_time": "2020-12-11T20:14:46.334546", "exception": false, "start_time": "2020-12-11T20:14:46.319492", "status": "completed"} tags=[]
# # Quick analysis

# %% papermill={"duration": 0.027054, "end_time": "2020-12-11T20:14:46.376539", "exception": false, "start_time": "2020-12-11T20:14:46.349485", "status": "completed"} tags=[]
(smultixcan_into_multiplier.loc["LV603"].sort_values(ascending=False).head(20))

# %% papermill={"duration": 0.027063, "end_time": "2020-12-11T20:14:46.419280", "exception": false, "start_time": "2020-12-11T20:14:46.392217", "status": "completed"} tags=[]
(smultixcan_into_multiplier.loc["LV136"].sort_values(ascending=False).head(20))

# %% [markdown] papermill={"duration": 0.015368, "end_time": "2020-12-11T20:14:46.450452", "exception": false, "start_time": "2020-12-11T20:14:46.435084", "status": "completed"} tags=[]
# # Save

# %% papermill={"duration": 0.025881, "end_time": "2020-12-11T20:14:46.491870", "exception": false, "start_time": "2020-12-11T20:14:46.465989", "status": "completed"} tags=[]
output_file = Path(
    RESULTS_PROJ_OUTPUT_DIR, f"projection-{results_filename_stem}.pkl"
).resolve()

display(output_file)

# %% papermill={"duration": 0.167471, "end_time": "2020-12-11T20:14:46.675370", "exception": false, "start_time": "2020-12-11T20:14:46.507899", "status": "completed"} tags=[]
smultixcan_into_multiplier.to_pickle(output_file)

# %% papermill={"duration": 0.01715, "end_time": "2020-12-11T20:14:46.739012", "exception": false, "start_time": "2020-12-11T20:14:46.721862", "status": "completed"} tags=[]
