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

# %% [markdown] papermill={"duration": 0.047033, "end_time": "2020-12-14T21:31:29.800874", "exception": false, "start_time": "2020-12-14T21:31:29.753841", "status": "completed"} tags=[]
# # Description

# %% [markdown]
# It projects the PhenomeXcan results (EFO version) into the MultiPLIER latent space.

# %% [markdown] papermill={"duration": 0.011994, "end_time": "2020-12-14T21:31:29.828385", "exception": false, "start_time": "2020-12-14T21:31:29.816391", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.023354, "end_time": "2020-12-14T21:31:29.864028", "exception": false, "start_time": "2020-12-14T21:31:29.840674", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 0.190401, "end_time": "2020-12-14T21:31:30.067170", "exception": false, "start_time": "2020-12-14T21:31:29.876769", "status": "completed"} tags=[]
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

import conf
from data.cache import read_data

# %% [markdown] papermill={"duration": 0.012328, "end_time": "2020-12-14T21:31:30.092521", "exception": false, "start_time": "2020-12-14T21:31:30.080193", "status": "completed"} tags=[]
# # Settings

# %% papermill={"duration": 0.027255, "end_time": "2020-12-14T21:31:30.132036", "exception": false, "start_time": "2020-12-14T21:31:30.104781", "status": "completed"} tags=[]
# The percentile name indicates the top percentage of genes retained
PERCENTILE_NAME = 'pALL'

display(PERCENTILE_NAME)

# %% papermill={"duration": 0.02383, "end_time": "2020-12-14T21:31:30.169498", "exception": false, "start_time": "2020-12-14T21:31:30.145668", "status": "completed"} tags=[]
RESULTS_PROJ_OUTPUT_DIR = Path(
    conf.RESULTS['PROJECTIONS_DIR']
)

RESULTS_PROJ_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_PROJ_OUTPUT_DIR)

# %% [markdown] papermill={"duration": 0.013697, "end_time": "2020-12-14T21:31:30.197325", "exception": false, "start_time": "2020-12-14T21:31:30.183628", "status": "completed"} tags=[]
# # Read gene mappings

# %% papermill={"duration": 0.035891, "end_time": "2020-12-14T21:31:30.245844", "exception": false, "start_time": "2020-12-14T21:31:30.209953", "status": "completed"} tags=[]
GENE_ID_TO_NAME_MAP = read_data(conf.PHENOMEXCAN["GENE_MAP_ID_TO_NAME"])
GENE_NAME_TO_ID_MAP = read_data(conf.PHENOMEXCAN["GENE_MAP_NAME_TO_ID"])

# %% [markdown] papermill={"duration": 0.012438, "end_time": "2020-12-14T21:31:30.271523", "exception": false, "start_time": "2020-12-14T21:31:30.259085", "status": "completed"} tags=[]
# # Load PhenomeXcan data (S-MultiXcan)

# %% papermill={"duration": 0.023574, "end_time": "2020-12-14T21:31:30.308387", "exception": false, "start_time": "2020-12-14T21:31:30.284813", "status": "completed"} tags=[]
smultixcan_results_filename = conf.PHENOMEXCAN["SMULTIXCAN_EFO_PARTIAL_MASHR_ZSCORES_FILE"]

display(smultixcan_results_filename)

# %% papermill={"duration": 0.023418, "end_time": "2020-12-14T21:31:30.345381", "exception": false, "start_time": "2020-12-14T21:31:30.321963", "status": "completed"} tags=[]
results_filename_stem = smultixcan_results_filename.stem
display(results_filename_stem)

# %% papermill={"duration": 0.238024, "end_time": "2020-12-14T21:31:30.597055", "exception": false, "start_time": "2020-12-14T21:31:30.359031", "status": "completed"} tags=[]
smultixcan_results = pd.read_pickle(smultixcan_results_filename)

# %% papermill={"duration": 0.024174, "end_time": "2020-12-14T21:31:30.636763", "exception": false, "start_time": "2020-12-14T21:31:30.612589", "status": "completed"} tags=[]
smultixcan_results.shape

# %% papermill={"duration": 0.037316, "end_time": "2020-12-14T21:31:30.687761", "exception": false, "start_time": "2020-12-14T21:31:30.650445", "status": "completed"} tags=[]
smultixcan_results.head()

# %% [markdown] papermill={"duration": 0.013902, "end_time": "2020-12-14T21:31:30.716289", "exception": false, "start_time": "2020-12-14T21:31:30.702387", "status": "completed"} tags=[]
# ## Gene IDs to Gene names

# %% papermill={"duration": 0.167814, "end_time": "2020-12-14T21:31:30.897768", "exception": false, "start_time": "2020-12-14T21:31:30.729954", "status": "completed"} tags=[]
smultixcan_results = smultixcan_results.rename(index=GENE_ID_TO_NAME_MAP)

# %% papermill={"duration": 0.025327, "end_time": "2020-12-14T21:31:30.938922", "exception": false, "start_time": "2020-12-14T21:31:30.913595", "status": "completed"} tags=[]
smultixcan_results.shape

# %% papermill={"duration": 0.036406, "end_time": "2020-12-14T21:31:30.989819", "exception": false, "start_time": "2020-12-14T21:31:30.953413", "status": "completed"} tags=[]
smultixcan_results.head()

# %% [markdown] papermill={"duration": 0.014396, "end_time": "2020-12-14T21:31:31.019300", "exception": false, "start_time": "2020-12-14T21:31:31.004904", "status": "completed"} tags=[]
# ## Remove duplicated gene entries

# %% papermill={"duration": 0.02823, "end_time": "2020-12-14T21:31:31.061744", "exception": false, "start_time": "2020-12-14T21:31:31.033514", "status": "completed"} tags=[]
smultixcan_results.index[smultixcan_results.index.duplicated(keep='first')]

# %% papermill={"duration": 0.168551, "end_time": "2020-12-14T21:31:31.245604", "exception": false, "start_time": "2020-12-14T21:31:31.077053", "status": "completed"} tags=[]
smultixcan_results = smultixcan_results.loc[~smultixcan_results.index.duplicated(keep='first')]

# %% papermill={"duration": 0.025256, "end_time": "2020-12-14T21:31:31.286738", "exception": false, "start_time": "2020-12-14T21:31:31.261482", "status": "completed"} tags=[]
smultixcan_results.shape

# %% [markdown] papermill={"duration": 0.01438, "end_time": "2020-12-14T21:31:31.316160", "exception": false, "start_time": "2020-12-14T21:31:31.301780", "status": "completed"} tags=[]
# ## Remove NaN values

# %% [markdown] papermill={"duration": 0.014617, "end_time": "2020-12-14T21:31:31.345449", "exception": false, "start_time": "2020-12-14T21:31:31.330832", "status": "completed"} tags=[]
# **TODO**: it might be better to try to impute this values

# %% papermill={"duration": 0.305567, "end_time": "2020-12-14T21:31:31.665528", "exception": false, "start_time": "2020-12-14T21:31:31.359961", "status": "completed"} tags=[]
smultixcan_results = smultixcan_results.dropna(how='any')

# %% papermill={"duration": 0.024867, "end_time": "2020-12-14T21:31:31.706535", "exception": false, "start_time": "2020-12-14T21:31:31.681668", "status": "completed"} tags=[]
smultixcan_results.shape

# %% [markdown] papermill={"duration": 0.015047, "end_time": "2020-12-14T21:31:31.737171", "exception": false, "start_time": "2020-12-14T21:31:31.722124", "status": "completed"} tags=[]
# # Project S-MultiXcan data into MultiPLIER latent space

# %% papermill={"duration": 0.024402, "end_time": "2020-12-14T21:31:31.776644", "exception": false, "start_time": "2020-12-14T21:31:31.752242", "status": "completed"} tags=[]
from multiplier import MultiplierProjection

# %% papermill={"duration": 0.024302, "end_time": "2020-12-14T21:31:31.815949", "exception": false, "start_time": "2020-12-14T21:31:31.791647", "status": "completed"} tags=[]
mproj = MultiplierProjection()

# %% papermill={"duration": 2.856489, "end_time": "2020-12-14T21:31:34.687961", "exception": false, "start_time": "2020-12-14T21:31:31.831472", "status": "completed"} tags=[]
smultixcan_into_multiplier = mproj.transform(smultixcan_results)

# %% papermill={"duration": 0.025345, "end_time": "2020-12-14T21:31:34.729226", "exception": false, "start_time": "2020-12-14T21:31:34.703881", "status": "completed"} tags=[]
smultixcan_into_multiplier.shape

# %% papermill={"duration": 0.037484, "end_time": "2020-12-14T21:31:34.782824", "exception": false, "start_time": "2020-12-14T21:31:34.745340", "status": "completed"} tags=[]
smultixcan_into_multiplier.head()

# %% [markdown] papermill={"duration": 0.015675, "end_time": "2020-12-14T21:31:34.815029", "exception": false, "start_time": "2020-12-14T21:31:34.799354", "status": "completed"} tags=[]
# # Quick analysis

# %% papermill={"duration": 0.02758, "end_time": "2020-12-14T21:31:34.858382", "exception": false, "start_time": "2020-12-14T21:31:34.830802", "status": "completed"} tags=[]
(
    smultixcan_into_multiplier.loc['LV603']
    .sort_values(ascending=False)
    .head(20)
)

# %% papermill={"duration": 0.027231, "end_time": "2020-12-14T21:31:34.901425", "exception": false, "start_time": "2020-12-14T21:31:34.874194", "status": "completed"} tags=[]
(
    smultixcan_into_multiplier.loc['LV136']
    .sort_values(ascending=False)
    .head(20)
)

# %% [markdown] papermill={"duration": 0.01596, "end_time": "2020-12-14T21:31:34.933702", "exception": false, "start_time": "2020-12-14T21:31:34.917742", "status": "completed"} tags=[]
# # Save

# %% papermill={"duration": 0.026234, "end_time": "2020-12-14T21:31:34.975461", "exception": false, "start_time": "2020-12-14T21:31:34.949227", "status": "completed"} tags=[]
output_file = Path(
    RESULTS_PROJ_OUTPUT_DIR,
    f'projection-{results_filename_stem}.pkl'
).resolve()

display(output_file)

# %% papermill={"duration": 0.207322, "end_time": "2020-12-14T21:31:35.199214", "exception": false, "start_time": "2020-12-14T21:31:34.991892", "status": "completed"} tags=[]
smultixcan_into_multiplier.to_pickle(output_file)

# %% papermill={"duration": 0.016809, "end_time": "2020-12-14T21:31:35.244229", "exception": false, "start_time": "2020-12-14T21:31:35.227420", "status": "completed"} tags=[]
