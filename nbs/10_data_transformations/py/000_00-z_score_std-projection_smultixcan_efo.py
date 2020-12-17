# ---
# jupyter:
#   jupytext:
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

# %% [markdown] papermill={"duration": 0.011523, "end_time": "2020-11-30T18:31:20.953979", "exception": false, "start_time": "2020-11-30T18:31:20.942456", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.006771, "end_time": "2020-11-30T18:31:20.967901", "exception": false, "start_time": "2020-11-30T18:31:20.961130", "status": "completed"} tags=[]
# It standardize (z-score) S-MultiXcan results projected into the MultiPLIER latent space.

# %% [markdown] papermill={"duration": 0.00691, "end_time": "2020-11-30T18:31:20.981753", "exception": false, "start_time": "2020-11-30T18:31:20.974843", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.018399, "end_time": "2020-11-30T18:31:21.007191", "exception": false, "start_time": "2020-11-30T18:31:20.988792", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 0.455238, "end_time": "2020-11-30T18:31:21.470327", "exception": false, "start_time": "2020-11-30T18:31:21.015089", "status": "completed"} tags=[]
from pathlib import Path
from IPython.display import display

import numpy as np
import pandas as pd
from sklearn.preprocessing import scale

import conf

# %% [markdown] papermill={"duration": 0.007125, "end_time": "2020-11-30T18:31:21.485093", "exception": false, "start_time": "2020-11-30T18:31:21.477968", "status": "completed"} tags=[]
# # Settings

# %% [markdown] papermill={"duration": 0.007, "end_time": "2020-11-30T18:31:21.499356", "exception": false, "start_time": "2020-11-30T18:31:21.492356", "status": "completed"} tags=[]
# ## Input data

# %% papermill={"duration": 0.025765, "end_time": "2020-11-30T18:31:21.532259", "exception": false, "start_time": "2020-11-30T18:31:21.506494", "status": "completed"} tags=[]
INPUT_FILEPATH = Path(
    conf.RESULTS["PROJECTIONS_DIR"],
    "projection-smultixcan-efo_partial-mashr-zscores.pkl",
).resolve()
display(INPUT_FILEPATH)

input_filepath_stem = INPUT_FILEPATH.stem
display(input_filepath_stem)

# %% [markdown] papermill={"duration": 0.00849, "end_time": "2020-11-30T18:31:21.549078", "exception": false, "start_time": "2020-11-30T18:31:21.540588", "status": "completed"} tags=[]
# ## Output folder

# %% papermill={"duration": 0.020543, "end_time": "2020-11-30T18:31:21.577496", "exception": false, "start_time": "2020-11-30T18:31:21.556953", "status": "completed"} tags=[]
# output dir for this notebook
RESULTS_DIR = Path(conf.RESULTS["DATA_TRANSFORMATIONS_DIR"], "z_score_std").resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %% [markdown] papermill={"duration": 0.00823, "end_time": "2020-11-30T18:31:21.594080", "exception": false, "start_time": "2020-11-30T18:31:21.585850", "status": "completed"} tags=[]
# # Load input file

# %% papermill={"duration": 0.033123, "end_time": "2020-11-30T18:31:21.635051", "exception": false, "start_time": "2020-11-30T18:31:21.601928", "status": "completed"} tags=[]
data = pd.read_pickle(INPUT_FILEPATH).T

# %% papermill={"duration": 0.021271, "end_time": "2020-11-30T18:31:21.665729", "exception": false, "start_time": "2020-11-30T18:31:21.644458", "status": "completed"} tags=[]
display(data.shape)

# %% papermill={"duration": 0.034551, "end_time": "2020-11-30T18:31:21.709652", "exception": false, "start_time": "2020-11-30T18:31:21.675101", "status": "completed"} tags=[]
display(data.head())

# %% [markdown] papermill={"duration": 0.009354, "end_time": "2020-11-30T18:31:21.728656", "exception": false, "start_time": "2020-11-30T18:31:21.719302", "status": "completed"} tags=[]
# # z-score standardization

# %% papermill={"duration": 0.04679, "end_time": "2020-11-30T18:31:21.784465", "exception": false, "start_time": "2020-11-30T18:31:21.737675", "status": "completed"} tags=[]
data_stats = data.iloc[:, :10].describe()
display(data_stats)

# %% papermill={"duration": 0.11351, "end_time": "2020-11-30T18:31:21.907840", "exception": false, "start_time": "2020-11-30T18:31:21.794330", "status": "completed"} tags=[]
scaled_data = pd.DataFrame(
    data=scale(data), index=data.index.copy(), columns=data.columns.copy()
)

# %% papermill={"duration": 0.02348, "end_time": "2020-11-30T18:31:21.942449", "exception": false, "start_time": "2020-11-30T18:31:21.918969", "status": "completed"} tags=[]
display(scaled_data.shape)

# %% papermill={"duration": 0.035982, "end_time": "2020-11-30T18:31:21.989620", "exception": false, "start_time": "2020-11-30T18:31:21.953638", "status": "completed"} tags=[]
display(scaled_data.head())

# %% papermill={"duration": 0.050483, "end_time": "2020-11-30T18:31:22.051794", "exception": false, "start_time": "2020-11-30T18:31:22.001311", "status": "completed"} tags=[]
scaled_data_stats = scaled_data.iloc[:, :10].describe()
display(scaled_data_stats)

# %% [markdown] papermill={"duration": 0.010906, "end_time": "2020-11-30T18:31:22.073630", "exception": false, "start_time": "2020-11-30T18:31:22.062724", "status": "completed"} tags=[]
# ## Testing

# %% papermill={"duration": 0.023383, "end_time": "2020-11-30T18:31:22.107636", "exception": false, "start_time": "2020-11-30T18:31:22.084253", "status": "completed"} tags=[]
assert np.all(
    [
        np.isclose(scaled_data_stats.loc["mean", c], 0.0)
        for c in scaled_data_stats.columns
    ]
)

# %% papermill={"duration": 0.023189, "end_time": "2020-11-30T18:31:22.142127", "exception": false, "start_time": "2020-11-30T18:31:22.118938", "status": "completed"} tags=[]
assert np.all(
    [
        np.isclose(scaled_data_stats.loc["std", c], 1.0, atol=1e-03)
        for c in scaled_data_stats.columns
    ]
)

# %% [markdown] papermill={"duration": 0.010762, "end_time": "2020-11-30T18:31:22.164014", "exception": false, "start_time": "2020-11-30T18:31:22.153252", "status": "completed"} tags=[]
# # Save

# %% papermill={"duration": 0.024102, "end_time": "2020-11-30T18:31:22.199112", "exception": false, "start_time": "2020-11-30T18:31:22.175010", "status": "completed"} tags=[]
output_file = Path(
    RESULTS_DIR,
    f"z_score_std-{input_filepath_stem}.pkl",
).resolve()

display(output_file)

# %% papermill={"duration": 0.052456, "end_time": "2020-11-30T18:31:22.263115", "exception": false, "start_time": "2020-11-30T18:31:22.210659", "status": "completed"} tags=[]
scaled_data.to_pickle(output_file)

# %% papermill={"duration": 0.011676, "end_time": "2020-11-30T18:31:22.286862", "exception": false, "start_time": "2020-11-30T18:31:22.275186", "status": "completed"} tags=[]
