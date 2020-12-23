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

# %% [markdown] papermill={"duration": 0.019234, "end_time": "2020-12-23T18:04:57.485194", "exception": false, "start_time": "2020-12-23T18:04:57.465960", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.008833, "end_time": "2020-12-23T18:04:57.510470", "exception": false, "start_time": "2020-12-23T18:04:57.501637", "status": "completed"} tags=[]
# It makes sure that the output directory for predictions is empty before these notebooks are run.

# %% [markdown] papermill={"duration": 0.00523, "end_time": "2020-12-23T18:04:57.522081", "exception": false, "start_time": "2020-12-23T18:04:57.516851", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.015659, "end_time": "2020-12-23T18:04:57.542339", "exception": false, "start_time": "2020-12-23T18:04:57.526680", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 0.1833, "end_time": "2020-12-23T18:04:57.729890", "exception": false, "start_time": "2020-12-23T18:04:57.546590", "status": "completed"} tags=[]
from pathlib import Path

import numpy as np
import pandas as pd

import conf

# %% [markdown] papermill={"duration": 0.003867, "end_time": "2020-12-23T18:04:57.737952", "exception": false, "start_time": "2020-12-23T18:04:57.734085", "status": "completed"} tags=[]
# # Settings

# %% papermill={"duration": 0.018298, "end_time": "2020-12-23T18:04:57.760191", "exception": false, "start_time": "2020-12-23T18:04:57.741893", "status": "completed"} tags=[]
OUTPUT_DIR = conf.RESULTS["DRUG_DISEASE_ANALYSES"]
display(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% papermill={"duration": 0.014887, "end_time": "2020-12-23T18:04:57.779457", "exception": false, "start_time": "2020-12-23T18:04:57.764570", "status": "completed"} tags=[]
OUTPUT_PREDICTIONS_DIR = Path(OUTPUT_DIR, "predictions")
display(OUTPUT_PREDICTIONS_DIR)
OUTPUT_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown] papermill={"duration": 0.004625, "end_time": "2020-12-23T18:04:57.788975", "exception": false, "start_time": "2020-12-23T18:04:57.784350", "status": "completed"} tags=[]
# # Remove previous predictions

# %% papermill={"duration": 0.015255, "end_time": "2020-12-23T18:04:57.808959", "exception": false, "start_time": "2020-12-23T18:04:57.793704", "status": "completed"} tags=[]
current_prediction_files = OUTPUT_PREDICTIONS_DIR.rglob("*.h5")

# %% papermill={"duration": 0.01444, "end_time": "2020-12-23T18:04:57.828347", "exception": false, "start_time": "2020-12-23T18:04:57.813907", "status": "completed"} tags=[]
for f in current_prediction_files:
    display(f)
    f.unlink()

# %% papermill={"duration": 0.004645, "end_time": "2020-12-23T18:04:57.837888", "exception": false, "start_time": "2020-12-23T18:04:57.833243", "status": "completed"} tags=[]
