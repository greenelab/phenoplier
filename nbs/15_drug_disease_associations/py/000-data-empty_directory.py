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

# %% [markdown] papermill={"duration": 0.007976, "end_time": "2020-12-23T17:59:22.240217", "exception": false, "start_time": "2020-12-23T17:59:22.232241", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.003976, "end_time": "2020-12-23T17:59:22.248255", "exception": false, "start_time": "2020-12-23T17:59:22.244279", "status": "completed"} tags=[]
# It makes sure that the data directory for predictions is empty before these notebooks are run.
#
# Do I need this?

# %% [markdown] papermill={"duration": 0.003769, "end_time": "2020-12-23T17:59:22.255905", "exception": false, "start_time": "2020-12-23T17:59:22.252136", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.014707, "end_time": "2020-12-23T17:59:22.274511", "exception": false, "start_time": "2020-12-23T17:59:22.259804", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 0.211138, "end_time": "2020-12-23T17:59:22.489841", "exception": false, "start_time": "2020-12-23T17:59:22.278703", "status": "completed"} tags=[]
from pathlib import Path

import numpy as np
import pandas as pd

import conf

# %% [markdown] papermill={"duration": 0.004028, "end_time": "2020-12-23T17:59:22.498156", "exception": false, "start_time": "2020-12-23T17:59:22.494128", "status": "completed"} tags=[]
# # Settings

# %% papermill={"duration": 0.018476, "end_time": "2020-12-23T17:59:22.520574", "exception": false, "start_time": "2020-12-23T17:59:22.502098", "status": "completed"} tags=[]
OUTPUT_DIR = conf.RESULTS["DRUG_DISEASE_ANALYSES"]
display(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% papermill={"duration": 0.014514, "end_time": "2020-12-23T17:59:22.539593", "exception": false, "start_time": "2020-12-23T17:59:22.525079", "status": "completed"} tags=[]
OUTPUT_DATA_DIR = Path(OUTPUT_DIR, "data")
display(OUTPUT_DATA_DIR)
OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown] papermill={"duration": 0.004541, "end_time": "2020-12-23T17:59:22.548952", "exception": false, "start_time": "2020-12-23T17:59:22.544411", "status": "completed"} tags=[]
# # Remove previous predictions

# %% papermill={"duration": 0.014057, "end_time": "2020-12-23T17:59:22.567665", "exception": false, "start_time": "2020-12-23T17:59:22.553608", "status": "completed"} tags=[]
current_prediction_files = OUTPUT_DATA_DIR.rglob("*.pkl")

# %% papermill={"duration": 1.185649, "end_time": "2020-12-23T17:59:23.758106", "exception": false, "start_time": "2020-12-23T17:59:22.572457", "status": "completed"} tags=[]
for f in current_prediction_files:
    display(f)
    f.unlink()

# %% papermill={"duration": 0.027515, "end_time": "2020-12-23T17:59:23.813540", "exception": false, "start_time": "2020-12-23T17:59:23.786025", "status": "completed"} tags=[]
