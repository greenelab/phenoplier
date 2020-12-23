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

# %% [markdown] papermill={"duration": 0.039383, "end_time": "2020-12-23T18:05:20.790329", "exception": false, "start_time": "2020-12-23T18:05:20.750946", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.010248, "end_time": "2020-12-23T18:05:20.814612", "exception": false, "start_time": "2020-12-23T18:05:20.804364", "status": "completed"} tags=[]
# **TODO**

# %% [markdown] papermill={"duration": 0.009786, "end_time": "2020-12-23T18:05:20.834637", "exception": false, "start_time": "2020-12-23T18:05:20.824851", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.022074, "end_time": "2020-12-23T18:05:20.866594", "exception": false, "start_time": "2020-12-23T18:05:20.844520", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 0.238689, "end_time": "2020-12-23T18:05:21.115690", "exception": false, "start_time": "2020-12-23T18:05:20.877001", "status": "completed"} tags=[]
from IPython.display import display
from pathlib import Path

import pandas as pd

import conf

# %% [markdown] papermill={"duration": 0.010343, "end_time": "2020-12-23T18:05:21.136621", "exception": false, "start_time": "2020-12-23T18:05:21.126278", "status": "completed"} tags=[]
# # Settings

# %% papermill={"duration": 0.020294, "end_time": "2020-12-23T18:05:21.166974", "exception": false, "start_time": "2020-12-23T18:05:21.146680", "status": "completed"} tags=[]
# if True, then it doesn't check if result files already exist and runs everything again
FORCE_RUN = False

# %% papermill={"duration": 0.020017, "end_time": "2020-12-23T18:05:21.197349", "exception": false, "start_time": "2020-12-23T18:05:21.177332", "status": "completed"} tags=[]
PREDICTION_METHOD = "Gene-based"

# %% papermill={"duration": 0.026973, "end_time": "2020-12-23T18:05:21.234826", "exception": false, "start_time": "2020-12-23T18:05:21.207853", "status": "completed"} tags=[]
OUTPUT_DIR = conf.RESULTS["DRUG_DISEASE_ANALYSES"]
display(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% papermill={"duration": 0.022135, "end_time": "2020-12-23T18:05:21.268833", "exception": false, "start_time": "2020-12-23T18:05:21.246698", "status": "completed"} tags=[]
# OUTPUT_DATA_DIR = Path(OUTPUT_DIR, "data")
# display(OUTPUT_DATA_DIR)
# OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# %% papermill={"duration": 0.025055, "end_time": "2020-12-23T18:05:21.305648", "exception": false, "start_time": "2020-12-23T18:05:21.280593", "status": "completed"} tags=[]
INPUT_DATA_DIR = Path(
    OUTPUT_DIR,
    "data",
    "raw",
)
display(INPUT_DATA_DIR)
INPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# %% papermill={"duration": 0.021082, "end_time": "2020-12-23T18:05:21.337916", "exception": false, "start_time": "2020-12-23T18:05:21.316834", "status": "completed"} tags=[]
OUTPUT_PREDICTIONS_DIR = Path(OUTPUT_DIR, "predictions")
display(OUTPUT_PREDICTIONS_DIR)
OUTPUT_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown] papermill={"duration": 0.010727, "end_time": "2020-12-23T18:05:21.359594", "exception": false, "start_time": "2020-12-23T18:05:21.348867", "status": "completed"} tags=[]
# # Load PharmacotherapyDB gold standard

# %% papermill={"duration": 0.021616, "end_time": "2020-12-23T18:05:21.392509", "exception": false, "start_time": "2020-12-23T18:05:21.370893", "status": "completed"} tags=[]
gold_standard = pd.read_pickle(
    Path(OUTPUT_DIR, "gold_standard.pkl"),
)

# %% papermill={"duration": 0.021491, "end_time": "2020-12-23T18:05:21.425152", "exception": false, "start_time": "2020-12-23T18:05:21.403661", "status": "completed"} tags=[]
display(gold_standard.shape)

# %% papermill={"duration": 0.02498, "end_time": "2020-12-23T18:05:21.461457", "exception": false, "start_time": "2020-12-23T18:05:21.436477", "status": "completed"} tags=[]
display(gold_standard.head())

# %% papermill={"duration": 0.02083, "end_time": "2020-12-23T18:05:21.493934", "exception": false, "start_time": "2020-12-23T18:05:21.473104", "status": "completed"} tags=[]
doids_in_gold_standard = set(gold_standard["trait"])

# %% [markdown] papermill={"duration": 0.011148, "end_time": "2020-12-23T18:05:21.516609", "exception": false, "start_time": "2020-12-23T18:05:21.505461", "status": "completed"} tags=[]
# # Load LINCS

# %% [markdown] papermill={"duration": 0.010843, "end_time": "2020-12-23T18:05:21.538630", "exception": false, "start_time": "2020-12-23T18:05:21.527787", "status": "completed"} tags=[]
# ## Raw data

# %% papermill={"duration": 0.021893, "end_time": "2020-12-23T18:05:21.572955", "exception": false, "start_time": "2020-12-23T18:05:21.551062", "status": "completed"} tags=[]
input_file = Path(INPUT_DATA_DIR, "lincs-data.pkl").resolve()

display(input_file)

# %% papermill={"duration": 0.050725, "end_time": "2020-12-23T18:05:21.635650", "exception": false, "start_time": "2020-12-23T18:05:21.584925", "status": "completed"} tags=[]
lincs_projection = pd.read_pickle(input_file)

# %% papermill={"duration": 0.022481, "end_time": "2020-12-23T18:05:21.671018", "exception": false, "start_time": "2020-12-23T18:05:21.648537", "status": "completed"} tags=[]
display(lincs_projection.shape)

# %% papermill={"duration": 0.034702, "end_time": "2020-12-23T18:05:21.718059", "exception": false, "start_time": "2020-12-23T18:05:21.683357", "status": "completed"} tags=[]
display(lincs_projection.head())

# %% [markdown] papermill={"duration": 0.012171, "end_time": "2020-12-23T18:05:21.743039", "exception": false, "start_time": "2020-12-23T18:05:21.730868", "status": "completed"} tags=[]
# # Load S-PrediXcan

# %% papermill={"duration": 0.224496, "end_time": "2020-12-23T18:05:21.980123", "exception": false, "start_time": "2020-12-23T18:05:21.755627", "status": "completed"} tags=[]
from entity import Trait

# %% papermill={"duration": 0.023597, "end_time": "2020-12-23T18:05:22.017319", "exception": false, "start_time": "2020-12-23T18:05:21.993722", "status": "completed"} tags=[]
phenomexcan_input_file_list = [
    f
    for f in INPUT_DATA_DIR.glob("*.pkl")
    if f.name.startswith(("smultixcan-", "spredixcan-"))
]

# %% papermill={"duration": 0.022527, "end_time": "2020-12-23T18:05:22.052846", "exception": false, "start_time": "2020-12-23T18:05:22.030319", "status": "completed"} tags=[]
display(len(phenomexcan_input_file_list))

# %% [markdown] papermill={"duration": 0.013435, "end_time": "2020-12-23T18:05:22.078841", "exception": false, "start_time": "2020-12-23T18:05:22.065406", "status": "completed"} tags=[]
# # Predict drug-disease associations

# %% papermill={"duration": 0.022645, "end_time": "2020-12-23T18:05:22.114498", "exception": false, "start_time": "2020-12-23T18:05:22.091853", "status": "completed"} tags=[]
from drug_disease import predict_dotprod, predict_dotprod_neg

# %% papermill={"duration": 2059.56362, "end_time": "2020-12-23T18:39:41.691237", "exception": false, "start_time": "2020-12-23T18:05:22.127617", "status": "completed"} tags=[]
for phenomexcan_input_file in phenomexcan_input_file_list:
    print(phenomexcan_input_file.name)

    # read phenomexcan data
    phenomexcan_projection = pd.read_pickle(phenomexcan_input_file)

    # get common genes with lincs
    common_genes = phenomexcan_projection.index.intersection(lincs_projection.index)
    phenomexcan_projection = phenomexcan_projection.loc[common_genes]
    lincs_projection = lincs_projection.loc[common_genes]

    print(f"  shape: {phenomexcan_projection.shape}")

    predict_dotprod(
        lincs_projection,
        phenomexcan_input_file,
        phenomexcan_projection,
        OUTPUT_PREDICTIONS_DIR,
        PREDICTION_METHOD,
        doids_in_gold_standard,
        FORCE_RUN,
    )

    predict_dotprod_neg(
        lincs_projection,
        phenomexcan_input_file,
        phenomexcan_projection,
        OUTPUT_PREDICTIONS_DIR,
        PREDICTION_METHOD,
        doids_in_gold_standard,
        FORCE_RUN,
    )

    print("\n")

# %% papermill={"duration": 0.113934, "end_time": "2020-12-23T18:39:41.917951", "exception": false, "start_time": "2020-12-23T18:39:41.804017", "status": "completed"} tags=[]
