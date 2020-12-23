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

# %% [markdown] papermill={"duration": 0.026465, "end_time": "2020-12-23T19:50:41.173233", "exception": false, "start_time": "2020-12-23T19:50:41.146768", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.008954, "end_time": "2020-12-23T19:50:41.193904", "exception": false, "start_time": "2020-12-23T19:50:41.184950", "status": "completed"} tags=[]
# **TODO**

# %% [markdown] papermill={"duration": 0.008741, "end_time": "2020-12-23T19:50:41.211367", "exception": false, "start_time": "2020-12-23T19:50:41.202626", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.020906, "end_time": "2020-12-23T19:50:41.241264", "exception": false, "start_time": "2020-12-23T19:50:41.220358", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 0.241664, "end_time": "2020-12-23T19:50:41.492228", "exception": false, "start_time": "2020-12-23T19:50:41.250564", "status": "completed"} tags=[]
from IPython.display import display
from pathlib import Path

import pandas as pd

import conf

# %% [markdown] papermill={"duration": 0.009209, "end_time": "2020-12-23T19:50:41.511289", "exception": false, "start_time": "2020-12-23T19:50:41.502080", "status": "completed"} tags=[]
# # Settings

# %% papermill={"duration": 0.01881, "end_time": "2020-12-23T19:50:41.539028", "exception": false, "start_time": "2020-12-23T19:50:41.520218", "status": "completed"} tags=[]
# if True, then it doesn't check if result files already exist and runs everything again
FORCE_RUN = False

# %% papermill={"duration": 0.019014, "end_time": "2020-12-23T19:50:41.567599", "exception": false, "start_time": "2020-12-23T19:50:41.548585", "status": "completed"} tags=[]
PREDICTION_METHOD = "Gene-based"

# %% papermill={"duration": 0.023785, "end_time": "2020-12-23T19:50:41.600953", "exception": false, "start_time": "2020-12-23T19:50:41.577168", "status": "completed"} tags=[]
OUTPUT_DIR = conf.RESULTS["DRUG_DISEASE_ANALYSES"]
display(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% papermill={"duration": 0.023779, "end_time": "2020-12-23T19:50:41.634743", "exception": false, "start_time": "2020-12-23T19:50:41.610964", "status": "completed"} tags=[]
# OUTPUT_DATA_DIR = Path(OUTPUT_DIR, "data")
# display(OUTPUT_DATA_DIR)
# OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# %% papermill={"duration": 0.022631, "end_time": "2020-12-23T19:50:41.676613", "exception": false, "start_time": "2020-12-23T19:50:41.653982", "status": "completed"} tags=[]
INPUT_DATA_DIR = Path(
    OUTPUT_DIR,
    "data",
    "raw",
)
display(INPUT_DATA_DIR)
INPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# %% papermill={"duration": 0.020424, "end_time": "2020-12-23T19:50:41.707114", "exception": false, "start_time": "2020-12-23T19:50:41.686690", "status": "completed"} tags=[]
OUTPUT_PREDICTIONS_DIR = Path(OUTPUT_DIR, "predictions")
display(OUTPUT_PREDICTIONS_DIR)
OUTPUT_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown] papermill={"duration": 0.009936, "end_time": "2020-12-23T19:50:41.727668", "exception": false, "start_time": "2020-12-23T19:50:41.717732", "status": "completed"} tags=[]
# # Load PharmacotherapyDB gold standard

# %% papermill={"duration": 0.020685, "end_time": "2020-12-23T19:50:41.758217", "exception": false, "start_time": "2020-12-23T19:50:41.737532", "status": "completed"} tags=[]
gold_standard = pd.read_pickle(
    Path(OUTPUT_DIR, "gold_standard.pkl"),
)

# %% papermill={"duration": 0.021437, "end_time": "2020-12-23T19:50:41.790924", "exception": false, "start_time": "2020-12-23T19:50:41.769487", "status": "completed"} tags=[]
display(gold_standard.shape)

# %% papermill={"duration": 0.025095, "end_time": "2020-12-23T19:50:41.827547", "exception": false, "start_time": "2020-12-23T19:50:41.802452", "status": "completed"} tags=[]
display(gold_standard.head())

# %% papermill={"duration": 0.021356, "end_time": "2020-12-23T19:50:41.861048", "exception": false, "start_time": "2020-12-23T19:50:41.839692", "status": "completed"} tags=[]
doids_in_gold_standard = set(gold_standard["trait"])

# %% [markdown] papermill={"duration": 0.011051, "end_time": "2020-12-23T19:50:41.883598", "exception": false, "start_time": "2020-12-23T19:50:41.872547", "status": "completed"} tags=[]
# # Load LINCS

# %% [markdown] papermill={"duration": 0.010577, "end_time": "2020-12-23T19:50:41.904997", "exception": false, "start_time": "2020-12-23T19:50:41.894420", "status": "completed"} tags=[]
# ## Raw data

# %% papermill={"duration": 0.021823, "end_time": "2020-12-23T19:50:41.937362", "exception": false, "start_time": "2020-12-23T19:50:41.915539", "status": "completed"} tags=[]
input_file = Path(INPUT_DATA_DIR, "lincs-data.pkl").resolve()

display(input_file)

# %% papermill={"duration": 0.05252, "end_time": "2020-12-23T19:50:42.001671", "exception": false, "start_time": "2020-12-23T19:50:41.949151", "status": "completed"} tags=[]
lincs_projection = pd.read_pickle(input_file)

# %% papermill={"duration": 0.022634, "end_time": "2020-12-23T19:50:42.037283", "exception": false, "start_time": "2020-12-23T19:50:42.014649", "status": "completed"} tags=[]
display(lincs_projection.shape)

# %% papermill={"duration": 0.035095, "end_time": "2020-12-23T19:50:42.084412", "exception": false, "start_time": "2020-12-23T19:50:42.049317", "status": "completed"} tags=[]
display(lincs_projection.head())

# %% [markdown] papermill={"duration": 0.011754, "end_time": "2020-12-23T19:50:42.108487", "exception": false, "start_time": "2020-12-23T19:50:42.096733", "status": "completed"} tags=[]
# # Load S-PrediXcan

# %% papermill={"duration": 0.224618, "end_time": "2020-12-23T19:50:42.345643", "exception": false, "start_time": "2020-12-23T19:50:42.121025", "status": "completed"} tags=[]
from entity import Trait

# %% papermill={"duration": 0.022572, "end_time": "2020-12-23T19:50:42.381817", "exception": false, "start_time": "2020-12-23T19:50:42.359245", "status": "completed"} tags=[]
phenomexcan_input_file_list = [
    f
    for f in INPUT_DATA_DIR.glob("*.pkl")
    if f.name.startswith(("smultixcan-", "spredixcan-"))
]

# %% papermill={"duration": 0.022258, "end_time": "2020-12-23T19:50:42.417071", "exception": false, "start_time": "2020-12-23T19:50:42.394813", "status": "completed"} tags=[]
display(len(phenomexcan_input_file_list))

# %% [markdown] papermill={"duration": 0.012297, "end_time": "2020-12-23T19:50:42.442121", "exception": false, "start_time": "2020-12-23T19:50:42.429824", "status": "completed"} tags=[]
# # Predict drug-disease associations

# %% papermill={"duration": 0.594541, "end_time": "2020-12-23T19:50:43.049740", "exception": false, "start_time": "2020-12-23T19:50:42.455199", "status": "completed"} tags=[]
from drug_disease import (
    predict_dotprod,
    predict_dotprod_neg,
    predict_pearson,
    predict_pearson_neg,
    predict_spearman,
    predict_spearman_neg,
)

# %% papermill={"duration": 5362.660602, "end_time": "2020-12-23T21:20:05.724677", "exception": false, "start_time": "2020-12-23T19:50:43.064075", "status": "completed"} tags=[]
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

    predict_pearson(
        lincs_projection,
        phenomexcan_input_file,
        phenomexcan_projection,
        OUTPUT_PREDICTIONS_DIR,
        PREDICTION_METHOD,
        doids_in_gold_standard,
        FORCE_RUN,
    )

    predict_pearson_neg(
        lincs_projection,
        phenomexcan_input_file,
        phenomexcan_projection,
        OUTPUT_PREDICTIONS_DIR,
        PREDICTION_METHOD,
        doids_in_gold_standard,
        FORCE_RUN,
    )

    predict_spearman(
        lincs_projection,
        phenomexcan_input_file,
        phenomexcan_projection,
        OUTPUT_PREDICTIONS_DIR,
        PREDICTION_METHOD,
        doids_in_gold_standard,
        FORCE_RUN,
    )

    predict_spearman_neg(
        lincs_projection,
        phenomexcan_input_file,
        phenomexcan_projection,
        OUTPUT_PREDICTIONS_DIR,
        PREDICTION_METHOD,
        doids_in_gold_standard,
        FORCE_RUN,
    )

    print("\n")

# %% papermill={"duration": 0.216001, "end_time": "2020-12-23T21:20:06.153906", "exception": false, "start_time": "2020-12-23T21:20:05.937905", "status": "completed"} tags=[]
