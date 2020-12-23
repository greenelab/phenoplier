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

# %% [markdown] papermill={"duration": 0.036359, "end_time": "2020-12-23T19:50:41.165143", "exception": false, "start_time": "2020-12-23T19:50:41.128784", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.009766, "end_time": "2020-12-23T19:50:41.189531", "exception": false, "start_time": "2020-12-23T19:50:41.179765", "status": "completed"} tags=[]
# **TODO**

# %% [markdown] papermill={"duration": 0.00867, "end_time": "2020-12-23T19:50:41.206942", "exception": false, "start_time": "2020-12-23T19:50:41.198272", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.020623, "end_time": "2020-12-23T19:50:41.236311", "exception": false, "start_time": "2020-12-23T19:50:41.215688", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 0.247588, "end_time": "2020-12-23T19:50:41.493393", "exception": false, "start_time": "2020-12-23T19:50:41.245805", "status": "completed"} tags=[]
from IPython.display import display
from pathlib import Path

import pandas as pd

import conf

# %% [markdown] papermill={"duration": 0.008919, "end_time": "2020-12-23T19:50:41.511867", "exception": false, "start_time": "2020-12-23T19:50:41.502948", "status": "completed"} tags=[]
# # Settings

# %% papermill={"duration": 0.018418, "end_time": "2020-12-23T19:50:41.539027", "exception": false, "start_time": "2020-12-23T19:50:41.520609", "status": "completed"} tags=[]
# if True, then it doesn't check if result files already exist and runs everything again
FORCE_RUN = False

# %% papermill={"duration": 0.019205, "end_time": "2020-12-23T19:50:41.567597", "exception": false, "start_time": "2020-12-23T19:50:41.548392", "status": "completed"} tags=[]
PREDICTION_METHOD = "Module-based"

# %% papermill={"duration": 0.024172, "end_time": "2020-12-23T19:50:41.601289", "exception": false, "start_time": "2020-12-23T19:50:41.577117", "status": "completed"} tags=[]
OUTPUT_DIR = conf.RESULTS["DRUG_DISEASE_ANALYSES"]
display(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% papermill={"duration": 0.023388, "end_time": "2020-12-23T19:50:41.634609", "exception": false, "start_time": "2020-12-23T19:50:41.611221", "status": "completed"} tags=[]
# OUTPUT_DATA_DIR = Path(OUTPUT_DIR, "data")
# display(OUTPUT_DATA_DIR)
# OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# %% papermill={"duration": 0.022741, "end_time": "2020-12-23T19:50:41.676575", "exception": false, "start_time": "2020-12-23T19:50:41.653834", "status": "completed"} tags=[]
INPUT_DATA_DIR = Path(
    OUTPUT_DIR,
    "data",
    "proj",
)
display(INPUT_DATA_DIR)
INPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# %% papermill={"duration": 0.022117, "end_time": "2020-12-23T19:50:41.708701", "exception": false, "start_time": "2020-12-23T19:50:41.686584", "status": "completed"} tags=[]
OUTPUT_PREDICTIONS_DIR = Path(OUTPUT_DIR, "predictions")
display(OUTPUT_PREDICTIONS_DIR)
OUTPUT_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown] papermill={"duration": 0.009765, "end_time": "2020-12-23T19:50:41.728870", "exception": false, "start_time": "2020-12-23T19:50:41.719105", "status": "completed"} tags=[]
# # Load PharmacotherapyDB gold standard

# %% papermill={"duration": 0.025208, "end_time": "2020-12-23T19:50:41.763784", "exception": false, "start_time": "2020-12-23T19:50:41.738576", "status": "completed"} tags=[]
gold_standard = pd.read_pickle(
    Path(OUTPUT_DIR, "gold_standard.pkl"),
)

# %% papermill={"duration": 0.021155, "end_time": "2020-12-23T19:50:41.796102", "exception": false, "start_time": "2020-12-23T19:50:41.774947", "status": "completed"} tags=[]
display(gold_standard.shape)

# %% papermill={"duration": 0.024921, "end_time": "2020-12-23T19:50:41.831886", "exception": false, "start_time": "2020-12-23T19:50:41.806965", "status": "completed"} tags=[]
display(gold_standard.head())

# %% papermill={"duration": 0.02118, "end_time": "2020-12-23T19:50:41.864638", "exception": false, "start_time": "2020-12-23T19:50:41.843458", "status": "completed"} tags=[]
doids_in_gold_standard = set(gold_standard["trait"])

# %% [markdown] papermill={"duration": 0.010857, "end_time": "2020-12-23T19:50:41.886745", "exception": false, "start_time": "2020-12-23T19:50:41.875888", "status": "completed"} tags=[]
# # Load LINCS

# %% [markdown] papermill={"duration": 0.010454, "end_time": "2020-12-23T19:50:41.907921", "exception": false, "start_time": "2020-12-23T19:50:41.897467", "status": "completed"} tags=[]
# ## Projected data

# %% papermill={"duration": 0.021901, "end_time": "2020-12-23T19:50:41.940473", "exception": false, "start_time": "2020-12-23T19:50:41.918572", "status": "completed"} tags=[]
input_file = Path(INPUT_DATA_DIR, "lincs-projection.pkl").resolve()

display(input_file)

# %% papermill={"duration": 0.026636, "end_time": "2020-12-23T19:50:41.978518", "exception": false, "start_time": "2020-12-23T19:50:41.951882", "status": "completed"} tags=[]
lincs_projection = pd.read_pickle(input_file)

# %% papermill={"duration": 0.024037, "end_time": "2020-12-23T19:50:42.015614", "exception": false, "start_time": "2020-12-23T19:50:41.991577", "status": "completed"} tags=[]
display(lincs_projection.shape)

# %% papermill={"duration": 0.034451, "end_time": "2020-12-23T19:50:42.062781", "exception": false, "start_time": "2020-12-23T19:50:42.028330", "status": "completed"} tags=[]
display(lincs_projection.head())

# %% [markdown] papermill={"duration": 0.012417, "end_time": "2020-12-23T19:50:42.087747", "exception": false, "start_time": "2020-12-23T19:50:42.075330", "status": "completed"} tags=[]
# # Load S-PrediXcan

# %% papermill={"duration": 0.217486, "end_time": "2020-12-23T19:50:42.317086", "exception": false, "start_time": "2020-12-23T19:50:42.099600", "status": "completed"} tags=[]
from entity import Trait

# %% papermill={"duration": 0.024685, "end_time": "2020-12-23T19:50:42.356631", "exception": false, "start_time": "2020-12-23T19:50:42.331946", "status": "completed"} tags=[]
phenomexcan_input_file_list = [
    f
    for f in INPUT_DATA_DIR.glob("*.pkl")
    if f.name.startswith(("smultixcan-", "spredixcan-"))
]

# %% papermill={"duration": 0.02244, "end_time": "2020-12-23T19:50:42.392015", "exception": false, "start_time": "2020-12-23T19:50:42.369575", "status": "completed"} tags=[]
display(len(phenomexcan_input_file_list))

# %% [markdown] papermill={"duration": 0.012838, "end_time": "2020-12-23T19:50:42.417450", "exception": false, "start_time": "2020-12-23T19:50:42.404612", "status": "completed"} tags=[]
# # Predict drug-disease associations

# %% papermill={"duration": 0.623517, "end_time": "2020-12-23T19:50:43.053428", "exception": false, "start_time": "2020-12-23T19:50:42.429911", "status": "completed"} tags=[]
from drug_disease import (
    predict_dotprod,
    predict_dotprod_neg,
    predict_pearson,
    predict_pearson_neg,
    predict_spearman,
    predict_spearman_neg,
)

# %% papermill={"duration": 4836.090114, "end_time": "2020-12-23T21:11:19.157026", "exception": false, "start_time": "2020-12-23T19:50:43.066912", "status": "completed"} tags=[]
for phenomexcan_input_file in phenomexcan_input_file_list:
    print(phenomexcan_input_file.name)

    # read phenomexcan data
    phenomexcan_projection = pd.read_pickle(phenomexcan_input_file)
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

# %% papermill={"duration": 0.211738, "end_time": "2020-12-23T21:11:19.579207", "exception": false, "start_time": "2020-12-23T21:11:19.367469", "status": "completed"} tags=[]
