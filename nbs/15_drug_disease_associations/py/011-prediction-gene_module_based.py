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

# %% [markdown] papermill={"duration": 0.04089, "end_time": "2020-12-23T18:05:20.790112", "exception": false, "start_time": "2020-12-23T18:05:20.749222", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.010506, "end_time": "2020-12-23T18:05:20.814922", "exception": false, "start_time": "2020-12-23T18:05:20.804416", "status": "completed"} tags=[]
# **TODO**

# %% [markdown] papermill={"duration": 0.010014, "end_time": "2020-12-23T18:05:20.835300", "exception": false, "start_time": "2020-12-23T18:05:20.825286", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.02428, "end_time": "2020-12-23T18:05:20.869748", "exception": false, "start_time": "2020-12-23T18:05:20.845468", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 0.196929, "end_time": "2020-12-23T18:05:21.077057", "exception": false, "start_time": "2020-12-23T18:05:20.880128", "status": "completed"} tags=[]
from IPython.display import display
from pathlib import Path

import pandas as pd

import conf

# %% [markdown] papermill={"duration": 0.010457, "end_time": "2020-12-23T18:05:21.098286", "exception": false, "start_time": "2020-12-23T18:05:21.087829", "status": "completed"} tags=[]
# # Settings

# %% papermill={"duration": 0.020419, "end_time": "2020-12-23T18:05:21.129046", "exception": false, "start_time": "2020-12-23T18:05:21.108627", "status": "completed"} tags=[]
# if True, then it doesn't check if result files already exist and runs everything again
FORCE_RUN = False

# %% papermill={"duration": 0.020341, "end_time": "2020-12-23T18:05:21.159804", "exception": false, "start_time": "2020-12-23T18:05:21.139463", "status": "completed"} tags=[]
PREDICTION_METHOD = "Module-based"

# %% papermill={"duration": 0.025232, "end_time": "2020-12-23T18:05:21.196196", "exception": false, "start_time": "2020-12-23T18:05:21.170964", "status": "completed"} tags=[]
OUTPUT_DIR = conf.RESULTS["DRUG_DISEASE_ANALYSES"]
display(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% papermill={"duration": 0.023756, "end_time": "2020-12-23T18:05:21.230855", "exception": false, "start_time": "2020-12-23T18:05:21.207099", "status": "completed"} tags=[]
# OUTPUT_DATA_DIR = Path(OUTPUT_DIR, "data")
# display(OUTPUT_DATA_DIR)
# OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# %% papermill={"duration": 0.032363, "end_time": "2020-12-23T18:05:21.275472", "exception": false, "start_time": "2020-12-23T18:05:21.243109", "status": "completed"} tags=[]
INPUT_DATA_DIR = Path(
    OUTPUT_DIR,
    "data",
    "proj",
)
display(INPUT_DATA_DIR)
INPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# %% papermill={"duration": 0.022844, "end_time": "2020-12-23T18:05:21.310394", "exception": false, "start_time": "2020-12-23T18:05:21.287550", "status": "completed"} tags=[]
OUTPUT_PREDICTIONS_DIR = Path(OUTPUT_DIR, "predictions")
display(OUTPUT_PREDICTIONS_DIR)
OUTPUT_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown] papermill={"duration": 0.011243, "end_time": "2020-12-23T18:05:21.333015", "exception": false, "start_time": "2020-12-23T18:05:21.321772", "status": "completed"} tags=[]
# # Load PharmacotherapyDB gold standard

# %% papermill={"duration": 0.021633, "end_time": "2020-12-23T18:05:21.366060", "exception": false, "start_time": "2020-12-23T18:05:21.344427", "status": "completed"} tags=[]
gold_standard = pd.read_pickle(
    Path(OUTPUT_DIR, "gold_standard.pkl"),
)

# %% papermill={"duration": 0.021866, "end_time": "2020-12-23T18:05:21.399127", "exception": false, "start_time": "2020-12-23T18:05:21.377261", "status": "completed"} tags=[]
display(gold_standard.shape)

# %% papermill={"duration": 0.02686, "end_time": "2020-12-23T18:05:21.437229", "exception": false, "start_time": "2020-12-23T18:05:21.410369", "status": "completed"} tags=[]
display(gold_standard.head())

# %% papermill={"duration": 0.021401, "end_time": "2020-12-23T18:05:21.470364", "exception": false, "start_time": "2020-12-23T18:05:21.448963", "status": "completed"} tags=[]
doids_in_gold_standard = set(gold_standard["trait"])

# %% [markdown] papermill={"duration": 0.011701, "end_time": "2020-12-23T18:05:21.493765", "exception": false, "start_time": "2020-12-23T18:05:21.482064", "status": "completed"} tags=[]
# # Load LINCS

# %% [markdown] papermill={"duration": 0.011496, "end_time": "2020-12-23T18:05:21.517009", "exception": false, "start_time": "2020-12-23T18:05:21.505513", "status": "completed"} tags=[]
# ## Projected data

# %% papermill={"duration": 0.022224, "end_time": "2020-12-23T18:05:21.550584", "exception": false, "start_time": "2020-12-23T18:05:21.528360", "status": "completed"} tags=[]
input_file = Path(INPUT_DATA_DIR, "lincs-projection.pkl").resolve()

display(input_file)

# %% papermill={"duration": 0.026689, "end_time": "2020-12-23T18:05:21.589109", "exception": false, "start_time": "2020-12-23T18:05:21.562420", "status": "completed"} tags=[]
lincs_projection = pd.read_pickle(input_file)

# %% papermill={"duration": 0.023879, "end_time": "2020-12-23T18:05:21.625255", "exception": false, "start_time": "2020-12-23T18:05:21.601376", "status": "completed"} tags=[]
display(lincs_projection.shape)

# %% papermill={"duration": 0.03622, "end_time": "2020-12-23T18:05:21.675585", "exception": false, "start_time": "2020-12-23T18:05:21.639365", "status": "completed"} tags=[]
display(lincs_projection.head())

# %% [markdown] papermill={"duration": 0.012484, "end_time": "2020-12-23T18:05:21.700768", "exception": false, "start_time": "2020-12-23T18:05:21.688284", "status": "completed"} tags=[]
# # Load S-PrediXcan

# %% papermill={"duration": 0.225438, "end_time": "2020-12-23T18:05:21.938669", "exception": false, "start_time": "2020-12-23T18:05:21.713231", "status": "completed"} tags=[]
from entity import Trait

# %% papermill={"duration": 0.025545, "end_time": "2020-12-23T18:05:21.979243", "exception": false, "start_time": "2020-12-23T18:05:21.953698", "status": "completed"} tags=[]
phenomexcan_input_file_list = [
    f
    for f in INPUT_DATA_DIR.glob("*.pkl")
    if f.name.startswith(("smultixcan-", "spredixcan-"))
]

# %% papermill={"duration": 0.023138, "end_time": "2020-12-23T18:05:22.016221", "exception": false, "start_time": "2020-12-23T18:05:21.993083", "status": "completed"} tags=[]
display(len(phenomexcan_input_file_list))

# %% [markdown] papermill={"duration": 0.01245, "end_time": "2020-12-23T18:05:22.042111", "exception": false, "start_time": "2020-12-23T18:05:22.029661", "status": "completed"} tags=[]
# # Predict drug-disease associations

# %% papermill={"duration": 0.028271, "end_time": "2020-12-23T18:05:22.083712", "exception": false, "start_time": "2020-12-23T18:05:22.055441", "status": "completed"} tags=[]
from drug_disease import (
    predict_dotprod,
    predict_dotprod_neg,
    predict_pearson,
    predict_pearson_neg,
    predict_spearman,
    predict_spearman_neg,
)

# %% papermill={"duration": 2056.463755, "end_time": "2020-12-23T18:39:38.560240", "exception": false, "start_time": "2020-12-23T18:05:22.096485", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.112221, "end_time": "2020-12-23T18:39:38.786608", "exception": false, "start_time": "2020-12-23T18:39:38.674387", "status": "completed"} tags=[]
