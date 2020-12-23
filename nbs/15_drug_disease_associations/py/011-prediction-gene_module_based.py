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

# %% [markdown] papermill={"duration": 0.044577, "end_time": "2020-12-18T22:38:21.345879", "exception": false, "start_time": "2020-12-18T22:38:21.301302", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.011764, "end_time": "2020-12-18T22:38:21.398073", "exception": false, "start_time": "2020-12-18T22:38:21.386309", "status": "completed"} tags=[]
# **TODO**

# %% [markdown] papermill={"duration": 0.011799, "end_time": "2020-12-18T22:38:21.422136", "exception": false, "start_time": "2020-12-18T22:38:21.410337", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.022467, "end_time": "2020-12-18T22:38:21.456126", "exception": false, "start_time": "2020-12-18T22:38:21.433659", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 0.192251, "end_time": "2020-12-18T22:38:21.659996", "exception": false, "start_time": "2020-12-18T22:38:21.467745", "status": "completed"} tags=[]
from IPython.display import display
from pathlib import Path

import pandas as pd

import conf

# %% [markdown] papermill={"duration": 0.011416, "end_time": "2020-12-18T22:38:21.683356", "exception": false, "start_time": "2020-12-18T22:38:21.671940", "status": "completed"} tags=[]
# # Settings

# %%
# if True, then it doesn't check if result files already exist and runs everything again
FORCE_RUN = False

# %%
PREDICTION_METHOD = "Module-based"

# %%
OUTPUT_DIR = conf.RESULTS["DRUG_DISEASE_ANALYSES"]
display(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
# OUTPUT_DATA_DIR = Path(OUTPUT_DIR, "data")
# display(OUTPUT_DATA_DIR)
# OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# %%
INPUT_DATA_DIR = Path(
    OUTPUT_DIR,
    "data",
    "proj",
)
display(INPUT_DATA_DIR)
INPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# %%
OUTPUT_PREDICTIONS_DIR = Path(OUTPUT_DIR, "predictions")
display(OUTPUT_PREDICTIONS_DIR)
OUTPUT_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# # Load PharmacotherapyDB gold standard

# %%
gold_standard = pd.read_pickle(
    Path(OUTPUT_DIR, "gold_standard.pkl"),
)

# %%
display(gold_standard.shape)

# %%
display(gold_standard.head())

# %%
doids_in_gold_standard = set(gold_standard["trait"])

# %% [markdown]
# # Load LINCS

# %% [markdown]
# ## Projected data

# %%
input_file = Path(INPUT_DATA_DIR, "lincs-projection.pkl").resolve()

display(input_file)

# %%
lincs_projection = pd.read_pickle(input_file)

# %%
display(lincs_projection.shape)

# %%
display(lincs_projection.head())

# %% [markdown]
# # Load S-PrediXcan

# %%
from entity import Trait

# %%
phenomexcan_input_file_list = [
    f
    for f in INPUT_DATA_DIR.glob("*.pkl")
    if f.name.startswith(("smultixcan-", "spredixcan-"))
]

# %%
display(len(phenomexcan_input_file_list))

# %% [markdown]
# # Predict drug-disease associations

# %%
from drug_disease import predict_dotprod, predict_dotprod_neg

# %%
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

    print("\n")

# %%
