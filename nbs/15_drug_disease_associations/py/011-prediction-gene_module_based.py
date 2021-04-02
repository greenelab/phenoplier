# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
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

# %% [markdown] tags=[]
# # Description

# %% [markdown] tags=[]
# **TODO**

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
from IPython.display import display
from pathlib import Path

import pandas as pd

import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
# if True, then it doesn't check if result files already exist and runs everything again
FORCE_RUN = True

# %% tags=[]
PREDICTION_METHOD = "Module-based"

# %% tags=[]
OUTPUT_DIR = conf.RESULTS["DRUG_DISEASE_ANALYSES"]
display(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% tags=[]
# OUTPUT_DATA_DIR = Path(OUTPUT_DIR, "data")
# display(OUTPUT_DATA_DIR)
# OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# %% tags=[]
INPUT_DATA_DIR = Path(
    OUTPUT_DIR,
    "data",
    "proj",
)
display(INPUT_DATA_DIR)
INPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# %% tags=[]
OUTPUT_PREDICTIONS_DIR = Path(OUTPUT_DIR, "predictions")
display(OUTPUT_PREDICTIONS_DIR)
OUTPUT_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown] tags=[]
# # Load PharmacotherapyDB gold standard

# %% tags=[]
gold_standard = pd.read_pickle(
    Path(OUTPUT_DIR, "gold_standard.pkl"),
)

# %% tags=[]
display(gold_standard.shape)

# %% tags=[]
display(gold_standard.head())

# %% tags=[]
doids_in_gold_standard = set(gold_standard["trait"])

# %% [markdown] tags=[]
# # Load LINCS

# %% [markdown] tags=[]
# ## Projected data

# %% tags=[]
input_file = Path(INPUT_DATA_DIR, "lincs-projection.pkl").resolve()

display(input_file)

# %% tags=[]
lincs_projection = pd.read_pickle(input_file)

# %% tags=[]
display(lincs_projection.shape)

# %% tags=[]
display(lincs_projection.head())

# %% [markdown] tags=[]
# # Load S-PrediXcan

# %% tags=[]
from entity import Trait

# %% tags=[]
phenomexcan_input_file_list = [
    f
    for f in INPUT_DATA_DIR.glob("*.pkl")
    if f.name.startswith("spredixcan-")
    #     if f.name.startswith(("smultixcan-", "spredixcan-"))
]

# %% tags=[]
display(len(phenomexcan_input_file_list))

# %%
pd.read_pickle(phenomexcan_input_file_list[10]).head()

# %% [markdown] tags=[]
# # Predict drug-disease associations

# %% tags=[]
from drug_disease import (
#     predict_dotprod,
    predict_dotprod_neg,
#     predict_pearson,
#     predict_pearson_neg,
#     predict_spearman,
#     predict_spearman_neg,
)

# %% tags=[]
for phenomexcan_input_file in phenomexcan_input_file_list:
    print(phenomexcan_input_file.name)

    # read phenomexcan data
    phenomexcan_projection = pd.read_pickle(phenomexcan_input_file)
    print(f"  shape: {phenomexcan_projection.shape}")

    #     predict_dotprod(
    #         lincs_projection,
    #         phenomexcan_input_file,
    #         phenomexcan_projection,
    #         OUTPUT_PREDICTIONS_DIR,
    #         PREDICTION_METHOD,
    #         doids_in_gold_standard,
    #         FORCE_RUN,
    #     )

    predict_dotprod_neg(
        lincs_projection,
        phenomexcan_input_file,
        phenomexcan_projection,
        OUTPUT_PREDICTIONS_DIR,
        PREDICTION_METHOD,
        doids_in_gold_standard,
        FORCE_RUN,
    )

    #     predict_pearson(
    #         lincs_projection,
    #         phenomexcan_input_file,
    #         phenomexcan_projection,
    #         OUTPUT_PREDICTIONS_DIR,
    #         PREDICTION_METHOD,
    #         doids_in_gold_standard,
    #         FORCE_RUN,
    #     )

    #     predict_pearson_neg(
    #         lincs_projection,
    #         phenomexcan_input_file,
    #         phenomexcan_projection,
    #         OUTPUT_PREDICTIONS_DIR,
    #         PREDICTION_METHOD,
    #         doids_in_gold_standard,
    #         FORCE_RUN,
    #     )

    #     predict_spearman(
    #         lincs_projection,
    #         phenomexcan_input_file,
    #         phenomexcan_projection,
    #         OUTPUT_PREDICTIONS_DIR,
    #         PREDICTION_METHOD,
    #         doids_in_gold_standard,
    #         FORCE_RUN,
    #     )

    #     predict_spearman_neg(
    #         lincs_projection,
    #         phenomexcan_input_file,
    #         phenomexcan_projection,
    #         OUTPUT_PREDICTIONS_DIR,
    #         PREDICTION_METHOD,
    #         doids_in_gold_standard,
    #         FORCE_RUN,
    #     )

    print("\n")

# %% tags=[]
