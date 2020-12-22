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
from pathlib import Path

import numpy as np
import pandas as pd

import conf

# %% [markdown] papermill={"duration": 0.011416, "end_time": "2020-12-18T22:38:21.683356", "exception": false, "start_time": "2020-12-18T22:38:21.671940", "status": "completed"} tags=[]
# # Settings

# %%
PREDICTION_METHOD="Module-based"

# %%
OUTPUT_DIR = conf.RESULTS["DRUG_DISEASE_ANALYSES"]
display(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
OUTPUT_DATA_DIR = Path(OUTPUT_DIR, "data")
display(OUTPUT_DATA_DIR)
OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# %%
OUTPUT_RAW_DATA_DIR = Path(OUTPUT_DATA_DIR, "raw")
display(OUTPUT_RAW_DATA_DIR)
OUTPUT_RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# %%
OUTPUT_PROJ_DATA_DIR = Path(OUTPUT_DATA_DIR, "proj")
display(OUTPUT_PROJ_DATA_DIR)
OUTPUT_PROJ_DATA_DIR.mkdir(parents=True, exist_ok=True)

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
gold_standard.shape

# %%
gold_standard.head()

# %%
doids_in_gold_standard = set(gold_standard['trait'])

# %% [markdown]
# # Load LINCS

# %% [markdown]
# ## Projected data

# %%
# TODO: hardcoded
input_file = Path(
    OUTPUT_PROJ_DATA_DIR,
    "lincs-projection.pkl"
).resolve()

display(input_file)

# %%
lincs_projection = pd.read_pickle(input_file)

# %%
lincs_projection.shape

# %%
lincs_projection.head()

# %% [markdown]
# # Load S-PrediXcan

# %%
from entity import Trait

# %%
phenomexcan_input_file_list = [
    f for f in OUTPUT_PROJ_DATA_DIR.glob("*.pkl") if f.name.startswith(('smultixcan-', 'spredixcan-'))
]

# %%
display(len(phenomexcan_input_file_list))

# %% [markdown]
# # Predict drug-disease associations

# %%
for phenomexcan_input_file in phenomexcan_input_file_list:
    print(phenomexcan_input_file.name)
    
    # read phenomexcan data
    phenomexcan_projection = pd.read_pickle(phenomexcan_input_file)
    print(f"  shape: {phenomexcan_projection.shape}")
    
    # prediction
    print(f"  predicting...")
    drug_disease_assocs = lincs_projection.T.dot(phenomexcan_projection)
    print(f"    shape: {drug_disease_assocs.shape}")
    drug_disease_assocs = Trait.map_to_doid(drug_disease_assocs, doids_in_gold_standard, combine="max")
    print(f"    shape (after DOID map): {drug_disease_assocs.shape}")
    assert drug_disease_assocs.index.is_unique
    assert drug_disease_assocs.columns.is_unique
    
    # build classifier data
    print(f"  building classifier data...")
    classifier_data = drug_disease_assocs\
        .unstack().reset_index()\
        .rename(columns={'level_0': 'trait', 'perturbagen': 'drug', 0: 'score'})
    assert classifier_data.shape == classifier_data.dropna().shape
    print(f"    shape: {classifier_data.shape}")
    display(classifier_data.describe())
    
    # save
    output_file = Path(
        OUTPUT_PREDICTIONS_DIR,
        f"{phenomexcan_input_file.stem}-prediction_scores.h5"
    ).resolve()
    print(f"    saving to: {str(output_file)}")
    
    classifier_data.to_hdf(output_file, mode="w", complevel=4, key='prediction')
    
    pd.Series({
        'method': PREDICTION_METHOD,
        'data': phenomexcan_input_file.stem,
    })\
    .to_hdf(output_file, mode="r+", key='metadata')
    
    print("")

# %%
