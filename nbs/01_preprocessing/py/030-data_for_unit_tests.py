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

# %% [markdown] papermill={"duration": 0.047347, "end_time": "2020-12-14T21:24:39.706211", "exception": false, "start_time": "2020-12-14T21:24:39.658864", "status": "completed"} tags=[]
# # Description

# %% [markdown]
# TODO

# %% [markdown] papermill={"duration": 0.031906, "end_time": "2020-12-14T21:24:39.770133", "exception": false, "start_time": "2020-12-14T21:24:39.738227", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.02319, "end_time": "2020-12-14T21:24:39.807557", "exception": false, "start_time": "2020-12-14T21:24:39.784367", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 0.36702, "end_time": "2020-12-14T21:24:40.187800", "exception": false, "start_time": "2020-12-14T21:24:39.820780", "status": "completed"} tags=[]
from IPython.display import display

# import numpy as np
import pandas as pd

import conf
from data.cache import read_data

# from entity import Trait

# %% [markdown] papermill={"duration": 0.01365, "end_time": "2020-12-14T21:24:40.215577", "exception": false, "start_time": "2020-12-14T21:24:40.201927", "status": "completed"} tags=[]
# # Load S-MultiXcan results

# %% papermill={"duration": 0.301476, "end_time": "2020-12-14T21:24:40.530059", "exception": false, "start_time": "2020-12-14T21:24:40.228583", "status": "completed"} tags=[]
smultixcan_zscores = read_data(conf.PHENOMEXCAN["SMULTIXCAN_MASHR_ZSCORES_FILE"])

# %% papermill={"duration": 0.028251, "end_time": "2020-12-14T21:24:40.573779", "exception": false, "start_time": "2020-12-14T21:24:40.545528", "status": "completed"} tags=[]
smultixcan_zscores.shape

# %% papermill={"duration": 0.037367, "end_time": "2020-12-14T21:24:40.624901", "exception": false, "start_time": "2020-12-14T21:24:40.587534", "status": "completed"} tags=[]
smultixcan_zscores.head()

# %% [markdown] papermill={"duration": 0.013405, "end_time": "2020-12-14T21:24:40.652569", "exception": false, "start_time": "2020-12-14T21:24:40.639164", "status": "completed"} tags=[]
# # Save slices of data for unit testing

# %% papermill={"duration": 0.037367, "end_time": "2020-12-14T21:24:40.624901", "exception": false, "start_time": "2020-12-14T21:24:40.587534", "status": "completed"} tags=[]
smultixcan_zscores.columns[
    smultixcan_zscores.columns.str.lower().str.contains("20002_1499")
]

# %% papermill={"duration": 0.557081, "end_time": "2020-12-14T21:24:41.222794", "exception": false, "start_time": "2020-12-14T21:24:40.665713", "status": "completed"} tags=[]
phenomexcan_fullcode_to_traits = [
    # Traits not mapped to DOID
    "50_raw-Standing_height",
    "20096_1-Size_of_red_wine_glass_drunk_small_125ml",
    # asthma  EFO_0000270
    "20002_1111-Noncancer_illness_code_selfreported_asthma",
    "22127-Doctor_diagnosed_asthma",
    "J45-Diagnoses_main_ICD10_J45_Asthma",
    # hypertension    EFO_0000537
    "20002_1065-Noncancer_illness_code_selfreported_hypertension",
    # primary hypertension    EFO_1002032
    "20002_1072-Noncancer_illness_code_selfreported_essential_hypertension",
    "I10-Diagnoses_main_ICD10_I10_Essential_primary_hypertension",
    # preeclampsia    EFO_0000668
    "O16-Diagnoses_main_ICD10_O16_Unspecified_maternal_hypertension",
    "O14-Diagnoses_main_ICD10_O14_Gestational_pregnancyinduced_hypertension_with_significant_proteinuria",
    # tuberculosis    Orphanet_3389
    "20002_1440-Noncancer_illness_code_selfreported_tuberculosis_tb",
    "22137-Doctor_diagnosed_tuberculosis",
    # labyrinthitis   EFO_0009604 -> this one maps to several DOID: DOID:3930, DOID:1468
    "20002_1499-Noncancer_illness_code_selfreported_labyrinthitis",
]

# %%
smultixcan_slice = smultixcan_zscores.sample(n=10, random_state=0)[
    phenomexcan_fullcode_to_traits
]

# %%
smultixcan_slice.shape

# %%
smultixcan_slice

# %% [markdown] papermill={"duration": 0.01664, "end_time": "2020-12-14T21:24:51.785620", "exception": false, "start_time": "2020-12-14T21:24:51.768980", "status": "completed"} tags=[]
# # Save

# %%
import inspect
from pathlib import Path

from entity import Trait

# %%
output_folder = Path(
    Path(inspect.getfile(Trait)).parent.parent,
    "tests",
    "test_cases",
    "smultixcan_zscores",
).resolve()
output_folder.mkdir(exist_ok=True, parents=True)

display(output_folder)

# %%
output_file = Path(output_folder, f"smultixcan-slice01.pkl").resolve()

display(output_file)

# %% papermill={"duration": 0.027261, "end_time": "2020-12-14T21:24:51.829640", "exception": false, "start_time": "2020-12-14T21:24:51.802379", "status": "completed"} tags=[]
smultixcan_slice.to_pickle(output_file)

# %% papermill={"duration": 0.01856, "end_time": "2020-12-14T21:31:28.294646", "exception": false, "start_time": "2020-12-14T21:31:28.276086", "status": "completed"} tags=[]
