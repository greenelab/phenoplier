# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill
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
# It makes sure that the data directory for predictions is empty before these notebooks are run.
#
# Do I need this?

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
from pathlib import Path

import numpy as np
import pandas as pd

import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
OUTPUT_DIR = conf.RESULTS["DRUG_DISEASE_ANALYSES"]
display(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% tags=[]
OUTPUT_DATA_DIR = Path(OUTPUT_DIR, "data")
display(OUTPUT_DATA_DIR)
OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown] tags=[]
# # Remove previous predictions

# %% tags=[]
current_prediction_files = OUTPUT_DATA_DIR.rglob("*.pkl")

# %% tags=[]
for f in current_prediction_files:
    display(f)
    f.unlink()

# %% tags=[]
