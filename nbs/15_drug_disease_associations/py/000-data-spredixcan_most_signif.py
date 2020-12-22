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
# **TODO**: should probably be moved to preprocessing folder.

# %% [markdown] papermill={"duration": 0.011799, "end_time": "2020-12-18T22:38:21.422136", "exception": false, "start_time": "2020-12-18T22:38:21.410337", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.022467, "end_time": "2020-12-18T22:38:21.456126", "exception": false, "start_time": "2020-12-18T22:38:21.433659", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 0.192251, "end_time": "2020-12-18T22:38:21.659996", "exception": false, "start_time": "2020-12-18T22:38:21.467745", "status": "completed"} tags=[]
from pathlib import Path

# import numpy as np
import pandas as pd

import conf
from multiplier import MultiplierProjection
# from entity import Gene
# from data.cache import read_data
# from data.hdf5 import simplify_trait_fullcode, HDF5_FILE_PATTERN

# %% [markdown] papermill={"duration": 0.011416, "end_time": "2020-12-18T22:38:21.683356", "exception": false, "start_time": "2020-12-18T22:38:21.671940", "status": "completed"} tags=[]
# # Settings

# %%
OUTPUT_DIR = conf.RESULTS["DRUG_DISEASE_ANALYSES"]
display(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
OUTPUT_DATA_DIR = Path(OUTPUT_DIR, "data")
display(OUTPUT_DATA_DIR)
OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# # Load S-PrediXcan results

# %%
input_file = Path(
    conf.PHENOMEXCAN["SPREDIXCAN_MASHR_ZSCORES_FOLDER"],
    "most_signif",
    "spredixcan-most_signif.pkl"
).resolve()

display(input_file)

# %%
phenomexcan_data = pd.read_pickle(input_file)

# %%
phenomexcan_data.shape

# %%
phenomexcan_data.head()

# %%
assert phenomexcan_data.index.is_unique

# %%
assert phenomexcan_data.columns.is_unique

# %% [markdown]
# ## Remove NaN values

# %%
phenomexcan_data = phenomexcan_data.dropna(how='any')

# %%
phenomexcan_data.shape

# %%
assert not phenomexcan_data.isna().any().any()

# %% [markdown]
# ## Save

# %%
output_file = Path(
    OUTPUT_DATA_DIR,
    "spredixcan-most_signif.pkl"
).resolve()
display(output_file)

# %%
phenomexcan_data.to_pickle(output_file)

# %% [markdown]
# # Project into MultiPLIER

# %%
mproj = MultiplierProjection()

# %%
phenomexcan_projection = mproj.transform(phenomexcan_data)

# %%
phenomexcan_projection.shape

# %%
phenomexcan_projection.head()

# %% [markdown]
# ## Save

# %%
output_file = Path(
    OUTPUT_DATA_DIR,
    "spredixcan-most_signif-projection.pkl"
).resolve()
display(output_file)

# %%
phenomexcan_projection.to_pickle(output_file)

# %%
