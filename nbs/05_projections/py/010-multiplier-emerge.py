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
# It projects eMERGE S-MultiXcan results into the MultiPLIER latent space.

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[] trusted=true
# %load_ext autoreload
# %autoreload 2

# %% tags=[] trusted=true
from pathlib import Path

from IPython.display import display
import pandas as pd

import conf
from data.cache import read_data
from multiplier import MultiplierProjection
from entity import Gene

# %% [markdown] tags=[]
# # Settings

# %% tags=[] trusted=true
RESULTS_PROJ_OUTPUT_DIR = Path(conf.RESULTS["PROJECTIONS_DIR"])

RESULTS_PROJ_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_PROJ_OUTPUT_DIR)

# %% [markdown] tags=[]
# # Load eMERGE data (S-MultiXcan)

# %% tags=[] trusted=true
smultixcan_results_filename = conf.EMERGE["SMULTIXCAN_MASHR_ZSCORES_FILE"]

display(smultixcan_results_filename)

# %% tags=[] trusted=true
results_filename_stem = smultixcan_results_filename.stem
display(results_filename_stem)

# %% tags=[] trusted=true
smultixcan_results = pd.read_pickle(smultixcan_results_filename)

# %% tags=[] trusted=true
smultixcan_results.shape

# %% tags=[] trusted=true
smultixcan_results.head()

# %% [markdown] tags=[]
# ## Gene IDs to Gene names

# %% tags=[] trusted=true
smultixcan_results = smultixcan_results.rename(index=Gene.GENE_ID_TO_NAME_MAP)

# %% tags=[] trusted=true
smultixcan_results.shape

# %% tags=[] trusted=true
smultixcan_results.head()

# %% [markdown] tags=[]
# ## Remove duplicated gene entries

# %% tags=[] trusted=true
smultixcan_results.index[smultixcan_results.index.duplicated(keep="first")]

# %% tags=[] trusted=true
smultixcan_results = smultixcan_results.loc[
    ~smultixcan_results.index.duplicated(keep="first")
]

# %% tags=[] trusted=true
smultixcan_results.shape

# %% [markdown] tags=[]
# ## Remove NaN values

# %% [markdown] tags=[]
# **TODO**: it might be better to try to impute these values

# %% tags=[] trusted=true
smultixcan_results = smultixcan_results.dropna(how="any")

# %% tags=[] trusted=true
smultixcan_results.shape

# %% [markdown] tags=[]
# # Project S-MultiXcan data into MultiPLIER latent space

# %% tags=[] trusted=true
mproj = MultiplierProjection()

# %% tags=[] trusted=true
smultixcan_into_multiplier = mproj.transform(smultixcan_results)

# %% tags=[] trusted=true
smultixcan_into_multiplier.shape

# %% tags=[] trusted=true
smultixcan_into_multiplier.head()

# %% [markdown] tags=[]
# # Quick analysis

# %% tags=[] trusted=true
(smultixcan_into_multiplier.loc["LV603"].sort_values(ascending=False).head(20))

# %% tags=[] trusted=true
(smultixcan_into_multiplier.loc["LV136"].sort_values(ascending=False).head(20))

# %% [markdown] tags=[]
# # Save

# %% tags=[] trusted=true
output_file = Path(
    RESULTS_PROJ_OUTPUT_DIR, f"projection-{results_filename_stem}.pkl"
).resolve()

display(output_file)

# %% tags=[] trusted=true
smultixcan_into_multiplier.to_pickle(output_file)

# %% tags=[]
