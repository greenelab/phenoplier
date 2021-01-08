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

# %% [markdown]
# It projects the PhenomeXcan results (EFO version) into the MultiPLIER latent space.

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
from pathlib import Path

from IPython.display import display
import pandas as pd

import conf
from data.cache import read_data
from multiplier import MultiplierProjection

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
# The percentile name indicates the top percentage of genes retained
PERCENTILE_NAME = "pALL"

display(PERCENTILE_NAME)

# %% tags=[]
RESULTS_PROJ_OUTPUT_DIR = Path(conf.RESULTS["PROJECTIONS_DIR"])

RESULTS_PROJ_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_PROJ_OUTPUT_DIR)

# %% [markdown] tags=[]
# # Read gene mappings

# %% tags=[]
GENE_ID_TO_NAME_MAP = read_data(conf.PHENOMEXCAN["GENE_MAP_ID_TO_NAME"])
GENE_NAME_TO_ID_MAP = read_data(conf.PHENOMEXCAN["GENE_MAP_NAME_TO_ID"])

# %% [markdown] tags=[]
# # Load PhenomeXcan data (S-MultiXcan)

# %% tags=[]
smultixcan_results_filename = conf.PHENOMEXCAN[
    "SMULTIXCAN_EFO_PARTIAL_MASHR_ZSCORES_FILE"
]

display(smultixcan_results_filename)

# %% tags=[]
results_filename_stem = smultixcan_results_filename.stem
display(results_filename_stem)

# %% tags=[]
smultixcan_results = pd.read_pickle(smultixcan_results_filename)

# %% tags=[]
smultixcan_results.shape

# %% tags=[]
smultixcan_results.head()

# %% [markdown] tags=[]
# ## Gene IDs to Gene names

# %% tags=[]
smultixcan_results = smultixcan_results.rename(index=GENE_ID_TO_NAME_MAP)

# %% tags=[]
smultixcan_results.shape

# %% tags=[]
smultixcan_results.head()

# %% [markdown] tags=[]
# ## Remove duplicated gene entries

# %% tags=[]
smultixcan_results.index[smultixcan_results.index.duplicated(keep="first")]

# %% tags=[]
smultixcan_results = smultixcan_results.loc[
    ~smultixcan_results.index.duplicated(keep="first")
]

# %% tags=[]
smultixcan_results.shape

# %% [markdown] tags=[]
# ## Remove NaN values

# %% [markdown] tags=[]
# **TODO**: it might be better to try to impute this values

# %% tags=[]
smultixcan_results = smultixcan_results.dropna(how="any")

# %% tags=[]
smultixcan_results.shape

# %% [markdown] tags=[]
# # Project S-MultiXcan data into MultiPLIER latent space

# %% tags=[]
mproj = MultiplierProjection()

# %% tags=[]
smultixcan_into_multiplier = mproj.transform(smultixcan_results)

# %% tags=[]
smultixcan_into_multiplier.shape

# %% tags=[]
smultixcan_into_multiplier.head()

# %% [markdown] tags=[]
# # Quick analysis

# %% tags=[]
(smultixcan_into_multiplier.loc["LV603"].sort_values(ascending=False).head(20))

# %% tags=[]
(smultixcan_into_multiplier.loc["LV136"].sort_values(ascending=False).head(20))

# %% [markdown] tags=[]
# # Save

# %% tags=[]
output_file = Path(
    RESULTS_PROJ_OUTPUT_DIR, f"projection-{results_filename_stem}.pkl"
).resolve()

display(output_file)

# %% tags=[]
smultixcan_into_multiplier.to_pickle(output_file)

# %% tags=[]
