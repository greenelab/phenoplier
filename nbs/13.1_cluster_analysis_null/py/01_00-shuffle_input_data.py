# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
#     formats: ipynb,py//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] tags=[]
# # Description

# %% [markdown] tags=[]
# **UPDATED for nulls**:
# * Shuffle the projection of S-MultiXcan into LVs
# * Then run z-scores, PCA and UMAP on this.
# * And then continue all the clustering pipeline from base clusterings to consensus clustering
# * Then create clustering tree.
#
# It projects the PhenomeXcan results (S-MultiXcan, EFO version) into the MultiPLIER latent space.
# Before projecting, repeated gene symbols as well as genes with NaN are removed;
# additionally (also before projecting), S-MultiXcan results are adjusted for highly polygenic traits.

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
from pathlib import Path

from IPython.display import display
import numpy as np
from scipy import stats
import pandas as pd
import pytest

# import rpy2.robjects as ro
# from rpy2.robjects import pandas2ri
# from rpy2.robjects.conversion import localconverter

import conf

# from entity import Gene
# from data.cache import read_data
# from multiplier import MultiplierProjection

# %% tags=[]
# readRDS = ro.r["readRDS"]

# %% tags=[]
# saveRDS = ro.r["saveRDS"]

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
RESULTS_PROJ_OUTPUT_DIR = Path(conf.RESULTS["PROJECTIONS_DIR"])
RESULTS_PROJ_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_PROJ_OUTPUT_DIR)

# %%
OUTPUT_DIR = Path(
    conf.RESULTS["CLUSTERING_NULL_DIR"],
    "projections",
).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

display(OUTPUT_DIR)

# %%
rs = np.random.RandomState(0)

# %% [markdown] tags=[]
# # Load projection of S-MultiXcan into LV space

# %% tags=[]
smultixcan_results_filename = conf.PHENOMEXCAN[
    "SMULTIXCAN_EFO_PARTIAL_MASHR_ZSCORES_FILE"
]

display(smultixcan_results_filename)
assert smultixcan_results_filename.exists()

# %% tags=[]
results_filename_stem = smultixcan_results_filename.stem
display(results_filename_stem)

# %%
input_file = Path(
    RESULTS_PROJ_OUTPUT_DIR, f"projection-{results_filename_stem}.pkl"
).resolve()

display(input_file)

# %%
projected_data = pd.read_pickle(input_file)

# %%
projected_data.shape

# %%
projected_data.head()

# %% [markdown] tags=[]
# # Shuffle projected data

# %%
shuffled_projected_data = projected_data.apply(
    lambda x: x.sample(frac=1, random_state=rs).to_numpy()
)

# %%
shuffled_projected_data.shape

# %%
shuffled_projected_data.head()

# %% [markdown]
# ## Testing

# %%
assert stats.pearsonr(projected_data.loc["LV1"], projected_data.loc["LV1"])[
    0
] == pytest.approx(1.0)
assert stats.pearsonr(
    shuffled_projected_data.loc["LV1"], shuffled_projected_data.loc["LV1"]
)[0] == pytest.approx(1.0)

# %%
_tmp = stats.pearsonr(shuffled_projected_data.loc["LV1"], projected_data.loc["LV1"])
display(_tmp)
assert _tmp[0] == pytest.approx(0.0, rel=0, abs=0.01)

# %% [markdown] tags=[]
# # Quick analysis

# %% [markdown] tags=[]
# Ensure we broke known relationships

# %% tags=[]
(shuffled_projected_data.loc["LV603"].sort_values(ascending=False).head(20))

# %% tags=[]
(shuffled_projected_data.loc["LV136"].sort_values(ascending=False).head(20))

# %% tags=[]
(shuffled_projected_data.loc["LV844"].sort_values(ascending=False).head(20))

# %% tags=[]
(shuffled_projected_data.loc["LV246"].sort_values(ascending=False).head(20))

# %% [markdown] tags=[]
# # Save

# %% tags=[]
output_file = Path(OUTPUT_DIR, f"projection-{results_filename_stem}.pkl").resolve()

display(output_file)

# %% tags=[]
shuffled_projected_data.to_pickle(output_file)

# %% tags=[]
