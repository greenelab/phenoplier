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
# It projects the PhenomeXcan results (S-MultiXcan) into the MultiPLIER latent space.
# Before projecting, repeated gene symbols as well as genes with NaN are removed.

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
from pathlib import Path

from IPython.display import display
import pandas as pd

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

import conf
from entity import Gene
from data.cache import read_data
from multiplier import MultiplierProjection

# %%
readRDS = ro.r["readRDS"]

# %% tags=[]
saveRDS = ro.r["saveRDS"]

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
RESULTS_PROJ_OUTPUT_DIR = Path(conf.RESULTS["PROJECTIONS_DIR"])
RESULTS_PROJ_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_PROJ_OUTPUT_DIR)

# %% [markdown] tags=[]
# # Load PhenomeXcan data (S-MultiXcan)

# %% tags=[]
smultixcan_results_filename = conf.PHENOMEXCAN["SMULTIXCAN_MASHR_ZSCORES_FILE"]

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
smultixcan_results = smultixcan_results.rename(index=Gene.GENE_ID_TO_NAME_MAP)

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
# **TODO**: it might be better to try to impute these values

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

# %% tags=[]
(smultixcan_into_multiplier.loc["LV844"].sort_values(ascending=False).head(20))

# %% [markdown] tags=[]
# # Save

# %% tags=[]
output_file = Path(
    RESULTS_PROJ_OUTPUT_DIR, f"projection-{results_filename_stem}.pkl"
).resolve()

display(output_file)

# %% tags=[]
smultixcan_into_multiplier.to_pickle(output_file)

# %% [markdown] tags=[]
# ## RDS format

# %% tags=[]
output_rds_file = output_file.with_suffix(".rds")
display(output_rds_file)

# %%
with localconverter(ro.default_converter + pandas2ri.converter):
    data_r = ro.conversion.py2rpy(smultixcan_into_multiplier)

# %%
data_r

# %% tags=[]
saveRDS(data_r, str(output_rds_file))

# %%
# testing: load the rds file again
data_r = readRDS(str(output_rds_file))

# %%
with localconverter(ro.default_converter + pandas2ri.converter):
    data_again = ro.conversion.rpy2py(data_r)
#     data_again.index = data_again.index.astype(int)

# %%
data_again.shape

# %%
data_again.head()

# %%
pd.testing.assert_frame_equal(
    smultixcan_into_multiplier,
    data_again,
    check_names=False,
    check_exact=True,
    #     rtol=0.0,
    #     atol=1e-50,
    #     check_dtype=False,
)

# %% [markdown] tags=[]
# ## Text format

# %% tags=[]
# tsv format
output_text_file = output_file.with_suffix(".tsv.gz")
display(output_text_file)

# %% tags=[]
smultixcan_into_multiplier.to_csv(output_text_file, sep="\t", index=True, float_format="%.5e")

# %%
# testing
# data2 = data.copy()
# data2.index = list(range(0, data2.shape[0]))

data_again = pd.read_csv(output_text_file, sep="\t", index_col="gene_name")

# data_again.index = list(data_again.index)
# data_again["part_k"] = data_again["part_k"].astype(float)

# %%
data_again.shape

# %%
data_again.head()

# %%
pd.testing.assert_frame_equal(
    smultixcan_into_multiplier,
    data_again,
    check_exact=False,
    rtol=0.0,
    atol=5e-5,
)

# %% tags=[]
