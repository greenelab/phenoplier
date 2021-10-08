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
# This notebook exports results into other more accessible data formats.

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

import conf

# %%
readRDS = ro.r["readRDS"]

# %% tags=[]
saveRDS = ro.r["saveRDS"]

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
OUTPUT_DIR = conf.RESULTS["GLS"]
display(OUTPUT_DIR)

assert OUTPUT_DIR.exists()

# %% [markdown] tags=[]
# # Get results files

# %%
input_filepath = OUTPUT_DIR / "gls_phenotypes-combined-phenomexcan.pkl"
display(input_filepath)

# %%
data = pd.read_pickle(input_filepath)

# %% tags=[]
data.shape

# %% tags=[]
data.head()

# %%
assert not data.isna().any().any()

# %% [markdown] tags=[]
# # Save

# %% [markdown] tags=[]
# ## RDS format

# %% tags=[]
output_file = input_filepath.with_suffix(".rds")
display(output_file)

# %%
with localconverter(ro.default_converter + pandas2ri.converter):
    data_r = ro.conversion.py2rpy(data)

# %%
data_r

# %% tags=[]
saveRDS(data_r, str(output_file))

# %%
# testing: load the rds file again
data_r = readRDS(str(output_file))

# %%
with localconverter(ro.default_converter + pandas2ri.converter):
    data_again = ro.conversion.rpy2py(data_r)
    data_again.index = data_again.index.astype(int)

# %%
data_again.shape

# %%
data_again.head()

# %%
pd.testing.assert_frame_equal(
    data,
    data_again,
    check_dtype=False,
)

# %% [markdown] tags=[]
# ## Text format

# %% tags=[]
# tsv format
output_file = input_filepath.with_suffix(".tsv.gz")
display(output_file)

# %% tags=[]
data.to_csv(output_file, sep="\t", index=False, float_format="%.5e")

# %%
# testing
data2 = data.copy()
data2.index = list(range(0, data2.shape[0]))

data_again = pd.read_csv(output_file, sep="\t")

data_again.index = list(data_again.index)
data_again["part_k"] = data_again["part_k"].astype(float)

# %%
data_again.shape

# %%
data_again.head()

# %%
pd.testing.assert_frame_equal(
    data2,
    data_again,
    check_dtype=False,
)

# %%
