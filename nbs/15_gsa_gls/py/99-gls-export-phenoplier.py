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
# This notebook exports results into other more accessible data formats.

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
from pathlib import Path
import shutil

import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

import conf
from utils import remove_all_file_extensions

# %% tags=[]
readRDS = ro.r["readRDS"]

# %% tags=[]
saveRDS = ro.r["saveRDS"]

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
OUTPUT_DIR = Path(conf.__file__).parent.parent / "data" / "gls"
display(OUTPUT_DIR)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# %% [markdown] tags=[]
# # Get results files

# %% tags=[]
input_filepath = conf.RESULTS["GLS"] / "gls-summary-phenomexcan.pkl.gz"
display(input_filepath)

# %% tags=[]
data = pd.read_pickle(input_filepath)

# %% tags=[]
data.shape

# %% tags=[]
data.head()

# %% tags=[]
assert not data.isna().any().any()

# %% [markdown] tags=[]
# # Save

# %% [markdown] tags=[]
# ## Pickle format

# %% tags=[]
shutil.copy(input_filepath, OUTPUT_DIR)

# %% [markdown] tags=[]
# ## RDS format

# %% tags=[]
output_file = remove_all_file_extensions(input_filepath).with_suffix(".rds")
output_file = OUTPUT_DIR / output_file.name
display(output_file)

# %% tags=[]
with localconverter(ro.default_converter + pandas2ri.converter):
    data_r = ro.conversion.py2rpy(data)

# %% tags=[]
data_r

# %% tags=[]
saveRDS(data_r, str(output_file))

# %% tags=[]
# testing: load the rds file again
data_r = readRDS(str(output_file))

# %% tags=[]
with localconverter(ro.default_converter + pandas2ri.converter):
    data_again = ro.conversion.rpy2py(data_r)
    data_again.index = data_again.index.astype(int)

# %% tags=[]
data_again.shape

# %% tags=[]
data_again.head()

# %% tags=[]
pd.testing.assert_frame_equal(
    data,
    data_again,
    check_dtype=False,
)

# %% [markdown] tags=[]
# ## Text format

# %% tags=[]
# tsv format
output_file = remove_all_file_extensions(input_filepath).with_suffix(".tsv.gz")
output_file = OUTPUT_DIR / output_file.name
display(output_file)

# %% tags=[]
data.to_csv(output_file, sep="\t", index=False, float_format="%.5e")

# %% tags=[]
# testing
data2 = data.copy()
data2.index = list(range(0, data2.shape[0]))

data_again = pd.read_csv(output_file, sep="\t")
data_again.index = list(data_again.index)

# %% tags=[]
data_again.shape

# %% tags=[]
data_again.head()

# %% tags=[]
pd.testing.assert_frame_equal(
    data2,
    data_again,
    check_categorical=False,
    check_dtype=False,
)

# %% tags=[]
