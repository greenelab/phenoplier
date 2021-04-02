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
# **COMPLETE**: projects S-PrediXcan results on all tissues.

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
from IPython.display import display
from pathlib import Path

import pandas as pd

import conf
from multiplier import MultiplierProjection

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

# %% tags=[]
OUTPUT_RAW_DATA_DIR = Path(OUTPUT_DATA_DIR, "raw")
display(OUTPUT_RAW_DATA_DIR)
OUTPUT_RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# %% tags=[]
OUTPUT_PROJ_DATA_DIR = Path(OUTPUT_DATA_DIR, "proj")
display(OUTPUT_PROJ_DATA_DIR)
OUTPUT_PROJ_DATA_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown] tags=[]
# # Load PhenomeXcan results

# %% tags=[]
input_file_list = [
    conf.PHENOMEXCAN["SMULTIXCAN_MASHR_ZSCORES_FILE"],
    Path(
        conf.PHENOMEXCAN["SPREDIXCAN_MASHR_ZSCORES_FOLDER"],
        "most_signif",
        "spredixcan-most_signif.pkl",
    ),
]

# %% tags=[]
# add S-PrediXcan results for each tissue
input_file_list = input_file_list + [
    f
    for f in Path(conf.PHENOMEXCAN["SPREDIXCAN_MASHR_ZSCORES_FOLDER"], "pkl").glob(
        "*.pkl"
    )
]

# %% tags=[]
_tmp = len(input_file_list)
display(_tmp)
assert _tmp == 51

# %% tags=[]
for input_file in input_file_list:
    print(input_file.name)

    # read data
    phenomexcan_data = pd.read_pickle(input_file)
    print(f"  shape: {phenomexcan_data.shape}")

    assert phenomexcan_data.index.is_unique
    assert phenomexcan_data.columns.is_unique

    phenomexcan_data = phenomexcan_data.dropna(how="any")
    print(f"  shape (no NaN): {phenomexcan_data.shape}")
    assert not phenomexcan_data.isna().any().any()

    print("  standardizing")
    phenomexcan_data = phenomexcan_data.apply(lambda x: x / x.abs().sum())
    
    output_file = Path(OUTPUT_RAW_DATA_DIR, f"{input_file.stem}-data.pkl").resolve()
    print(f"  saving to: {str(output_file)}")
    phenomexcan_data.to_pickle(output_file)

    # project
    print("  projecting...")
    mproj = MultiplierProjection()
    phenomexcan_projection = mproj.transform(phenomexcan_data)
    print(f"    shape: {phenomexcan_projection.shape}")

    output_file = Path(
        OUTPUT_PROJ_DATA_DIR, f"{input_file.stem}-projection.pkl"
    ).resolve()
    print(f"    saving to: {str(output_file)}")
    phenomexcan_projection.to_pickle(output_file)

    print("")

# %% tags=[]
