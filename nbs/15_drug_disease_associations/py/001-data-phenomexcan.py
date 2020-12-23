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

# %% [markdown] papermill={"duration": 0.025145, "end_time": "2020-12-23T17:59:44.890969", "exception": false, "start_time": "2020-12-23T17:59:44.865824", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.005927, "end_time": "2020-12-23T17:59:44.903360", "exception": false, "start_time": "2020-12-23T17:59:44.897433", "status": "completed"} tags=[]
# **TODO**: should probably be moved to preprocessing folder.

# %% [markdown] papermill={"duration": 0.005535, "end_time": "2020-12-23T17:59:44.914868", "exception": false, "start_time": "2020-12-23T17:59:44.909333", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.016928, "end_time": "2020-12-23T17:59:44.937412", "exception": false, "start_time": "2020-12-23T17:59:44.920484", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 0.221381, "end_time": "2020-12-23T17:59:45.164612", "exception": false, "start_time": "2020-12-23T17:59:44.943231", "status": "completed"} tags=[]
from IPython.display import display
from pathlib import Path

import pandas as pd

import conf
from multiplier import MultiplierProjection

# %% [markdown] papermill={"duration": 0.005632, "end_time": "2020-12-23T17:59:45.176196", "exception": false, "start_time": "2020-12-23T17:59:45.170564", "status": "completed"} tags=[]
# # Settings

# %% papermill={"duration": 0.020531, "end_time": "2020-12-23T17:59:45.202520", "exception": false, "start_time": "2020-12-23T17:59:45.181989", "status": "completed"} tags=[]
OUTPUT_DIR = conf.RESULTS["DRUG_DISEASE_ANALYSES"]
display(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% papermill={"duration": 0.016809, "end_time": "2020-12-23T17:59:45.225470", "exception": false, "start_time": "2020-12-23T17:59:45.208661", "status": "completed"} tags=[]
OUTPUT_DATA_DIR = Path(OUTPUT_DIR, "data")
display(OUTPUT_DATA_DIR)
OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# %% papermill={"duration": 0.017168, "end_time": "2020-12-23T17:59:45.249232", "exception": false, "start_time": "2020-12-23T17:59:45.232064", "status": "completed"} tags=[]
OUTPUT_RAW_DATA_DIR = Path(OUTPUT_DATA_DIR, "raw")
display(OUTPUT_RAW_DATA_DIR)
OUTPUT_RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# %% papermill={"duration": 0.017076, "end_time": "2020-12-23T17:59:45.273070", "exception": false, "start_time": "2020-12-23T17:59:45.255994", "status": "completed"} tags=[]
OUTPUT_PROJ_DATA_DIR = Path(OUTPUT_DATA_DIR, "proj")
display(OUTPUT_PROJ_DATA_DIR)
OUTPUT_PROJ_DATA_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown] papermill={"duration": 0.006797, "end_time": "2020-12-23T17:59:45.286928", "exception": false, "start_time": "2020-12-23T17:59:45.280131", "status": "completed"} tags=[]
# # Load PhenomeXcan results

# %% papermill={"duration": 0.016576, "end_time": "2020-12-23T17:59:45.310298", "exception": false, "start_time": "2020-12-23T17:59:45.293722", "status": "completed"} tags=[]
input_file_list = [
    conf.PHENOMEXCAN["SMULTIXCAN_MASHR_ZSCORES_FILE"],
    Path(
        conf.PHENOMEXCAN["SPREDIXCAN_MASHR_ZSCORES_FOLDER"],
        "most_signif",
        "spredixcan-most_signif.pkl",
    ),
]

# %% papermill={"duration": 0.017001, "end_time": "2020-12-23T17:59:45.334111", "exception": false, "start_time": "2020-12-23T17:59:45.317110", "status": "completed"} tags=[]
# add S-PrediXcan results for each tissue
input_file_list = input_file_list + [
    f
    for f in Path(conf.PHENOMEXCAN["SPREDIXCAN_MASHR_ZSCORES_FOLDER"], "pkl").glob(
        "*.pkl"
    )
]

# %% papermill={"duration": 0.016575, "end_time": "2020-12-23T17:59:45.357589", "exception": false, "start_time": "2020-12-23T17:59:45.341014", "status": "completed"} tags=[]
_tmp = len(input_file_list)
display(_tmp)
assert _tmp == 51

# %% papermill={"duration": 176.85296, "end_time": "2020-12-23T18:02:42.217692", "exception": false, "start_time": "2020-12-23T17:59:45.364732", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.036917, "end_time": "2020-12-23T18:02:42.293797", "exception": false, "start_time": "2020-12-23T18:02:42.256880", "status": "completed"} tags=[]
