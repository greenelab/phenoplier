# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-trusted
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

# %% [markdown] papermill={"duration": 0.02314, "end_time": "2021-04-23T03:19:24.000632", "exception": false, "start_time": "2021-04-23T03:19:23.977492", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.017745, "end_time": "2021-04-23T03:19:24.037233", "exception": false, "start_time": "2021-04-23T03:19:24.019488", "status": "completed"} tags=[]
# This notebook reads .h5 files generated in PhenomeXcan by [this notebook](https://github.com/hakyimlab/phenomexcan/blob/master/scripts/100_postprocessing/05_spredixcan.ipynb), and saves one file per tissue with the results in pandas DataFrame format (genes in rows, traits in columns). It saves these in two formats: pickle and tsv.gz
#
# The notebook will generate two other folders in the parent of `SPREDIXCAN_H5_FOLDER`: `pkl` and `tsv`
#
# **The idea** is to have the data in a friendly format.

# %% [markdown] papermill={"duration": 0.017783, "end_time": "2021-04-23T03:19:24.072985", "exception": false, "start_time": "2021-04-23T03:19:24.055202", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.03736, "end_time": "2021-04-23T03:19:24.128335", "exception": false, "start_time": "2021-04-23T03:19:24.090975", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 0.306121, "end_time": "2021-04-23T03:19:24.453190", "exception": false, "start_time": "2021-04-23T03:19:24.147069", "status": "completed"} tags=[]
from pathlib import Path

import numpy as np
import pandas as pd

import conf
from data.cache import read_data
from data.hdf5 import simplify_trait_fullcode, HDF5_FILE_PATTERN

# %% [markdown] papermill={"duration": 0.020026, "end_time": "2021-04-23T03:19:24.517072", "exception": false, "start_time": "2021-04-23T03:19:24.497046", "status": "completed"} tags=[]
# # Settings

# %% papermill={"duration": 0.038362, "end_time": "2021-04-23T03:19:24.574495", "exception": false, "start_time": "2021-04-23T03:19:24.536133", "status": "completed"} tags=[]
SPREDIXCAN_H5_FOLDER = Path(
    conf.PHENOMEXCAN["SPREDIXCAN_MASHR_ZSCORES_FOLDER"],
    "hdf5",
)
display(SPREDIXCAN_H5_FOLDER)
assert SPREDIXCAN_H5_FOLDER.is_dir(), "The folder does not exist"

# %% papermill={"duration": 0.028686, "end_time": "2021-04-23T03:19:24.621820", "exception": false, "start_time": "2021-04-23T03:19:24.593134", "status": "completed"} tags=[]
spredixcan_pkl_output_folder = Path(SPREDIXCAN_H5_FOLDER.parent, "pkl").resolve()
spredixcan_pkl_output_folder.mkdir(exist_ok=True, parents=True)
display(spredixcan_pkl_output_folder)

# %% papermill={"duration": 0.029585, "end_time": "2021-04-23T03:19:24.670272", "exception": false, "start_time": "2021-04-23T03:19:24.640687", "status": "completed"} tags=[]
spredixcan_tsv_output_folder = Path(SPREDIXCAN_H5_FOLDER.parent, "tsv").resolve()
spredixcan_tsv_output_folder.mkdir(exist_ok=True, parents=True)
display(spredixcan_tsv_output_folder)

# %% [markdown] papermill={"duration": 0.018906, "end_time": "2021-04-23T03:19:24.708770", "exception": false, "start_time": "2021-04-23T03:19:24.689864", "status": "completed"} tags=[]
# # Read S-PrediXcan results

# %% [markdown] papermill={"duration": 0.018975, "end_time": "2021-04-23T03:19:24.746525", "exception": false, "start_time": "2021-04-23T03:19:24.727550", "status": "completed"} tags=[]
# ## Get list of files

# %% papermill={"duration": 0.028734, "end_time": "2021-04-23T03:19:24.794482", "exception": false, "start_time": "2021-04-23T03:19:24.765748", "status": "completed"} tags=[]
from glob import glob

# %% papermill={"duration": 0.028449, "end_time": "2021-04-23T03:19:24.842153", "exception": false, "start_time": "2021-04-23T03:19:24.813704", "status": "completed"} tags=[]
spredixcan_files = list(SPREDIXCAN_H5_FOLDER.glob("*.h5"))

# %% papermill={"duration": 0.028956, "end_time": "2021-04-23T03:19:24.890480", "exception": false, "start_time": "2021-04-23T03:19:24.861524", "status": "completed"} tags=[]
display(spredixcan_files[:5])
assert len(spredixcan_files) == 49

# %% [markdown] papermill={"duration": 0.01886, "end_time": "2021-04-23T03:19:24.928861", "exception": false, "start_time": "2021-04-23T03:19:24.910001", "status": "completed"} tags=[]
# ## Get tissue name from file list

# %% papermill={"duration": 0.02822, "end_time": "2021-04-23T03:19:24.975888", "exception": false, "start_time": "2021-04-23T03:19:24.947668", "status": "completed"} tags=[]
import re

# %% papermill={"duration": 0.02849, "end_time": "2021-04-23T03:19:25.023490", "exception": false, "start_time": "2021-04-23T03:19:24.995000", "status": "completed"} tags=[]
# some testing
match = re.search(HDF5_FILE_PATTERN, "spredixcan-Esophagus_Muscularis-zscore.h5")
assert match.group("tissue") == "Esophagus_Muscularis"

match = re.search(
    HDF5_FILE_PATTERN, "spredixcan-Brain_Anterior_cingulate_cortex_BA24-zscore.h5"
)
assert match.group("tissue") == "Brain_Anterior_cingulate_cortex_BA24"

# %% [markdown] papermill={"duration": 0.019608, "end_time": "2021-04-23T03:19:25.062918", "exception": false, "start_time": "2021-04-23T03:19:25.043310", "status": "completed"} tags=[]
# # Load S-PrediXcan results

# %% [markdown] papermill={"duration": 0.019041, "end_time": "2021-04-23T03:19:25.100915", "exception": false, "start_time": "2021-04-23T03:19:25.081874", "status": "completed"} tags=[]
# ## Get all PhenomeXcan trait full codes

# %% papermill={"duration": 0.317156, "end_time": "2021-04-23T03:19:25.436949", "exception": false, "start_time": "2021-04-23T03:19:25.119793", "status": "completed"} tags=[]
from entity import Trait

# %% papermill={"duration": 3.537499, "end_time": "2021-04-23T03:19:28.998167", "exception": false, "start_time": "2021-04-23T03:19:25.460668", "status": "completed"} tags=[]
all_phenomexcan_traits = {
    trait_fullcode
    for trait_fullcode in read_data(
        conf.PHENOMEXCAN["SMULTIXCAN_MASHR_ZSCORES_FILE"]
    ).columns
}

# %% papermill={"duration": 0.029812, "end_time": "2021-04-23T03:19:29.064655", "exception": false, "start_time": "2021-04-23T03:19:29.034843", "status": "completed"} tags=[]
display(len(all_phenomexcan_traits))
assert len(all_phenomexcan_traits) == 4091

# %% [markdown] papermill={"duration": 0.018745, "end_time": "2021-04-23T03:19:29.103018", "exception": false, "start_time": "2021-04-23T03:19:29.084273", "status": "completed"} tags=[]
# ## Read all results

# %% papermill={"duration": 18305.907855, "end_time": "2021-04-23T08:24:35.030046", "exception": false, "start_time": "2021-04-23T03:19:29.122191", "status": "completed"} tags=[]
for f_idx, f in enumerate(spredixcan_files):
    f_tissue = re.search(HDF5_FILE_PATTERN, f.name).group("tissue")
    print(f"{f_idx}. {f.name}")

    with pd.HDFStore(f, mode="r") as store:
        traits_keys = list(store.keys())
        assert len(traits_keys) == len(all_phenomexcan_traits)

        store_data = {}

        for trait_fullcode in all_phenomexcan_traits:
            trait_hdf5 = simplify_trait_fullcode(trait_fullcode)

            trait_data = store[trait_hdf5].rename_axis("gene_id")

            store_data[trait_fullcode] = trait_data

        df = pd.DataFrame(store_data)
        assert df.shape[1] == len(all_phenomexcan_traits)
        assert df.index.is_unique

        # output filename
        output_filename_prefix = f"spredixcan-mashr-zscores-{f_tissue}"

        # Save pickle
        df.to_pickle(
            Path(spredixcan_pkl_output_folder, f"{output_filename_prefix}.pkl")
        )

        # Save tsv
        df.to_csv(
            Path(spredixcan_tsv_output_folder, f"{output_filename_prefix}.tsv.gz"),
            sep="\t",
            float_format="%.5e",
        )

# %% [markdown] papermill={"duration": 0.030059, "end_time": "2021-04-23T08:24:35.096196", "exception": false, "start_time": "2021-04-23T08:24:35.066137", "status": "completed"} tags=[]
# # Testing

# %% [markdown] papermill={"duration": 0.028135, "end_time": "2021-04-23T08:24:35.152241", "exception": false, "start_time": "2021-04-23T08:24:35.124106", "status": "completed"} tags=[]
# ## List of traits match those in S-MultiXcan

# %% papermill={"duration": 0.04596, "end_time": "2021-04-23T08:24:35.225870", "exception": false, "start_time": "2021-04-23T08:24:35.179910", "status": "completed"} tags=[]
_phenomexcan_trait_fullcodes = pd.Index(all_phenomexcan_traits)
display(_phenomexcan_trait_fullcodes)
assert _phenomexcan_trait_fullcodes.is_unique

# %% [markdown] papermill={"duration": 0.026848, "end_time": "2021-04-23T08:24:35.281138", "exception": false, "start_time": "2021-04-23T08:24:35.254290", "status": "completed"} tags=[]
# ### pickle

# %% papermill={"duration": 2.359798, "end_time": "2021-04-23T08:24:37.668827", "exception": false, "start_time": "2021-04-23T08:24:35.309029", "status": "completed"} tags=[]
output_filename_prefix = f"spredixcan-mashr-zscores-Adipose_Subcutaneous"
filepath = Path(spredixcan_pkl_output_folder, f"{output_filename_prefix}.pkl")
_spredixcan_traits = pd.read_pickle(filepath).columns

# %% papermill={"duration": 0.038805, "end_time": "2021-04-23T08:24:37.738471", "exception": false, "start_time": "2021-04-23T08:24:37.699666", "status": "completed"} tags=[]
_spredixcan_traits

# %% papermill={"duration": 0.037878, "end_time": "2021-04-23T08:24:37.805264", "exception": false, "start_time": "2021-04-23T08:24:37.767386", "status": "completed"} tags=[]
assert _spredixcan_traits.is_unique

# %% papermill={"duration": 0.038632, "end_time": "2021-04-23T08:24:37.871550", "exception": false, "start_time": "2021-04-23T08:24:37.832918", "status": "completed"} tags=[]
_tmp = _phenomexcan_trait_fullcodes.intersection(_spredixcan_traits)
display(_tmp)
assert _tmp.shape[0] == _phenomexcan_trait_fullcodes.shape[0]

# %% [markdown] papermill={"duration": 0.02767, "end_time": "2021-04-23T08:24:37.928387", "exception": false, "start_time": "2021-04-23T08:24:37.900717", "status": "completed"} tags=[]
# ### tsv.gz

# %% papermill={"duration": 16.334215, "end_time": "2021-04-23T08:24:54.289956", "exception": false, "start_time": "2021-04-23T08:24:37.955741", "status": "completed"} tags=[]
output_filename_prefix = f"spredixcan-mashr-zscores-Adipose_Visceral_Omentum"
filepath = Path(spredixcan_tsv_output_folder, f"{output_filename_prefix}.tsv.gz")
_spredixcan_traits = pd.read_csv(filepath, sep="\t", index_col="gene_id").columns

# %% papermill={"duration": 0.038867, "end_time": "2021-04-23T08:24:54.363037", "exception": false, "start_time": "2021-04-23T08:24:54.324170", "status": "completed"} tags=[]
_spredixcan_traits

# %% papermill={"duration": 0.037572, "end_time": "2021-04-23T08:24:54.428890", "exception": false, "start_time": "2021-04-23T08:24:54.391318", "status": "completed"} tags=[]
assert _spredixcan_traits.is_unique

# %% papermill={"duration": 0.039529, "end_time": "2021-04-23T08:24:54.496498", "exception": false, "start_time": "2021-04-23T08:24:54.456969", "status": "completed"} tags=[]
_tmp = _phenomexcan_trait_fullcodes.intersection(_spredixcan_traits)
display(_tmp)
assert _tmp.shape[0] == _phenomexcan_trait_fullcodes.shape[0]

# %% [markdown] papermill={"duration": 0.027324, "end_time": "2021-04-23T08:24:54.552389", "exception": false, "start_time": "2021-04-23T08:24:54.525065", "status": "completed"} tags=[]
# ## Values

# %% [markdown] papermill={"duration": 0.027508, "end_time": "2021-04-23T08:24:54.607564", "exception": false, "start_time": "2021-04-23T08:24:54.580056", "status": "completed"} tags=[]
# Tests taken from: https://github.com/hakyimlab/phenomexcan/blob/master/scripts/100_postprocessing/05_spredixcan.ipynb

# %% papermill={"duration": 0.24338, "end_time": "2021-04-23T08:24:54.878512", "exception": false, "start_time": "2021-04-23T08:24:54.635132", "status": "completed"} tags=[]
# pkl
output_filename_prefix = f"spredixcan-mashr-zscores-Thyroid"
filepath = Path(spredixcan_pkl_output_folder, f"{output_filename_prefix}.pkl")
df = pd.read_pickle(filepath)[
    "N02-Diagnoses_main_ICD10_N02_Recurrent_and_persistent_haematuria"
]

assert df.shape[0] == 15289
assert df.loc["ENSG00000213965"] == -3.6753054157625686
assert pd.isnull(df.loc["ENSG00000198670"])
assert df.loc["ENSG00000177025"] == 4.316259089446458

# %% papermill={"duration": 16.689601, "end_time": "2021-04-23T08:25:11.600257", "exception": false, "start_time": "2021-04-23T08:24:54.910656", "status": "completed"} tags=[]
# tsv.gz
output_filename_prefix = f"spredixcan-mashr-zscores-Thyroid"
filepath = Path(spredixcan_tsv_output_folder, f"{output_filename_prefix}.tsv.gz")
df = pd.read_csv(filepath, sep="\t", index_col="gene_id")[
    "N02-Diagnoses_main_ICD10_N02_Recurrent_and_persistent_haematuria"
]

assert df.shape[0] == 15289
assert df.loc["ENSG00000213965"].round(5) == -3.67531
assert pd.isnull(df.loc["ENSG00000198670"])
assert df.loc["ENSG00000177025"].round(5) == 4.31626

# %% [markdown] papermill={"duration": 0.028047, "end_time": "2021-04-23T08:25:11.660322", "exception": false, "start_time": "2021-04-23T08:25:11.632275", "status": "completed"} tags=[]
# Check if small values in tsv.gz are correctly saved:

# %% papermill={"duration": 16.365751, "end_time": "2021-04-23T08:25:28.054796", "exception": false, "start_time": "2021-04-23T08:25:11.689045", "status": "completed"} tags=[]
# tsv.gz
output_filename_prefix = f"spredixcan-mashr-zscores-Adipose_Subcutaneous"
filepath = Path(spredixcan_tsv_output_folder, f"{output_filename_prefix}.tsv.gz")
df = pd.read_csv(filepath, sep="\t", index_col="gene_id")

assert (
    df.loc[
        "ENSG00000002746",
        "20003_1141153242-Treatmentmedication_code_balsalazide_disodium",
    ].round(5)
    == 0.00327
)
assert df.loc["ENSG00000074706", "MAGNETIC_HDL.C"] == 0.00
assert (
    np.format_float_scientific(
        df.loc[
            "ENSG00000164112",
            "N13-Diagnoses_main_ICD10_N13_Obstructive_and_reflux_uropathy",
        ],
        5,
    )
    == "-1.80052e-07"
)
assert (
    np.format_float_scientific(
        df.loc[
            "ENSG00000257467",
            "20411_0-Ever_been_injured_or_injured_someone_else_through_drinking_alcohol_No",
        ],
        5,
    )
    == "-3.89826e-09"
)

# %% [markdown] papermill={"duration": 0.027918, "end_time": "2021-04-23T08:25:28.121294", "exception": false, "start_time": "2021-04-23T08:25:28.093376", "status": "completed"} tags=[]
# More tests taken from the webapp:

# %% [markdown] papermill={"duration": 0.027701, "end_time": "2021-04-23T08:25:28.176552", "exception": false, "start_time": "2021-04-23T08:25:28.148851", "status": "completed"} tags=[]
# Standing height

# %% papermill={"duration": 0.199171, "end_time": "2021-04-23T08:25:28.403131", "exception": false, "start_time": "2021-04-23T08:25:28.203960", "status": "completed"} tags=[]
output_filename_prefix = f"spredixcan-mashr-zscores-Whole_Blood"
filepath = Path(spredixcan_pkl_output_folder, f"{output_filename_prefix}.pkl")
_tmp = pd.read_pickle(filepath)["50_raw-Standing_height"]
assert _tmp.shape == (12610,)

# %% papermill={"duration": 0.039071, "end_time": "2021-04-23T08:25:28.474622", "exception": false, "start_time": "2021-04-23T08:25:28.435551", "status": "completed"} tags=[]
assert _tmp.loc["ENSG00000101019"].round(3) == -34.024

# %% papermill={"duration": 0.037342, "end_time": "2021-04-23T08:25:28.540467", "exception": false, "start_time": "2021-04-23T08:25:28.503125", "status": "completed"} tags=[]
assert _tmp.loc["ENSG00000109805"].round(3) == -22.855

# %% papermill={"duration": 0.037507, "end_time": "2021-04-23T08:25:28.606469", "exception": false, "start_time": "2021-04-23T08:25:28.568962", "status": "completed"} tags=[]
assert _tmp.loc["ENSG00000177311"].round(3) == 33.819

# %% papermill={"duration": 13.840854, "end_time": "2021-04-23T08:25:42.475301", "exception": false, "start_time": "2021-04-23T08:25:28.634447", "status": "completed"} tags=[]
output_filename_prefix = f"spredixcan-mashr-zscores-Whole_Blood"
filepath = Path(spredixcan_tsv_output_folder, f"{output_filename_prefix}.tsv.gz")
_tmp = pd.read_csv(filepath, sep="\t", index_col="gene_id")["50_raw-Standing_height"]
assert _tmp.shape == (12610,)

# %% papermill={"duration": 0.038931, "end_time": "2021-04-23T08:25:42.546791", "exception": false, "start_time": "2021-04-23T08:25:42.507860", "status": "completed"} tags=[]
assert _tmp.loc["ENSG00000101019"].round(3) == -34.024

# %% papermill={"duration": 0.037849, "end_time": "2021-04-23T08:25:42.612953", "exception": false, "start_time": "2021-04-23T08:25:42.575104", "status": "completed"} tags=[]
assert _tmp.loc["ENSG00000109805"].round(3) == -22.855

# %% papermill={"duration": 0.037922, "end_time": "2021-04-23T08:25:42.679148", "exception": false, "start_time": "2021-04-23T08:25:42.641226", "status": "completed"} tags=[]
assert _tmp.loc["ENSG00000177311"].round(3) == 33.819

# %% [markdown] papermill={"duration": 0.027489, "end_time": "2021-04-23T08:25:42.734860", "exception": false, "start_time": "2021-04-23T08:25:42.707371", "status": "completed"} tags=[]
# Schizophrenia

# %% papermill={"duration": 0.203188, "end_time": "2021-04-23T08:25:42.967111", "exception": false, "start_time": "2021-04-23T08:25:42.763923", "status": "completed"} tags=[]
output_filename_prefix = f"spredixcan-mashr-zscores-Prostate"
filepath = Path(spredixcan_pkl_output_folder, f"{output_filename_prefix}.pkl")
_tmp = pd.read_pickle(filepath)["pgc.scz2"]

# %% papermill={"duration": 0.039983, "end_time": "2021-04-23T08:25:43.037731", "exception": false, "start_time": "2021-04-23T08:25:42.997748", "status": "completed"} tags=[]
assert _tmp.loc["ENSG00000233822"].round(3) == 10.752

# %% papermill={"duration": 0.038271, "end_time": "2021-04-23T08:25:43.105661", "exception": false, "start_time": "2021-04-23T08:25:43.067390", "status": "completed"} tags=[]
assert _tmp.loc["ENSG00000137312"].round(3) == -8.827

# %% papermill={"duration": 0.037969, "end_time": "2021-04-23T08:25:43.172492", "exception": false, "start_time": "2021-04-23T08:25:43.134523", "status": "completed"} tags=[]
assert _tmp.loc["ENSG00000204257"].round(3) == -7.965

# %% papermill={"duration": 15.793113, "end_time": "2021-04-23T08:25:58.994545", "exception": false, "start_time": "2021-04-23T08:25:43.201432", "status": "completed"} tags=[]
output_filename_prefix = f"spredixcan-mashr-zscores-Prostate"
filepath = Path(spredixcan_tsv_output_folder, f"{output_filename_prefix}.tsv.gz")
_tmp = pd.read_csv(filepath, sep="\t", index_col="gene_id")["pgc.scz2"]

# %% papermill={"duration": 0.039145, "end_time": "2021-04-23T08:25:59.066187", "exception": false, "start_time": "2021-04-23T08:25:59.027042", "status": "completed"} tags=[]
assert _tmp.loc["ENSG00000233822"].round(3) == 10.752

# %% papermill={"duration": 0.039253, "end_time": "2021-04-23T08:25:59.145996", "exception": false, "start_time": "2021-04-23T08:25:59.106743", "status": "completed"} tags=[]
assert _tmp.loc["ENSG00000137312"].round(3) == -8.827

# %% papermill={"duration": 0.037367, "end_time": "2021-04-23T08:25:59.212573", "exception": false, "start_time": "2021-04-23T08:25:59.175206", "status": "completed"} tags=[]
assert _tmp.loc["ENSG00000204257"].round(3) == -7.965

# %% papermill={"duration": 0.027461, "end_time": "2021-04-23T08:25:59.268494", "exception": false, "start_time": "2021-04-23T08:25:59.241033", "status": "completed"} tags=[]
