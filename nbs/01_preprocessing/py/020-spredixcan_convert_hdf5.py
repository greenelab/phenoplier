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

# %% [markdown] papermill={"duration": 0.090218, "end_time": "2020-12-18T23:50:38.326556", "exception": false, "start_time": "2020-12-18T23:50:38.236338", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.050521, "end_time": "2020-12-18T23:50:38.468335", "exception": false, "start_time": "2020-12-18T23:50:38.417814", "status": "completed"} tags=[]
# This notebook reads .h5 files generated in PhenomeXcan by [this notebook](https://github.com/hakyimlab/phenomexcan/blob/master/scripts/100_postprocessing/05_spredixcan.ipynb), and saves one file per tissue with the results in pandas DataFrame format (genes in rows, traits in columns). It saves these in two formats: pickle and tsv.gz
#
# The notebook with generate two other folders in the parent of `SPREDIXCAN_H5_FOLDER`: `pkl` and `tsv`
#
# **The idea** is to have the data in a friendly format.

# %% [markdown] papermill={"duration": 0.016094, "end_time": "2020-12-18T23:50:38.502287", "exception": false, "start_time": "2020-12-18T23:50:38.486193", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.026911, "end_time": "2020-12-18T23:50:38.544923", "exception": false, "start_time": "2020-12-18T23:50:38.518012", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 0.206677, "end_time": "2020-12-18T23:50:38.768435", "exception": false, "start_time": "2020-12-18T23:50:38.561758", "status": "completed"} tags=[]
from pathlib import Path

import numpy as np
import pandas as pd

import conf
from data.cache import read_data
from data.hdf5 import simplify_trait_fullcode, HDF5_FILE_PATTERN

# %% [markdown] papermill={"duration": 0.016582, "end_time": "2020-12-18T23:50:38.802319", "exception": false, "start_time": "2020-12-18T23:50:38.785737", "status": "completed"} tags=[]
# # Settings

# %% papermill={"duration": 0.031439, "end_time": "2020-12-18T23:50:38.849882", "exception": false, "start_time": "2020-12-18T23:50:38.818443", "status": "completed"} tags=[]
SPREDIXCAN_H5_FOLDER = Path(
    conf.PHENOMEXCAN["SPREDIXCAN_MASHR_ZSCORES_FOLDER"],
    "hdf5",
)
display(SPREDIXCAN_H5_FOLDER)
assert SPREDIXCAN_H5_FOLDER.is_dir(), "The folder does not exist"

# %% papermill={"duration": 0.027386, "end_time": "2020-12-18T23:50:38.899215", "exception": false, "start_time": "2020-12-18T23:50:38.871829", "status": "completed"} tags=[]
spredixcan_pkl_output_folder = Path(SPREDIXCAN_H5_FOLDER.parent, "pkl").resolve()
spredixcan_pkl_output_folder.mkdir(exist_ok=True)
display(spredixcan_pkl_output_folder)

# %% papermill={"duration": 0.026616, "end_time": "2020-12-18T23:50:38.942918", "exception": false, "start_time": "2020-12-18T23:50:38.916302", "status": "completed"} tags=[]
spredixcan_tsv_output_folder = Path(SPREDIXCAN_H5_FOLDER.parent, "tsv").resolve()
spredixcan_tsv_output_folder.mkdir(exist_ok=True)
display(spredixcan_tsv_output_folder)

# %% [markdown] papermill={"duration": 0.017139, "end_time": "2020-12-18T23:50:38.977602", "exception": false, "start_time": "2020-12-18T23:50:38.960463", "status": "completed"} tags=[]
# # Read S-PrediXcan results

# %% [markdown] papermill={"duration": 0.016849, "end_time": "2020-12-18T23:50:39.011504", "exception": false, "start_time": "2020-12-18T23:50:38.994655", "status": "completed"} tags=[]
# ## Get list of files

# %% papermill={"duration": 0.026479, "end_time": "2020-12-18T23:50:39.054839", "exception": false, "start_time": "2020-12-18T23:50:39.028360", "status": "completed"} tags=[]
from glob import glob

# %% papermill={"duration": 0.026833, "end_time": "2020-12-18T23:50:39.099161", "exception": false, "start_time": "2020-12-18T23:50:39.072328", "status": "completed"} tags=[]
spredixcan_files = list(SPREDIXCAN_H5_FOLDER.glob("*.h5"))

# %% papermill={"duration": 0.028272, "end_time": "2020-12-18T23:50:39.145360", "exception": false, "start_time": "2020-12-18T23:50:39.117088", "status": "completed"} tags=[]
display(spredixcan_files[:5])
assert len(spredixcan_files) == 49

# %% [markdown] papermill={"duration": 0.017986, "end_time": "2020-12-18T23:50:39.182462", "exception": false, "start_time": "2020-12-18T23:50:39.164476", "status": "completed"} tags=[]
# ## Get tissue name from file list

# %% papermill={"duration": 0.026917, "end_time": "2020-12-18T23:50:39.226872", "exception": false, "start_time": "2020-12-18T23:50:39.199955", "status": "completed"} tags=[]
import re

# %% papermill={"duration": 0.027491, "end_time": "2020-12-18T23:50:39.272478", "exception": false, "start_time": "2020-12-18T23:50:39.244987", "status": "completed"} tags=[]
# some testing
match = re.search(HDF5_FILE_PATTERN, "spredixcan-Esophagus_Muscularis-zscore.h5")
assert match.group("tissue") == "Esophagus_Muscularis"

match = re.search(
    HDF5_FILE_PATTERN, "spredixcan-Brain_Anterior_cingulate_cortex_BA24-zscore.h5"
)
assert match.group("tissue") == "Brain_Anterior_cingulate_cortex_BA24"

# %% [markdown] papermill={"duration": 0.017957, "end_time": "2020-12-18T23:50:39.308937", "exception": false, "start_time": "2020-12-18T23:50:39.290980", "status": "completed"} tags=[]
# # Load S-PrediXcan results

# %% [markdown] papermill={"duration": 0.017422, "end_time": "2020-12-18T23:50:39.343651", "exception": false, "start_time": "2020-12-18T23:50:39.326229", "status": "completed"} tags=[]
# ## Get all PhenomeXcan trait full codes

# %% papermill={"duration": 0.204969, "end_time": "2020-12-18T23:50:39.566040", "exception": false, "start_time": "2020-12-18T23:50:39.361071", "status": "completed"} tags=[]
from entity import Trait

# %% papermill={"duration": 0.33429, "end_time": "2020-12-18T23:50:39.919472", "exception": false, "start_time": "2020-12-18T23:50:39.585182", "status": "completed"} tags=[]
all_phenomexcan_traits = {
    trait_fullcode
    for trait_fullcode in read_data(
        conf.PHENOMEXCAN["SMULTIXCAN_MASHR_ZSCORES_FILE"]
    ).columns
}

# %% papermill={"duration": 0.028867, "end_time": "2020-12-18T23:50:39.969314", "exception": false, "start_time": "2020-12-18T23:50:39.940447", "status": "completed"} tags=[]
display(len(all_phenomexcan_traits))
assert len(all_phenomexcan_traits) == 4091

# %% [markdown] papermill={"duration": 0.017718, "end_time": "2020-12-18T23:50:40.005136", "exception": false, "start_time": "2020-12-18T23:50:39.987418", "status": "completed"} tags=[]
# ## Read all results

# %% papermill={"duration": 16543.399286, "end_time": "2020-12-19T04:26:23.422002", "exception": false, "start_time": "2020-12-18T23:50:40.022716", "status": "completed"} tags=[]
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

# %% [markdown] papermill={"duration": 0.029082, "end_time": "2020-12-19T04:26:23.485688", "exception": false, "start_time": "2020-12-19T04:26:23.456606", "status": "completed"} tags=[]
# # Testing

# %% [markdown] papermill={"duration": 0.025997, "end_time": "2020-12-19T04:26:23.537564", "exception": false, "start_time": "2020-12-19T04:26:23.511567", "status": "completed"} tags=[]
# ## List of traits match those in S-MultiXcan

# %% papermill={"duration": 0.051068, "end_time": "2020-12-19T04:26:23.641801", "exception": false, "start_time": "2020-12-19T04:26:23.590733", "status": "completed"} tags=[]
_phenomexcan_trait_fullcodes = pd.Index(all_phenomexcan_traits)
display(_phenomexcan_trait_fullcodes)
assert _phenomexcan_trait_fullcodes.is_unique

# %% [markdown] papermill={"duration": 0.025616, "end_time": "2020-12-19T04:26:23.694724", "exception": false, "start_time": "2020-12-19T04:26:23.669108", "status": "completed"} tags=[]
# ### pickle

# %% papermill={"duration": 0.222144, "end_time": "2020-12-19T04:26:23.942519", "exception": false, "start_time": "2020-12-19T04:26:23.720375", "status": "completed"} tags=[]
output_filename_prefix = f"spredixcan-mashr-zscores-Adipose_Subcutaneous"
filepath = Path(spredixcan_pkl_output_folder, f"{output_filename_prefix}.pkl")
_spredixcan_traits = pd.read_pickle(filepath).columns

# %% papermill={"duration": 0.03774, "end_time": "2020-12-19T04:26:24.007657", "exception": false, "start_time": "2020-12-19T04:26:23.969917", "status": "completed"} tags=[]
_spredixcan_traits

# %% papermill={"duration": 0.036937, "end_time": "2020-12-19T04:26:24.071976", "exception": false, "start_time": "2020-12-19T04:26:24.035039", "status": "completed"} tags=[]
assert _spredixcan_traits.is_unique

# %% papermill={"duration": 0.038609, "end_time": "2020-12-19T04:26:24.137243", "exception": false, "start_time": "2020-12-19T04:26:24.098634", "status": "completed"} tags=[]
_tmp = _phenomexcan_trait_fullcodes.intersection(_spredixcan_traits)
display(_tmp)
assert _tmp.shape[0] == _phenomexcan_trait_fullcodes.shape[0]

# %% [markdown] papermill={"duration": 0.026432, "end_time": "2020-12-19T04:26:24.192239", "exception": false, "start_time": "2020-12-19T04:26:24.165807", "status": "completed"} tags=[]
# ### tsv.gz

# %% papermill={"duration": 16.328505, "end_time": "2020-12-19T04:26:40.547293", "exception": false, "start_time": "2020-12-19T04:26:24.218788", "status": "completed"} tags=[]
output_filename_prefix = f"spredixcan-mashr-zscores-Adipose_Visceral_Omentum"
filepath = Path(spredixcan_tsv_output_folder, f"{output_filename_prefix}.tsv.gz")
_spredixcan_traits = pd.read_csv(filepath, sep="\t", index_col="gene_id").columns

# %% papermill={"duration": 0.037996, "end_time": "2020-12-19T04:26:40.614756", "exception": false, "start_time": "2020-12-19T04:26:40.576760", "status": "completed"} tags=[]
_spredixcan_traits

# %% papermill={"duration": 0.037735, "end_time": "2020-12-19T04:26:40.679677", "exception": false, "start_time": "2020-12-19T04:26:40.641942", "status": "completed"} tags=[]
assert _spredixcan_traits.is_unique

# %% papermill={"duration": 0.038634, "end_time": "2020-12-19T04:26:40.746025", "exception": false, "start_time": "2020-12-19T04:26:40.707391", "status": "completed"} tags=[]
_tmp = _phenomexcan_trait_fullcodes.intersection(_spredixcan_traits)
display(_tmp)
assert _tmp.shape[0] == _phenomexcan_trait_fullcodes.shape[0]

# %% [markdown] papermill={"duration": 0.02712, "end_time": "2020-12-19T04:26:40.801457", "exception": false, "start_time": "2020-12-19T04:26:40.774337", "status": "completed"} tags=[]
# ## Values

# %% [markdown] papermill={"duration": 0.026976, "end_time": "2020-12-19T04:26:40.855783", "exception": false, "start_time": "2020-12-19T04:26:40.828807", "status": "completed"} tags=[]
# Tests taken from: https://github.com/hakyimlab/phenomexcan/blob/master/scripts/100_postprocessing/05_spredixcan.ipynb

# %% papermill={"duration": 0.2444, "end_time": "2020-12-19T04:26:41.127379", "exception": false, "start_time": "2020-12-19T04:26:40.882979", "status": "completed"} tags=[]
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

# %% papermill={"duration": 16.936349, "end_time": "2020-12-19T04:26:58.094257", "exception": false, "start_time": "2020-12-19T04:26:41.157908", "status": "completed"} tags=[]
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

# %% [markdown] papermill={"duration": 0.027664, "end_time": "2020-12-19T04:26:58.154146", "exception": false, "start_time": "2020-12-19T04:26:58.126482", "status": "completed"} tags=[]
# Check if small values in tsv.gz are correctly saved:

# %% papermill={"duration": 16.465154, "end_time": "2020-12-19T04:27:14.647412", "exception": false, "start_time": "2020-12-19T04:26:58.182258", "status": "completed"} tags=[]
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

# %% [markdown] papermill={"duration": 0.027112, "end_time": "2020-12-19T04:27:14.705905", "exception": false, "start_time": "2020-12-19T04:27:14.678793", "status": "completed"} tags=[]
# More tests taken from the webapp:

# %% [markdown] papermill={"duration": 0.0273, "end_time": "2020-12-19T04:27:14.760785", "exception": false, "start_time": "2020-12-19T04:27:14.733485", "status": "completed"} tags=[]
# Standing height

# %% papermill={"duration": 0.190563, "end_time": "2020-12-19T04:27:14.979411", "exception": false, "start_time": "2020-12-19T04:27:14.788848", "status": "completed"} tags=[]
output_filename_prefix = f"spredixcan-mashr-zscores-Whole_Blood"
filepath = Path(spredixcan_pkl_output_folder, f"{output_filename_prefix}.pkl")
_tmp = pd.read_pickle(filepath)["50_raw-Standing_height"]
assert _tmp.shape == (12610,)

# %% papermill={"duration": 0.039263, "end_time": "2020-12-19T04:27:15.048499", "exception": false, "start_time": "2020-12-19T04:27:15.009236", "status": "completed"} tags=[]
assert _tmp.loc["ENSG00000101019"].round(3) == -34.024

# %% papermill={"duration": 0.038372, "end_time": "2020-12-19T04:27:15.115043", "exception": false, "start_time": "2020-12-19T04:27:15.076671", "status": "completed"} tags=[]
assert _tmp.loc["ENSG00000109805"].round(3) == -22.855

# %% papermill={"duration": 0.038249, "end_time": "2020-12-19T04:27:15.182664", "exception": false, "start_time": "2020-12-19T04:27:15.144415", "status": "completed"} tags=[]
assert _tmp.loc["ENSG00000177311"].round(3) == 33.819

# %% papermill={"duration": 13.986454, "end_time": "2020-12-19T04:27:29.198339", "exception": false, "start_time": "2020-12-19T04:27:15.211885", "status": "completed"} tags=[]
output_filename_prefix = f"spredixcan-mashr-zscores-Whole_Blood"
filepath = Path(spredixcan_tsv_output_folder, f"{output_filename_prefix}.tsv.gz")
_tmp = pd.read_csv(filepath, sep="\t", index_col="gene_id")["50_raw-Standing_height"]
assert _tmp.shape == (12610,)

# %% papermill={"duration": 0.039225, "end_time": "2020-12-19T04:27:29.270924", "exception": false, "start_time": "2020-12-19T04:27:29.231699", "status": "completed"} tags=[]
assert _tmp.loc["ENSG00000101019"].round(3) == -34.024

# %% papermill={"duration": 0.038623, "end_time": "2020-12-19T04:27:29.338009", "exception": false, "start_time": "2020-12-19T04:27:29.299386", "status": "completed"} tags=[]
assert _tmp.loc["ENSG00000109805"].round(3) == -22.855

# %% papermill={"duration": 0.03786, "end_time": "2020-12-19T04:27:29.405210", "exception": false, "start_time": "2020-12-19T04:27:29.367350", "status": "completed"} tags=[]
assert _tmp.loc["ENSG00000177311"].round(3) == 33.819

# %% [markdown] papermill={"duration": 0.028052, "end_time": "2020-12-19T04:27:29.461591", "exception": false, "start_time": "2020-12-19T04:27:29.433539", "status": "completed"} tags=[]
# Schizophrenia

# %% papermill={"duration": 0.213025, "end_time": "2020-12-19T04:27:29.702289", "exception": false, "start_time": "2020-12-19T04:27:29.489264", "status": "completed"} tags=[]
output_filename_prefix = f"spredixcan-mashr-zscores-Prostate"
filepath = Path(spredixcan_pkl_output_folder, f"{output_filename_prefix}.pkl")
_tmp = pd.read_pickle(filepath)["pgc.scz2"]

# %% papermill={"duration": 0.040031, "end_time": "2020-12-19T04:27:29.773654", "exception": false, "start_time": "2020-12-19T04:27:29.733623", "status": "completed"} tags=[]
assert _tmp.loc["ENSG00000233822"].round(3) == 10.752

# %% papermill={"duration": 0.038456, "end_time": "2020-12-19T04:27:29.841036", "exception": false, "start_time": "2020-12-19T04:27:29.802580", "status": "completed"} tags=[]
assert _tmp.loc["ENSG00000137312"].round(3) == -8.827

# %% papermill={"duration": 0.03823, "end_time": "2020-12-19T04:27:29.908907", "exception": false, "start_time": "2020-12-19T04:27:29.870677", "status": "completed"} tags=[]
assert _tmp.loc["ENSG00000204257"].round(3) == -7.965

# %% papermill={"duration": 16.025026, "end_time": "2020-12-19T04:27:45.962916", "exception": false, "start_time": "2020-12-19T04:27:29.937890", "status": "completed"} tags=[]
output_filename_prefix = f"spredixcan-mashr-zscores-Prostate"
filepath = Path(spredixcan_tsv_output_folder, f"{output_filename_prefix}.tsv.gz")
_tmp = pd.read_csv(filepath, sep="\t", index_col="gene_id")["pgc.scz2"]

# %% papermill={"duration": 0.03967, "end_time": "2020-12-19T04:27:46.032677", "exception": false, "start_time": "2020-12-19T04:27:45.993007", "status": "completed"} tags=[]
assert _tmp.loc["ENSG00000233822"].round(3) == 10.752

# %% papermill={"duration": 0.038588, "end_time": "2020-12-19T04:27:46.100316", "exception": false, "start_time": "2020-12-19T04:27:46.061728", "status": "completed"} tags=[]
assert _tmp.loc["ENSG00000137312"].round(3) == -8.827

# %% papermill={"duration": 0.039568, "end_time": "2020-12-19T04:27:46.169166", "exception": false, "start_time": "2020-12-19T04:27:46.129598", "status": "completed"} tags=[]
assert _tmp.loc["ENSG00000204257"].round(3) == -7.965

# %% papermill={"duration": 0.028873, "end_time": "2020-12-19T04:27:46.227280", "exception": false, "start_time": "2020-12-19T04:27:46.198407", "status": "completed"} tags=[]
