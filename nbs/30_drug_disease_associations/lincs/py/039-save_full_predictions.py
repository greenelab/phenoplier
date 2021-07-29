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
# This notebook reads the *full predictions* results (with all traits in PhenomeXcan, it doesn't matter if it doesn't have DOID map) generated with the `011-prediction-*` notebooks and saves for later use.

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from data.hdf5 import simplify_trait_fullcode
import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
# these numbers are for testing/checking
N_TISSUES = 49
N_THRESHOLDS = 5

# %% tags=[]
INPUT_DIR = conf.RESULTS["DRUG_DISEASE_ANALYSES"] / "lincs"
display(INPUT_DIR)
assert INPUT_DIR.exists()

# %% tags=[]
INPUT_PREDICTIONS_DIR = Path(INPUT_DIR, "predictions", "dotprod_neg")
display(INPUT_PREDICTIONS_DIR)
INPUT_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

# %% tags=[]
OUTPUT_DIR = Path(INPUT_DIR, "predictions")
display(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% tags=[]
OUTPUT_FILENAME = Path(OUTPUT_DIR, "full_predictions_by_tissue-rank.h5")
display(OUTPUT_FILENAME)

# %% [markdown] tags=[]
# # Load drug-disease predictions

# %% tags=[]
from collections import defaultdict

# %% tags=[]
# get all prediction files

current_prediction_files = sorted(
    [f for f in list(INPUT_PREDICTIONS_DIR.glob("*.h5")) if "-projection-" in f.name]
)
display(len(current_prediction_files))

assert len(current_prediction_files) == (N_TISSUES * N_THRESHOLDS)

# %% tags=[]
current_prediction_files[:10]


# %%
def _get_tissue(x):
    """
    It extracts the tissue name from a filename.
    """
    if x.endswith("-projection"):
        return x.split("spredixcan-mashr-zscores-")[1].split("-projection")[0]
    else:
        return x.split("spredixcan-mashr-zscores-")[1].split("-data")[0]


# %%
# get all tissue names

all_tissues = set()
all_methods = set()

for f in tqdm(current_prediction_files, ncols=100):
    # read metadata
    metadata = pd.read_hdf(f, key="metadata")

    # get the tissue name
    _data = metadata.data.values[0]
    _tissue = _get_tissue(_data)
    all_tissues.add(_tissue)

    _n_top_genes = metadata.n_top_genes.values[0]
    all_methods.add(_n_top_genes)

# %%
assert len(all_methods) == N_THRESHOLDS
display(all_methods)

# %%
all_tissues = sorted(list(all_tissues))

# %%
assert len(all_tissues) == N_TISSUES

# %%
_tmp_df = pd.read_hdf(current_prediction_files[0], key="full_prediction")
all_traits = _tmp_df["trait"].drop_duplicates().tolist()
all_drugs = _tmp_df["drug"].drop_duplicates().tolist()

# %%
_tmp_df.head()

# %%
assert len(all_traits) == 4091

# %%
assert len(all_drugs) == 1170

# %% [markdown]
# ## Create predictions dataframe

# %% tags=[]
# Iterate for each prediction file and perform some preprocessing.
#
# Each prediction file (.h5) has the predictions of one method (either module-based
# or gene-based) for all drug-disease pairs across all S-PrediXcan tissues

with pd.HDFStore(OUTPUT_FILENAME, mode="w", complevel=4) as store:
    for tissue in tqdm(all_tissues, ncols=100):
        # get all the prediction files for one tissue
        tissue_prediction_files = [
            x for x in current_prediction_files if f"-{tissue}-" in x.name
        ]
        assert len(tissue_prediction_files) == len(all_methods)

        tissue_df = pd.DataFrame(
            data=0,
            index=all_traits.copy(),
            columns=all_drugs.copy(),
            dtype="float32",
        )

        for f in tissue_prediction_files:
            # read metadata
            metadata = pd.read_hdf(f, key="metadata")
            _data = metadata.data.values[0]
            _tissue = _get_tissue(_data)
            assert _tissue == tissue

            # get full predictions
            prediction_data = pd.read_hdf(f, key="full_prediction")
            prediction_data["score"] = prediction_data["score"].rank()
            prediction_data = prediction_data.pivot(
                index="trait", columns="drug", values="score"
            )
            prediction_data = prediction_data.astype("float32")

            # sum across N_THRESHOLDS (which is equals to len(all_methods))
            tissue_df += prediction_data.loc[tissue_df.index, tissue_df.columns]

        # save the average
        store.put(
            simplify_trait_fullcode(tissue, prefix=""),
            (tissue_df / len(all_methods)).astype("float32"),
            format="fixed",
        )

# %% [markdown] tags=[]
# ## Testing

# %% tags=[]
_tissue = "Adipose_Subcutaneous"

# %%
with pd.HDFStore(OUTPUT_FILENAME, mode="r") as store:
    tissue_df = store[simplify_trait_fullcode(_tissue, prefix="")]

# %%
assert not tissue_df.isna().any().any()

# %%
tissue_df.shape

# %%
tissue_df.head()

# %%
_files = [x for x in current_prediction_files if f"-{_tissue}-" in x.name]

# %%
display(len(_files))
assert len(_files) == N_THRESHOLDS

# %%
_files_data = [
    pd.read_hdf(f, key="full_prediction").set_index(["trait", "drug"]).squeeze().rank()
    for f in _files
]

# %%
_files_data[0].head(5)

# %%
_trait = "I9_PHLETHROMBDVTLOW-DVT_of_lower_extremities"
_drug = "DB00014"

# %%
assert tissue_df.loc[_trait, _drug].round(7) == np.mean(
    [x.loc[(_trait, _drug)] for x in _files_data]
).round(7).astype("float32")

# %%
