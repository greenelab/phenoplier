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

# %% [markdown]
# # Description

# %% [markdown]
# This notebook takes the top LVs driving the prediction of a drug and a trait. Then it reads the cell types/tissues associated with each LV.

# %% [markdown]
# # Modules loading

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from data.recount2 import LVAnalysis
import conf

# %% [markdown]
# # Settings

# %% tags=["parameters"]
SHORT_TRAIT_NAME = "ICD10_I70_Atherosclerosis"
FULL_TRAIT_NAME = "I70-Diagnoses_main_ICD10_I70_Atherosclerosis"

# %%
QUANTILE = 0.95

# %% [markdown]
# # Paths

# %% tags=[]
OUTPUT_DIR = conf.RESULTS["DRUG_DISEASE_ANALYSES"] / "lincs" / "analyses"
display(OUTPUT_DIR)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# %% tags=[]
OUTPUT_FIGURES_DIR = Path(
    conf.MANUSCRIPT["FIGURES_DIR"], "drug_disease_prediction"
).resolve()
display(OUTPUT_FIGURES_DIR)
OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# %%
OUTPUT_FILE = OUTPUT_DIR / "cardiovascular-niacin.h5"
display(OUTPUT_FILE)

# %% [markdown]
# # Load data

# %% [markdown]
# ## Original data

# %% tags=[]
INPUT_SUBSET = "z_score_std"

# %% tags=[]
INPUT_STEM = "projection-smultixcan-efo_partial-mashr-zscores"

# %% tags=[]
input_filepath = Path(
    conf.RESULTS["DATA_TRANSFORMATIONS_DIR"],
    INPUT_SUBSET,
    f"{INPUT_SUBSET}-{INPUT_STEM}.pkl",
).resolve()
display(input_filepath)

assert input_filepath.exists(), "Input file does not exist"

input_filepath_stem = input_filepath.stem
display(input_filepath_stem)

# %% tags=[]
data = pd.read_pickle(input_filepath)

# %% tags=[]
data.shape

# %% tags=[]
data.head()

# %% [markdown]
# ## Load drug/trait LVs

# %%
output_file = OUTPUT_DIR / "cardiovascular-niacin.h5"
display(output_file)

# %%
with pd.HDFStore(output_file, mode="r") as store:
    traits_module_tissue_data = store["traits_module_tissue_data"]
    drug_data = store["drug_data"]
    drug_trait_predictions = store["drug_trait_predictions"]

# %%
top_lvs = drug_trait_predictions[FULL_TRAIT_NAME].sort_values(ascending=False)

q = top_lvs.quantile(QUANTILE)

top_lvs = top_lvs[top_lvs > q]

# %%
top_lvs.shape

# %%
top_lvs

# %%
lvs_list = top_lvs.index.tolist()

# %%
len(lvs_list)


# %%
def _get_attributes(x):
    _cols = [c for c in x.index if not c.startswith("LV")]
    _tmp = x[_cols].dropna()
    if _tmp.shape[0] > 0:
        return _tmp.iloc[0]

    return None


# %% [markdown]
# # Get cell types/tissues for top LVs

# %%
cell_type_dfs = []
tissue_dfs = []

pbar = tqdm(lvs_list)
for lv_name in pbar:
    pbar.set_description(lv_name)

    lv_obj = LVAnalysis(lv_name, data)

    lv_data = lv_obj.get_experiments_data(debug=False, warnings=False)

    # get cell type attributes
    lv_attrs = pd.Series(lv_data.columns.tolist())
    lv_attrs = lv_attrs[
        lv_attrs.str.match(
            "(?:cell[^\w]*type$)",
            case=False,
            flags=re.IGNORECASE,
        ).values
    ].sort_values(ascending=False)

    lv_attrs_data = lv_data[lv_attrs.tolist() + [lv_name]]
    lv_attrs_data = lv_attrs_data.assign(
        attr=lv_attrs_data.apply(_get_attributes, axis=1)
    )
    lv_attrs_data = lv_attrs_data.drop(columns=lv_attrs.tolist())
    lv_attrs_data = lv_attrs_data.dropna().sort_values(lv_name, ascending=False)
    lv_attrs_data = lv_attrs_data.rename(columns={lv_name: "lv"})
    lv_attrs_data = lv_attrs_data.assign(lv_name=lv_name)
    cell_type_dfs.append(lv_attrs_data)

    # get tissue attributes
    lv_attrs = pd.Series(lv_data.columns.tolist())
    lv_attrs = lv_attrs[
        lv_attrs.str.match(
            "(?:tissue$)|(?:tissue[^\w]*type$)",
            case=False,
            flags=re.IGNORECASE,
        ).values
    ].sort_values(ascending=False)

    lv_attrs_data = lv_data[lv_attrs.tolist() + [lv_name]]
    lv_attrs_data = lv_attrs_data.assign(
        attr=lv_attrs_data.apply(_get_attributes, axis=1)
    )
    lv_attrs_data = lv_attrs_data.drop(columns=lv_attrs.tolist())
    lv_attrs_data = lv_attrs_data.dropna().sort_values(lv_name, ascending=False)
    lv_attrs_data = lv_attrs_data.rename(columns={lv_name: "lv"})
    lv_attrs_data = lv_attrs_data.assign(lv_name=lv_name)
    tissue_dfs.append(lv_attrs_data)

# %% [markdown]
# ## Prepare dataframe

# %%
cell_types_data = pd.concat(cell_type_dfs, ignore_index=True)

# %%
cell_types_data.shape

# %%
cell_types_data.head()

# %%
tissues_data = pd.concat(tissue_dfs, ignore_index=True)

# %%
tissues_data.shape

# %%
tissues_data.head()

# %% [markdown]
# ## Save

# %%
with pd.HDFStore(output_file, mode="r+", complevel=4) as store:
    store.put(f"traits/{SHORT_TRAIT_NAME}/top_lvs", top_lvs, format="fixed")
    store.put(f"traits/{SHORT_TRAIT_NAME}/cell_types", cell_types_data, format="fixed")
    store.put(f"traits/{SHORT_TRAIT_NAME}/tissues", tissues_data, format="fixed")

# %%
