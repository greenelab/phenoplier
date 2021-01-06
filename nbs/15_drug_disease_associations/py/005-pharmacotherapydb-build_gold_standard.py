# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill
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
# **TODO**

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
from pathlib import Path

import pandas as pd

import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
OUTPUT_DIR = conf.RESULTS["DRUG_DISEASE_ANALYSES"]
display(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown] tags=[]
# # PharmacotherapyDB: load gold standard

# %% [markdown] tags=[]
# FIXME: This could go into a separate and previous notebook

# %% [markdown] tags=[]
# ## Read data

# %% tags=[]
# TODO hardcoded
input_file = Path(
    conf.DATA_DIR, "hetionet/pharmacotherapydb-v1.0", "indications.tsv"
).resolve()
display(input_file)

pharmadb_gold_standard = pd.read_csv(input_file, sep="\t")

# %% tags=[]
pharmadb_gold_standard.shape

# %% tags=[]
pharmadb_gold_standard.head()

# %% tags=[]
pharmadb_gold_standard["doid_id"].unique().shape

# %% tags=[]
pharmadb_gold_standard["drugbank_id"].unique().shape

# %% [markdown] tags=[]
# ## Build gold standard

# %% tags=[]
pharmadb_gold_standard["category"].value_counts()

# %% tags=[]
gold_standard = (
    pharmadb_gold_standard[pharmadb_gold_standard["category"].isin(("DM", "NOT"))]
    .set_index(["doid_id", "drugbank_id"])
    .apply(lambda x: int(x.category in ("DM",)), axis=1)
    .reset_index()
    .rename(
        columns={
            "doid_id": "trait",
            "drugbank_id": "drug",
            0: "true_class",
        }
    )
)

# %% tags=[]
gold_standard.shape

# %% tags=[]
# assert gold_standard.shape[0] == 1388
assert gold_standard.shape[0] == 998

# %% tags=[]
gold_standard.head()

# %% tags=[]
gold_standard["trait"].unique().shape

# %% tags=[]
gold_standard["drug"].unique().shape

# %% tags=[]
gold_standard["true_class"].value_counts()

# %% tags=[]
gold_standard.dropna().shape

# %% tags=[]
doids_in_gold_standard = set(gold_standard["trait"].values)

# %% [markdown] tags=[]
# # Save

# %% tags=[]
output_file = Path(OUTPUT_DIR, "gold_standard.pkl").resolve()
display(output_file)

# %% tags=[]
gold_standard.to_pickle(output_file)

# %% tags=[]
