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
# This notebook builds the gold-standard for drug-disease prediction using [PharmarcotherapyDB](https://dx.doi.org/10.7554%2FeLife.26726)
#
# Instead of using all drug-disease pairs in PharmarcotherapyDB, we only use disease-modifying (DM) pairs as positive cases, and non-indications (NOT) as negative ones. We exclude symptomatic (SYM) because those might not exert an important effect to the disease.

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
# ## Read data

# %% tags=[]
input_file = conf.PHARMACOTHERAPYDB["INDICATIONS_FILE"]
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
