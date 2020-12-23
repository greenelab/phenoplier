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

# %% [markdown] papermill={"duration": 0.039298, "end_time": "2020-12-23T18:04:39.246832", "exception": false, "start_time": "2020-12-23T18:04:39.207534", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.010239, "end_time": "2020-12-23T18:04:39.272325", "exception": false, "start_time": "2020-12-23T18:04:39.262086", "status": "completed"} tags=[]
# **TODO**

# %% [markdown] papermill={"duration": 0.010515, "end_time": "2020-12-23T18:04:39.292047", "exception": false, "start_time": "2020-12-23T18:04:39.281532", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.020255, "end_time": "2020-12-23T18:04:39.321650", "exception": false, "start_time": "2020-12-23T18:04:39.301395", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 0.194124, "end_time": "2020-12-23T18:04:39.525509", "exception": false, "start_time": "2020-12-23T18:04:39.331385", "status": "completed"} tags=[]
from pathlib import Path

import pandas as pd

import conf

# %% [markdown] papermill={"duration": 0.009123, "end_time": "2020-12-23T18:04:39.544244", "exception": false, "start_time": "2020-12-23T18:04:39.535121", "status": "completed"} tags=[]
# # Settings

# %% papermill={"duration": 0.023479, "end_time": "2020-12-23T18:04:39.577159", "exception": false, "start_time": "2020-12-23T18:04:39.553680", "status": "completed"} tags=[]
OUTPUT_DIR = conf.RESULTS["DRUG_DISEASE_ANALYSES"]
display(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown] papermill={"duration": 0.009706, "end_time": "2020-12-23T18:04:39.596873", "exception": false, "start_time": "2020-12-23T18:04:39.587167", "status": "completed"} tags=[]
# # PharmacotherapyDB: load gold standard

# %% [markdown] papermill={"duration": 0.009703, "end_time": "2020-12-23T18:04:39.616377", "exception": false, "start_time": "2020-12-23T18:04:39.606674", "status": "completed"} tags=[]
# FIXME: This could go into a separate and previous notebook

# %% [markdown] papermill={"duration": 0.009187, "end_time": "2020-12-23T18:04:39.635350", "exception": false, "start_time": "2020-12-23T18:04:39.626163", "status": "completed"} tags=[]
# ## Read data

# %% papermill={"duration": 0.030171, "end_time": "2020-12-23T18:04:39.674843", "exception": false, "start_time": "2020-12-23T18:04:39.644672", "status": "completed"} tags=[]
# TODO hardcoded
input_file = Path(
    conf.DATA_DIR, "hetionet/pharmacotherapydb-v1.0", "indications.tsv"
).resolve()
display(input_file)

pharmadb_gold_standard = pd.read_csv(input_file, sep="\t")

# %% papermill={"duration": 0.020017, "end_time": "2020-12-23T18:04:39.704939", "exception": false, "start_time": "2020-12-23T18:04:39.684922", "status": "completed"} tags=[]
pharmadb_gold_standard.shape

# %% papermill={"duration": 0.025481, "end_time": "2020-12-23T18:04:39.740804", "exception": false, "start_time": "2020-12-23T18:04:39.715323", "status": "completed"} tags=[]
pharmadb_gold_standard.head()

# %% papermill={"duration": 0.021146, "end_time": "2020-12-23T18:04:39.772888", "exception": false, "start_time": "2020-12-23T18:04:39.751742", "status": "completed"} tags=[]
pharmadb_gold_standard["doid_id"].unique().shape

# %% papermill={"duration": 0.02135, "end_time": "2020-12-23T18:04:39.805324", "exception": false, "start_time": "2020-12-23T18:04:39.783974", "status": "completed"} tags=[]
pharmadb_gold_standard["drugbank_id"].unique().shape

# %% [markdown] papermill={"duration": 0.01117, "end_time": "2020-12-23T18:04:39.827693", "exception": false, "start_time": "2020-12-23T18:04:39.816523", "status": "completed"} tags=[]
# ## Build gold standard

# %% papermill={"duration": 0.022333, "end_time": "2020-12-23T18:04:39.860528", "exception": false, "start_time": "2020-12-23T18:04:39.838195", "status": "completed"} tags=[]
pharmadb_gold_standard["category"].value_counts()

# %% papermill={"duration": 0.04118, "end_time": "2020-12-23T18:04:39.912647", "exception": false, "start_time": "2020-12-23T18:04:39.871467", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.02125, "end_time": "2020-12-23T18:04:39.945192", "exception": false, "start_time": "2020-12-23T18:04:39.923942", "status": "completed"} tags=[]
gold_standard.shape

# %% papermill={"duration": 0.020612, "end_time": "2020-12-23T18:04:39.977406", "exception": false, "start_time": "2020-12-23T18:04:39.956794", "status": "completed"} tags=[]
# assert gold_standard.shape[0] == 1388
assert gold_standard.shape[0] == 998

# %% papermill={"duration": 0.023936, "end_time": "2020-12-23T18:04:40.012850", "exception": false, "start_time": "2020-12-23T18:04:39.988914", "status": "completed"} tags=[]
gold_standard.head()

# %% papermill={"duration": 0.022093, "end_time": "2020-12-23T18:04:40.046429", "exception": false, "start_time": "2020-12-23T18:04:40.024336", "status": "completed"} tags=[]
gold_standard["trait"].unique().shape

# %% papermill={"duration": 0.023393, "end_time": "2020-12-23T18:04:40.081507", "exception": false, "start_time": "2020-12-23T18:04:40.058114", "status": "completed"} tags=[]
gold_standard["drug"].unique().shape

# %% papermill={"duration": 0.023772, "end_time": "2020-12-23T18:04:40.117647", "exception": false, "start_time": "2020-12-23T18:04:40.093875", "status": "completed"} tags=[]
gold_standard["true_class"].value_counts()

# %% papermill={"duration": 0.025883, "end_time": "2020-12-23T18:04:40.155866", "exception": false, "start_time": "2020-12-23T18:04:40.129983", "status": "completed"} tags=[]
gold_standard.dropna().shape

# %% papermill={"duration": 0.022141, "end_time": "2020-12-23T18:04:40.190845", "exception": false, "start_time": "2020-12-23T18:04:40.168704", "status": "completed"} tags=[]
doids_in_gold_standard = set(gold_standard["trait"].values)

# %% [markdown] papermill={"duration": 0.011775, "end_time": "2020-12-23T18:04:40.215437", "exception": false, "start_time": "2020-12-23T18:04:40.203662", "status": "completed"} tags=[]
# # Save

# %% papermill={"duration": 0.022126, "end_time": "2020-12-23T18:04:40.249120", "exception": false, "start_time": "2020-12-23T18:04:40.226994", "status": "completed"} tags=[]
output_file = Path(OUTPUT_DIR, "gold_standard.pkl").resolve()
display(output_file)

# %% papermill={"duration": 0.022466, "end_time": "2020-12-23T18:04:40.283775", "exception": false, "start_time": "2020-12-23T18:04:40.261309", "status": "completed"} tags=[]
gold_standard.to_pickle(output_file)

# %% papermill={"duration": 0.012227, "end_time": "2020-12-23T18:04:40.308654", "exception": false, "start_time": "2020-12-23T18:04:40.296427", "status": "completed"} tags=[]
