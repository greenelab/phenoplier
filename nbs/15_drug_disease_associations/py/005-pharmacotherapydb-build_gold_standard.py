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

# %% [markdown] papermill={"duration": 0.044577, "end_time": "2020-12-18T22:38:21.345879", "exception": false, "start_time": "2020-12-18T22:38:21.301302", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.011764, "end_time": "2020-12-18T22:38:21.398073", "exception": false, "start_time": "2020-12-18T22:38:21.386309", "status": "completed"} tags=[]
# **TODO**

# %% [markdown] papermill={"duration": 0.011799, "end_time": "2020-12-18T22:38:21.422136", "exception": false, "start_time": "2020-12-18T22:38:21.410337", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.022467, "end_time": "2020-12-18T22:38:21.456126", "exception": false, "start_time": "2020-12-18T22:38:21.433659", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 0.192251, "end_time": "2020-12-18T22:38:21.659996", "exception": false, "start_time": "2020-12-18T22:38:21.467745", "status": "completed"} tags=[]
from pathlib import Path

import pandas as pd

import conf

# %% [markdown] papermill={"duration": 0.011416, "end_time": "2020-12-18T22:38:21.683356", "exception": false, "start_time": "2020-12-18T22:38:21.671940", "status": "completed"} tags=[]
# # Settings

# %%
OUTPUT_DIR = conf.RESULTS["DRUG_DISEASE_ANALYSES"]
display(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# # PharmacotherapyDB: load gold standard

# %% [markdown]
# FIXME: This could go into a separate and previous notebook

# %% [markdown]
# ## Read data

# %%
# TODO hardcoded
input_file = Path(
    conf.DATA_DIR,
    'hetionet/pharmacotherapydb-v1.0',
    'indications.tsv'
).resolve()
display(input_file)

pharmadb_gold_standard = pd.read_csv(input_file, sep='\t')

# %%
pharmadb_gold_standard.shape

# %%
pharmadb_gold_standard.head()

# %%
pharmadb_gold_standard['doid_id'].unique().shape

# %%
pharmadb_gold_standard['drugbank_id'].unique().shape

# %% [markdown]
# ## Build gold standard

# %%
pharmadb_gold_standard['category'].value_counts()

# %%
gold_standard = (
    pharmadb_gold_standard[pharmadb_gold_standard['category'].isin(('DM', 'NOT'))]
    .set_index(['doid_id', 'drugbank_id'])
    .apply(lambda x: int(x.category in ('DM',)), axis=1).reset_index()
    .rename(columns={
        'doid_id': 'trait',
        'drugbank_id': 'drug',
        0: 'true_class',
    })
)

# %%
gold_standard.shape

# %%
# assert gold_standard.shape[0] == 1388
assert gold_standard.shape[0] == 998

# %%
gold_standard.head()

# %%
gold_standard['trait'].unique().shape

# %%
gold_standard['drug'].unique().shape

# %%
gold_standard['true_class'].value_counts()

# %%
gold_standard.dropna().shape

# %%
doids_in_gold_standard = set(gold_standard['trait'].values)

# %% [markdown]
# # Save

# %%
output_file = Path(
    OUTPUT_DIR,
    "gold_standard.pkl"
).resolve()
display(output_file)

# %%
gold_standard.to_pickle(output_file)

# %%
