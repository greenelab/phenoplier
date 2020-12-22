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

import numpy as np
import pandas as pd

import conf
# from multiplier import MultiplierProjection
# from entity import Trait

# %% [markdown] papermill={"duration": 0.011416, "end_time": "2020-12-18T22:38:21.683356", "exception": false, "start_time": "2020-12-18T22:38:21.671940", "status": "completed"} tags=[]
# # Settings

# %%
OUTPUT_DIR = conf.RESULTS["DRUG_DISEASE_ANALYSES"]
display(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
OUTPUT_PREDICTIONS_DIR = Path(OUTPUT_DIR, "predictions")
display(OUTPUT_PREDICTIONS_DIR)
OUTPUT_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# # Load PharmacotherapyDB gold standard

# %%
gold_standard = pd.read_pickle(
    Path(OUTPUT_DIR, "gold_standard.pkl"),
)

# %%
gold_standard.shape

# %%
gold_standard.head()

# %%
gold_standard['true_class'].value_counts()

# %%
gold_standard['true_class'].value_counts(normalize=True)

# %%
doids_in_gold_standard = set(gold_standard['trait'])

# %% [markdown]
# # Load PhenomeXcan data

# %%
# input_file = Path(
#     conf.PHENOMEXCAN["SPREDIXCAN_MASHR_ZSCORES_FOLDER"],
#     "most_signif",
#     "spredixcan-most_signif.pkl"
# ).resolve()

# display(input_file)

# %%
# phenomexcan_data = pd.read_pickle(input_file)

# %%
# phenomexcan_data.shape

# %%
# phenomexcan_data = phenomexcan_data.dropna(how='any')

# %%
# phenomexcan_data.shape

# %%
# phenomexcan_data.head()

# %%
# assert phenomexcan_data.index.is_unique

# %% [markdown]
# # Load drug-disease predictions

# %%
from collections import defaultdict

# %%
current_prediction_files = list(OUTPUT_PREDICTIONS_DIR.glob("*.h5"))
display(len(current_prediction_files))

# %%
predictions = []

for f in current_prediction_files:
#     print(f.name)
    
    prediction_data = pd.read_hdf(f, key="prediction")
    prediction_data = pd.merge(
        prediction_data, gold_standard,
        on=['trait', 'drug'],
        how='inner'
    )
    
    metadata = pd.read_hdf(f, key="metadata")
    
#     new_predictions[f"{metadata.method}"][metadata.data] = prediction_data
    prediction_data['trait'] = prediction_data['trait'].astype('category')
    prediction_data['drug'] = prediction_data['drug'].astype('category')
    prediction_data = prediction_data.assign(method=metadata.method)
    prediction_data = prediction_data.assign(data=metadata.data)
    
    predictions.append(prediction_data)
    
#     print(f"  shape: {prediction_data.shape}")

# %%
predictions = pd.concat(predictions, ignore_index=True)

# %%
predictions.shape

# %%
predictions.head()


# %% [markdown]
# # Average predictions

# %%
def _reduce(x):
    return pd.Series({
        'score': x['score'].max(),
        'true_class': x['true_class'].unique()[0] if x['true_class'].unique().shape[0] == 1 else None,
        'data': x['method'].iloc[0],
    })


# %%
predictions_avg = predictions.groupby(['trait', 'drug', 'method']).apply(_reduce).dropna().sort_index().reset_index()

# %%
predictions_avg.shape

# %%
predictions_avg.head()

# %% [markdown]
# # ROC

# %%
from sklearn.metrics import roc_auc_score

# %%
predictions.groupby(['method', 'data']).apply(lambda x: roc_auc_score(x['true_class'], x['score'])).groupby('method').describe()

# %%
predictions_avg.groupby(['method', 'data']).apply(lambda x: roc_auc_score(x['true_class'], x['score'])).groupby('method').describe()

# %% [markdown]
# # PR

# %%
from sklearn.metrics import average_precision_score

# %%
predictions.groupby(['method', 'data']).apply(lambda x: average_precision_score(x['true_class'], x['score'])).groupby('method').describe()

# %%
predictions_avg.groupby(['method', 'data']).apply(lambda x: average_precision_score(x['true_class'], x['score'])).groupby('method').describe()

# %%
