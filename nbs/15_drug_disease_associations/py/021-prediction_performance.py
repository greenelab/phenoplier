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
from tqdm import tqdm

import conf

# %% [markdown] papermill={"duration": 0.011416, "end_time": "2020-12-18T22:38:21.683356", "exception": false, "start_time": "2020-12-18T22:38:21.671940", "status": "completed"} tags=[]
# # Settings

# %%
OUTPUT_DIR = conf.RESULTS["DRUG_DISEASE_ANALYSES"]
display(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
OUTPUT_PREDICTIONS_DIR = Path(OUTPUT_DIR, "predictions", "dotprod_neg")
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
gold_standard["true_class"].value_counts()

# %%
gold_standard["true_class"].value_counts(normalize=True)

# %%
doids_in_gold_standard = set(gold_standard["trait"])

# %% [markdown]
# # Load drug-disease predictions

# %%
from collections import defaultdict

# %%
current_prediction_files = list(OUTPUT_PREDICTIONS_DIR.glob("*.h5"))
display(len(current_prediction_files))

# %%
current_prediction_files[:5]

# %%
predictions = []

for f in tqdm(current_prediction_files, ncols=100):
    # exclude S-MultiXcan results, since they have no direction of effect
    if f.name.startswith("smultixcan-"):
        continue

    prediction_data = pd.read_hdf(f, key="prediction")
    prediction_data = pd.merge(
        prediction_data, gold_standard, on=["trait", "drug"], how="inner"
    )

    metadata = pd.read_hdf(f, key="metadata")

    prediction_data["score"] = prediction_data["score"].rank()
    prediction_data["trait"] = prediction_data["trait"].astype("category")
    prediction_data["drug"] = prediction_data["drug"].astype("category")
    prediction_data = prediction_data.assign(method=metadata.method)
    prediction_data = prediction_data.assign(data=metadata.data)

    predictions.append(prediction_data)

# %%
predictions = pd.concat(predictions, ignore_index=True)

# %%
predictions.shape

# %%
predictions.head()

# %% [markdown]
# ## Testing

# %%
# all prediction tables should have the same shape
predictions_shape = (
    predictions.groupby(["method", "data"]).apply(lambda x: x.shape).unique()
)
display(predictions_shape)
assert predictions_shape.shape[0] == 1

# %% [markdown]
# ## Save

# %%
output_file = Path(OUTPUT_DIR, "predictions", "predictions_results.pkl").resolve()
display(output_file)

# %%
predictions.to_pickle(output_file)


# %% [markdown]
# # Aggregate predictions

# %%
def _reduce(x):
    return pd.Series(
        {
            "score": x["score"].max(),
            "true_class": x["true_class"].unique()[0]
            if x["true_class"].unique().shape[0] == 1
            else None,
        }
    )


# %%
predictions_avg = (
    predictions.groupby(["trait", "drug", "method"])
    .apply(_reduce)
    .dropna()
    .sort_index()
    .reset_index()
)

# %%
predictions_avg.shape

# %%
# predictions_avg should have twice the number of rows in the predictions table, since has both methods
assert predictions_avg.shape[0] == int(predictions_shape[0][0] * 2)

# %%
assert predictions_avg.dropna().shape == predictions_avg.shape

# %%
predictions_avg.head()

# %% [markdown]
# ## Save

# %%
output_file = Path(
    OUTPUT_DIR, "predictions", "predictions_results_aggregated.pkl"
).resolve()
display(output_file)

# %%
predictions_avg.to_pickle(output_file)

# %% [markdown]
# # ROC

# %%
from sklearn.metrics import roc_auc_score

# %%
predictions.groupby(["method", "data"]).apply(
    lambda x: roc_auc_score(x["true_class"], x["score"])
).groupby("method").describe()

# %%
predictions_avg.groupby(["method"]).apply(
    lambda x: roc_auc_score(x["true_class"], x["score"])
).groupby("method").describe()

# %% [markdown]
# # PR

# %%
from sklearn.metrics import average_precision_score

# %%
predictions.groupby(["method", "data"]).apply(
    lambda x: average_precision_score(x["true_class"], x["score"])
).groupby("method").describe()

# %%
predictions_avg.groupby(["method"]).apply(
    lambda x: average_precision_score(x["true_class"], x["score"])
).groupby("method").describe()

# %%
