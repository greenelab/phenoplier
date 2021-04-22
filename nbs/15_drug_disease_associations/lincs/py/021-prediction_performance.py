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
# **TODO**

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

import conf

# %% [markdown] tags=[]
# # Settings

# %%
N_TISSUES = 49
# N_TISSUES = 1
N_THRESHOLDS = 5
N_PREDICTIONS = 646

# %%
OUTPUT_DIR = conf.RESULTS["DRUG_DISEASE_ANALYSES"] / "lincs"
display(OUTPUT_DIR)
assert OUTPUT_DIR.exists()

# %%
OUTPUT_PREDICTIONS_DIR = Path(OUTPUT_DIR, "predictions", "dotprod_neg")
display(OUTPUT_PREDICTIONS_DIR)
OUTPUT_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# # Load PharmacotherapyDB gold standard

# %%
gold_standard = pd.read_pickle(
    Path(conf.RESULTS["DRUG_DISEASE_ANALYSES"], "gold_standard.pkl"),
)

# %%
gold_standard.shape

# %%
gold_standard.head()

# %%
gold_standard["true_class"].value_counts()

# %%
gold_standard["true_class"].value_counts(normalize=True)

# %% [markdown]
# # Load drug-disease predictions

# %%
from collections import defaultdict

# %%
current_prediction_files = list(OUTPUT_PREDICTIONS_DIR.glob("*.h5"))
display(len(current_prediction_files))

assert len(current_prediction_files) == 2 * (
    N_TISSUES * N_THRESHOLDS
)  # two methods (single-gene and module-based)

# %%
current_prediction_files[:5]

# %%
predictions = []

for f in tqdm(current_prediction_files, ncols=100):
    # FIXME: it shouldn't be necessary to include this anymore
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

    prediction_data = prediction_data.assign(method=metadata.method.values[0])
    prediction_data["method"] = prediction_data["method"].astype("category")

    prediction_data = prediction_data.assign(n_top_genes=metadata.n_top_genes.values[0])
    #     prediction_data["n_top_genes"] = prediction_data["data"].astype("category")

    prediction_data = prediction_data.assign(data=metadata.data.values[0])
    prediction_data["data"] = prediction_data["data"].astype("category")

    predictions.append(prediction_data)

# %%
assert np.all(pred.shape[0] == N_PREDICTIONS for pred in predictions)

# %%
predictions = pd.concat(predictions, ignore_index=True)

# %%
display(predictions.shape)

assert predictions.shape[0] == 2 * (N_TISSUES * N_THRESHOLDS) * N_PREDICTIONS

# %%
predictions.head()

# %%
assert not predictions.isna().any().any()

# %%
_tmp = predictions["method"].value_counts()
display(_tmp)

assert _tmp.loc["Gene-based"] == N_TISSUES * N_THRESHOLDS * N_PREDICTIONS
assert _tmp.loc["Module-based"] == N_TISSUES * N_THRESHOLDS * N_PREDICTIONS

# %%
_tmp = predictions.groupby(["method", "n_top_genes"]).count()
display(_tmp)

assert np.all(_tmp == N_TISSUES * N_PREDICTIONS)


# %%
# FIXME: add this to the 011 notebooks... or maybe it's fine here (after submitting draft)
def _get_tissue(x):
    if x.endswith("-projection"):
        return x.split("spredixcan-mashr-zscores-")[1].split("-projection")[0]
    else:
        return x.split("spredixcan-mashr-zscores-")[1].split("-data")[0]


predictions = predictions.assign(tissue=predictions["data"].apply(_get_tissue))

# # FIXME: remove or better document; here just for the most_signif version
# predictions = predictions.assign(tissue="most_signif")

# %%
predictions.head()

# %%
_tmp = predictions.groupby(["method", "tissue"]).count()
display(_tmp)

assert np.all(_tmp.loc["Gene-based"] == (N_PREDICTIONS * N_THRESHOLDS))
assert np.all(_tmp.loc["Module-based"] == (N_PREDICTIONS * N_THRESHOLDS))

# %% [markdown]
# ## Testing

# %%
# all prediction tables should have the same shape
predictions_shape = (
    predictions.groupby(["method", "n_top_genes", "tissue"])
    .apply(lambda x: x.shape)
    .unique()
)
display(predictions_shape)

assert predictions_shape.shape[0] == 1
assert predictions_shape[0][0] == N_PREDICTIONS

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
def _reduce_mean(x):
    return pd.Series(
        {
            "score": x["score"].mean(),
            "true_class": x["true_class"].unique()[0]
            #             if x["true_class"].unique().shape[0] == 1
            #             else None,
        }
    )


def _reduce_max(x):
    return pd.Series(
        {
            "score": x["score"].max(),
            "true_class": x["true_class"].unique()[0]
            #             if x["true_class"].unique().shape[0] == 1
            #             else None,
        }
    )


# def _reduce_best(x):
# #     assert x["true_class"].unique() == FINISH
#     x_stand = (x["score"] - x["score"].mean()) / x["score"].std()

#     x_max_score = x_stand.max()
#     x_min_score = x_stand.min()

#     # select best score
#     x_selected = x["score"].max()
#     if abs(x_min_score) > abs(x_max_score):
#         x_selected = x["score"].min()

#     return pd.Series(
#         {
#             "score": x_selected,
#             "true_class": x["true_class"].unique()[0]
#             #             if x["true_class"].unique().shape[0] == 1
#             #             else None,
#         }
#     )

# %%
predictions_avg = (
    predictions.groupby(["trait", "drug", "method", "tissue"])
    #     predictions.groupby(["trait", "drug", "method"])
    .apply(_reduce_mean)
    .dropna()
    .groupby(["trait", "drug", "method"])
    .apply(_reduce_max)
    .dropna()
    .sort_index()
    .reset_index()
)

# %%
# predictions_avg should have twice the number of rows in the predictions table, since has both methods
display(predictions_avg.shape)
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

# %% [markdown]
# ## Predictions

# %%
# by method/n_top_genes
predictions.groupby(["method", "tissue", "n_top_genes"]).apply(
    lambda x: roc_auc_score(x["true_class"], x["score"])
).groupby(["method", "n_top_genes"]).describe()

# %%
# by method/tissue
predictions.groupby(["method", "tissue", "n_top_genes"]).apply(
    lambda x: roc_auc_score(x["true_class"], x["score"])
).groupby(["method", "tissue"]).describe()

# %% [markdown]
# ## Predictions avg

# %%
predictions_avg.head()

# %%
predictions_avg.groupby(["method"]).apply(
    lambda x: roc_auc_score(x["true_class"], x["score"])
)

# %% [markdown]
# # PR

# %%
from sklearn.metrics import average_precision_score

# %% [markdown]
# ## Predictions

# %%
# by method/n_top_genes
predictions.groupby(["method", "tissue", "n_top_genes"]).apply(
    lambda x: average_precision_score(x["true_class"], x["score"])
).groupby(["method", "n_top_genes"]).describe()

# %%
# by method/tissue
predictions.groupby(["method", "tissue", "n_top_genes"]).apply(
    lambda x: average_precision_score(x["true_class"], x["score"])
).groupby(["method", "tissue"]).describe()

# %% [markdown]
# ## Predictions avg

# %%
predictions_avg.groupby(["method"]).apply(
    lambda x: average_precision_score(x["true_class"], x["score"])
)

# %%
