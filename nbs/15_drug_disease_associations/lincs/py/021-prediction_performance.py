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
# This notebook reads the prediction results generated with the `011-prediction-*` notebooks and computes the final performance measures using the gold standard (PharmacotherapyDB).

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

# %% tags=[]
N_TISSUES = 49
N_THRESHOLDS = 5
N_PREDICTIONS = 646

# %% tags=[]
OUTPUT_DIR = conf.RESULTS["DRUG_DISEASE_ANALYSES"] / "lincs"
display(OUTPUT_DIR)
assert OUTPUT_DIR.exists()

# %% tags=[]
OUTPUT_PREDICTIONS_DIR = Path(OUTPUT_DIR, "predictions", "dotprod_neg")
display(OUTPUT_PREDICTIONS_DIR)
OUTPUT_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown] tags=[]
# # Load PharmacotherapyDB gold standard

# %% tags=[]
gold_standard = pd.read_pickle(
    Path(conf.RESULTS["DRUG_DISEASE_ANALYSES"], "gold_standard.pkl"),
)

# %% tags=[]
gold_standard.shape

# %% tags=[]
gold_standard.head()

# %% tags=[]
gold_standard["true_class"].value_counts()

# %% tags=[]
gold_standard["true_class"].value_counts(normalize=True)

# %% [markdown] tags=[]
# # Load drug-disease predictions

# %% tags=[]
from collections import defaultdict

# %% tags=[]
# get all prediction files

current_prediction_files = list(OUTPUT_PREDICTIONS_DIR.glob("*.h5"))
display(len(current_prediction_files))

assert len(current_prediction_files) == 2 * (
    N_TISSUES * N_THRESHOLDS
)  # two methods (single-gene and module-based)

# %% tags=[]
current_prediction_files[:5]

# %% tags=[]
# iterate for each prediction file and perform some preprocessing
# each prediction file (.h5) has the predictions of one method (either module-based
# or gene-based) for all drug-disease pairs across all S-PrediXcan tissues

predictions = []

for f in tqdm(current_prediction_files, ncols=100):
    # get predictions and merge with gold standard, keeping only the drug-disease pairs present there
    prediction_data = pd.read_hdf(f, key="prediction")
    prediction_data = pd.merge(
        prediction_data, gold_standard, on=["trait", "drug"], how="inner"
    )

    # transform scores into ranks, and change the type of columns to save memory
    prediction_data["score"] = prediction_data["score"].rank()
    prediction_data["trait"] = prediction_data["trait"].astype("category")
    prediction_data["drug"] = prediction_data["drug"].astype("category")

    # read metadata
    metadata = pd.read_hdf(f, key="metadata")

    # add the "method" column
    prediction_data = prediction_data.assign(method=metadata.method.values[0])
    prediction_data["method"] = prediction_data["method"].astype("category")

    # add the "n_top_genes" columns, which indicates the top genes/LVs used for this prediction
    prediction_data = prediction_data.assign(n_top_genes=metadata.n_top_genes.values[0])

    # add the "data" column, which has the tissue name
    prediction_data = prediction_data.assign(data=metadata.data.values[0])
    prediction_data["data"] = prediction_data["data"].astype("category")

    predictions.append(prediction_data)

# %% tags=[]
assert np.all(pred.shape[0] == N_PREDICTIONS for pred in predictions)

# %% tags=[]
predictions = pd.concat(predictions, ignore_index=True)


# %% tags=[]
# extract the tissue name from the "data" column


def _get_tissue(x):
    if x.endswith("-projection"):
        return x.split("spredixcan-mashr-zscores-")[1].split("-projection")[0]
    else:
        return x.split("spredixcan-mashr-zscores-")[1].split("-data")[0]


predictions = predictions.assign(tissue=predictions["data"].apply(_get_tissue))

# %% tags=[]
predictions.head()

# %% [markdown] tags=[]
# ## Testing

# %% tags=[]
display(predictions.shape)

assert predictions.shape[0] == 2 * (N_TISSUES * N_THRESHOLDS) * N_PREDICTIONS

# %% tags=[]
assert not predictions.isna().any().any()

# %% tags=[]
_tmp = predictions["method"].value_counts()
display(_tmp)

assert _tmp.loc["Gene-based"] == N_TISSUES * N_THRESHOLDS * N_PREDICTIONS
assert _tmp.loc["Module-based"] == N_TISSUES * N_THRESHOLDS * N_PREDICTIONS

# %% tags=[]
_tmp = predictions.groupby(["method", "n_top_genes"]).count()
display(_tmp)

assert np.all(_tmp == N_TISSUES * N_PREDICTIONS)

# %% tags=[]
_tmp = predictions.groupby(["method", "tissue"]).count()
display(_tmp)

assert np.all(_tmp.loc["Gene-based"] == (N_PREDICTIONS * N_THRESHOLDS))
assert np.all(_tmp.loc["Module-based"] == (N_PREDICTIONS * N_THRESHOLDS))

# %% tags=[]
# all prediction tables should have the same shape
predictions_shape = (
    predictions.groupby(["method", "n_top_genes", "tissue"])
    .apply(lambda x: x.shape)
    .unique()
)
display(predictions_shape)

assert predictions_shape.shape[0] == 1
assert predictions_shape[0][0] == N_PREDICTIONS

# %% [markdown] tags=[]
# ## Save

# %% tags=[]
output_file = Path(OUTPUT_DIR, "predictions", "predictions_results.pkl").resolve()
display(output_file)

# %% tags=[]
predictions.to_pickle(output_file)


# %% [markdown] tags=[]
# # Aggregate predictions

# %% [markdown]
# Here I get summaries from all predictions as follows:
#
#   1. Group by trait, drug, method, tissue, and average all scores across all gene/LVs thresholds. This is the same as it is done in the published method referenced before (the framework for drug-repositioning).
#   1. Then, group by trait, drug, method and take the maximum score across all tissues. The rationale for this is that 1) we don't know which tissue might have more information for a particular disease, and 2) tissue-specific TWAS results are not meaningful to extract conclusions of real tissue-specific effects, since there is a lot of eQTL sharing across tissues.
#
# These correspond to the final drug-disease predictions for each method.

# %% tags=[]
def _reduce_mean(x):
    return pd.Series(
        {"score": x["score"].mean(), "true_class": x["true_class"].unique()[0]}
    )


def _reduce_max(x):
    return pd.Series(
        {"score": x["score"].max(), "true_class": x["true_class"].unique()[0]}
    )


# %% tags=[]
predictions_avg = (
    # average across n_top_genes
    predictions.groupby(["trait", "drug", "method", "tissue"])
    .apply(_reduce_mean)
    .dropna()
    # take maximum across tissues
    .groupby(["trait", "drug", "method"])
    .apply(_reduce_max)
    .dropna()
    .sort_index()
    .reset_index()
)

# %% tags=[]
# predictions_avg should have twice the number of rows in the predictions table, since has both methods
display(predictions_avg.shape)
assert predictions_avg.shape[0] == int(predictions_shape[0][0] * 2)

# %% tags=[]
assert predictions_avg.dropna().shape == predictions_avg.shape

# %% tags=[]
predictions_avg.head()

# %% [markdown] tags=[]
# ## Save

# %% tags=[]
output_file = Path(
    OUTPUT_DIR, "predictions", "predictions_results_aggregated.pkl"
).resolve()
display(output_file)

# %% tags=[]
predictions_avg.to_pickle(output_file)

# %% [markdown] tags=[]
# # ROC

# %% tags=[]
from sklearn.metrics import roc_auc_score

# %% [markdown] tags=[]
# ## Predictions

# %% tags=[]
# AUROC by method/n_top_genes
predictions.groupby(["method", "tissue", "n_top_genes"]).apply(
    lambda x: roc_auc_score(x["true_class"], x["score"])
).groupby(["method", "n_top_genes"]).describe()

# %% tags=[]
# AUROC by method/tissue
predictions.groupby(["method", "tissue", "n_top_genes"]).apply(
    lambda x: roc_auc_score(x["true_class"], x["score"])
).groupby(["method", "tissue"]).describe()

# %% [markdown] tags=[]
# ## Predictions summaries

# %% tags=[]
predictions_avg.head()

# %% tags=[]
predictions_avg.groupby(["method"]).apply(
    lambda x: roc_auc_score(x["true_class"], x["score"])
)

# %% [markdown]
# These are the final performance measures using AUROC.

# %% [markdown] tags=[]
# # PR

# %% tags=[]
from sklearn.metrics import average_precision_score

# %% [markdown] tags=[]
# ## Predictions

# %% tags=[]
# Average precision by method/n_top_genes
predictions.groupby(["method", "tissue", "n_top_genes"]).apply(
    lambda x: average_precision_score(x["true_class"], x["score"])
).groupby(["method", "n_top_genes"]).describe()

# %% tags=[]
# Average precision by method/tissue
predictions.groupby(["method", "tissue", "n_top_genes"]).apply(
    lambda x: average_precision_score(x["true_class"], x["score"])
).groupby(["method", "tissue"]).describe()

# %% [markdown] tags=[]
# ## Predictions summaries

# %% tags=[]
predictions_avg.groupby(["method"]).apply(
    lambda x: average_precision_score(x["true_class"], x["score"])
)

# %% [markdown]
# These are the final performance measures using average precision.

# %% tags=[]
