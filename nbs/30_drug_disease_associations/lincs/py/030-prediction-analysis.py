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
# This notebooks analyzes the drug-disease pairs discussed in the manuscript.

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
from pathlib import Path

import numpy as np
import pandas as pd

from entity import Gene
import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
# OUTPUT_DIR = conf.RESULTS["DRUG_DISEASE_ANALYSES"]
# display(OUTPUT_DIR)

# assert OUTPUT_DIR.exists()
# # OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% tags=[]
# INPUT_DATA_DIR = Path(
#     conf.RESULTS["DRUG_DISEASE_ANALYSES"],
#     "data",
# )
# display(INPUT_DATA_DIR)

# %% tags=[]
# OUTPUT_PREDICTIONS_DIR = Path(
#     conf.RESULTS["DRUG_DISEASE_ANALYSES"], "predictions", "dotprod_neg"
# )
# display(OUTPUT_PREDICTIONS_DIR)

# %% [markdown] tags=[]
# # Data loading

# %% [markdown] tags=[]
# ## PharmacotherapyDB: load gold standard

# %% [markdown]
# ### Final

# %%
gold_standard = pd.read_pickle(
    Path(conf.RESULTS["DRUG_DISEASE_ANALYSES"], "gold_standard.pkl"),
)

# %%
gold_standard.shape

# %%
gold_standard.head()

# %% [markdown]
# ### Info

# %% tags=[]
# TODO hardcoded
input_file = Path(
    conf.DATA_DIR, "hetionet/pharmacotherapydb-v1.0", "indications.tsv"
).resolve()
display(input_file)

# %%
gold_standard_info = pd.read_csv(input_file, sep="\t")

# %%
gold_standard_info = gold_standard_info.rename(columns={"drug": "drug_name"})

# %% tags=[]
gold_standard_info.shape

# %% tags=[]
gold_standard_info.head()

# %%
gold_standard_info = (
    gold_standard.set_index(["trait", "drug"])
    .join(
        gold_standard_info.rename(
            columns={"doid_id": "trait", "drugbank_id": "drug"}
        ).set_index(["trait", "drug"])
    )
    .reset_index()
)

# %% tags=[]
gold_standard_info.shape

# %% tags=[]
gold_standard_info.head()

# %% [markdown] tags=[]
# ## LINCS data

# %% tags=[]
input_file = Path(
    conf.RESULTS["DRUG_DISEASE_ANALYSES"], "lincs", "lincs-data.pkl"
).resolve()

display(input_file)

# %% tags=[]
lincs_data = pd.read_pickle(input_file).T.rename(columns=Gene.GENE_ID_TO_NAME_MAP)

# %% tags=[]
display(lincs_data.shape)

# %% tags=[]
display(lincs_data.head())

# %% [markdown] tags=[]
# ## Prediction results (aggregated)

# %%
output_file = Path(
    conf.RESULTS["DRUG_DISEASE_ANALYSES"],
    "lincs",
    "predictions",
    "predictions_results_aggregated.pkl",
).resolve()
display(output_file)

# %%
predictions_avg = pd.read_pickle(output_file)

# %%
predictions_avg.shape

# %%
predictions_avg.head()

# %% [markdown]
# ### Merge with gold standard set

# %%
pharmadb_predictions = pd.merge(
    gold_standard_info,
    predictions_avg,
    on=["trait", "drug"],
    how="inner",
)

# %%
pharmadb_predictions

# %%
pharmadb_predictions = pharmadb_predictions[
    ["trait", "drug", "disease", "drug_name", "method", "score", "true_class_x"]
].rename(columns={"true_class_x": "true_class", "drug_x": "drug"})

# %%
display(pharmadb_predictions.shape)
assert pharmadb_predictions.shape[0] == predictions_avg.shape[0]

# %%
pharmadb_predictions.head()

# %% tags=[]
pharmadb_predictions["trait"].unique().shape

# %% tags=[]
pharmadb_predictions["drug"].unique().shape

# %%
data_stats = pharmadb_predictions.groupby("method")["score"].describe()
display(data_stats)


# %% [markdown]
# # Standardize scores for each method

# %%
# Standardize scores by method
def _standardize(x):
    return (x["score"] - data_stats.loc[x["method"], "mean"]) / data_stats.loc[
        x["method"], "std"
    ]


# %%
pharmadb_predictions = pharmadb_predictions.assign(
    score_std=pharmadb_predictions.apply(_standardize, axis=1)
)

# %%
pharmadb_predictions

# %% [markdown]
# ### Testing

# %%
_tmp = pharmadb_predictions.groupby("method")[["score", "score_std"]].describe()
display(_tmp)

# %%
_tmp0 = pharmadb_predictions[(pharmadb_predictions["method"] == "Gene-based")][
    ["score", "score_std"]
]

# %%
assert all(_tmp0.corr() > 0.99999)

# %%
_tmp0 = pharmadb_predictions[(pharmadb_predictions["method"] == "Module-based")][
    ["score", "score_std"]
]

# %%
assert all(_tmp0.corr() > 0.99999)

# %% [markdown]
# # List diseases

# %%
pharmadb_predictions["disease"].unique()


# %% [markdown]
# # Looks for differences in scores of both methods

# %%
def _compare(x):
    assert x.shape[0] == 2
    x_sign = np.sign(x["score_std"].values)
    x0 = x.iloc[0]["score_std"]
    x1 = x.iloc[1]["score_std"]

    return pd.Series(
        {"different_sign": x_sign[0] != x_sign[1], "score_difference": np.abs(x0 - x1)}
    )


# %%
pharmadb_predictions = pharmadb_predictions.set_index(["trait", "drug"]).join(
    pharmadb_predictions.groupby(["trait", "drug"]).apply(_compare)
)

# %%
pharmadb_predictions.head()

# %% [markdown]
# ## any disease

# %%
with pd.option_context(
    "display.max_rows", None, "display.max_columns", None, "max_colwidth", None
):
    _tmp = pharmadb_predictions[(pharmadb_predictions["different_sign"])].sort_values(
        ["score_difference", "drug_name", "method"], ascending=[False, False, False]
    )
    display(_tmp.head(50))


# %%
def find_differences(trait_name):
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "max_colwidth", None
    ):
        _tmp = pharmadb_predictions[
            (pharmadb_predictions["disease"] == trait_name)
            & (pharmadb_predictions["different_sign"])
        ].sort_values(
            ["score_difference", "drug_name", "method"], ascending=[False, False, False]
        )
        display(_tmp)


# %% [markdown]
# ## coronary artery disease

# %%
with pd.option_context(
    "display.max_rows", None, "display.max_columns", None, "max_colwidth", None
):
    _tmp = pharmadb_predictions[
        (pharmadb_predictions["drug_name"] == "Niacin")
        & (pharmadb_predictions["disease"] == "coronary artery disease")  # avoid cancer
    ].sort_values(
        ["score_difference", "drug_name", "method"], ascending=[False, False, False]
    )
    display(_tmp.head(50))

# %%
find_differences("coronary artery disease")

# %% [markdown]
# ## atherosclerosis

# %%
with pd.option_context(
    "display.max_rows", None, "display.max_columns", None, "max_colwidth", None
):
    _tmp = pharmadb_predictions[
        (pharmadb_predictions["drug_name"] == "Niacin")
        & (pharmadb_predictions["disease"] == "atherosclerosis")  # avoid cancer
    ].sort_values(
        ["score_difference", "drug_name", "method"], ascending=[False, False, False]
    )
    display(_tmp.head(50))

# %%
find_differences("atherosclerosis")

# %%
