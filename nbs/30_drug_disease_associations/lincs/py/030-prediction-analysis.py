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
# This notebooks analyzes the drug-disease prediction differences between the gene-based and module-based approaches.
# It focuses on pairs discussed in the manuscript.

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

# %% [markdown] tags=[]
# # Data loading

# %% [markdown] tags=[]
# ## PharmacotherapyDB

# %% [markdown] tags=[]
# ### Gold standard set

# %% tags=[]
gold_standard = pd.read_pickle(
    Path(conf.RESULTS["DRUG_DISEASE_ANALYSES"], "gold_standard.pkl"),
)

# %% tags=[]
gold_standard.shape

# %% tags=[]
gold_standard.head()

# %% [markdown] tags=[]
# ### Info

# %% tags=[]
input_file = conf.PHARMACOTHERAPYDB["INDICATIONS_FILE"]
display(input_file)

# %% tags=[]
gold_standard_info = pd.read_csv(input_file, sep="\t")

# %% tags=[]
gold_standard_info = gold_standard_info.rename(columns={"drug": "drug_name"})

# %% tags=[]
gold_standard_info.shape

# %% tags=[]
gold_standard_info.head()

# %% tags=[]
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

# %% tags=[]
output_file = Path(
    conf.RESULTS["DRUG_DISEASE_ANALYSES"],
    "lincs",
    "predictions",
    "predictions_results_aggregated.pkl",
).resolve()
display(output_file)

# %% tags=[]
predictions_avg = pd.read_pickle(output_file)

# %% tags=[]
predictions_avg.shape

# %% tags=[]
predictions_avg.head()

# %% [markdown] tags=[]
# ### Merge with gold standard set

# %% tags=[]
pharmadb_predictions = pd.merge(
    gold_standard_info,
    predictions_avg,
    on=["trait", "drug"],
    how="inner",
)

# %% tags=[]
pharmadb_predictions

# %% tags=[]
pharmadb_predictions = pharmadb_predictions[
    ["trait", "drug", "disease", "drug_name", "method", "score", "true_class_x"]
].rename(columns={"true_class_x": "true_class", "drug_x": "drug"})

# %% tags=[]
display(pharmadb_predictions.shape)
assert pharmadb_predictions.shape[0] == predictions_avg.shape[0]

# %% tags=[]
pharmadb_predictions.head()

# %% tags=[]
pharmadb_predictions["trait"].unique().shape

# %% tags=[]
pharmadb_predictions["drug"].unique().shape

# %% tags=[]
data_stats = pharmadb_predictions.groupby("method")["score"].describe()
display(data_stats)


# %% [markdown] tags=[]
# # Standardize scores for each method

# %% tags=[]
# Standardize scores by method
def _standardize(x):
    return (x["score"] - data_stats.loc[x["method"], "mean"]) / data_stats.loc[
        x["method"], "std"
    ]


# %% tags=[]
pharmadb_predictions = pharmadb_predictions.assign(
    score_std=pharmadb_predictions.apply(_standardize, axis=1)
)

# %% tags=[]
pharmadb_predictions

# %% [markdown] tags=[]
# ### Testing

# %% tags=[]
_tmp = pharmadb_predictions.groupby("method")[["score", "score_std"]].describe()
display(_tmp)

# %% tags=[]
_tmp0 = pharmadb_predictions[(pharmadb_predictions["method"] == "Gene-based")][
    ["score", "score_std"]
]

# %% tags=[]
assert all(_tmp0.corr() > 0.99999)

# %% tags=[]
_tmp0 = pharmadb_predictions[(pharmadb_predictions["method"] == "Module-based")][
    ["score", "score_std"]
]

# %% tags=[]
assert all(_tmp0.corr() > 0.99999)

# %% [markdown] tags=[]
# # List diseases

# %% tags=[]
pharmadb_predictions["disease"].unique()


# %% [markdown] tags=[]
# # Looks for differences in scores of both methods

# %% [markdown]
# Here I'm interested in seeing in which drug-disease pairs the method differ the most.

# %% tags=[]
def _compare(x):
    """
    It takes a DataFrame with the results for a drug-disease pair for each method (gene and module-based) and
    computes whether signs in both scores are different and the absolute value of the scores. For example, if the
    gene-based method assigned a score (standardized) of -0.76 (below the mean) and the module-based a score of 1.56 (above
    the mean), it will indicate that signs are different and return also the difference.
    """
    assert x.shape[0] == 2
    x_sign = np.sign(x["score_std"].values)
    x0 = x.iloc[0]["score_std"]
    x1 = x.iloc[1]["score_std"]

    return pd.Series(
        {"different_sign": x_sign[0] != x_sign[1], "score_difference": np.abs(x0 - x1)}
    )


# %% tags=[]
pharmadb_predictions = pharmadb_predictions.set_index(["trait", "drug"]).join(
    pharmadb_predictions.groupby(["trait", "drug"]).apply(_compare)
)

# %% tags=[]
pharmadb_predictions.head()

# %% [markdown] tags=[]
# ## Across all disease

# %% tags=[]
with pd.option_context(
    "display.max_rows", None, "display.max_columns", None, "max_colwidth", None
):
    _tmp = pharmadb_predictions[(pharmadb_predictions["different_sign"])].sort_values(
        ["score_difference", "drug_name", "method"], ascending=[False, False, False]
    )

    display(_tmp.shape)
    display(_tmp)


# %% tags=[]
def find_differences(trait_name):
    """
    Given the name of a trait, it shows for which drugs both methods provide different scores.
    """
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


# %% [markdown] tags=[]
# # Cardiovascular diseases

# %% [markdown] tags=[]
# Since in the manuscript, in a previous section, we are working on an LV involved lipids metabolism, here we pick Niacin and some cardiovascular traits for further analysis.

# %% [markdown] tags=[]
# ## coronary artery disease

# %% tags=[]
# take a look at CAD and niacin
with pd.option_context(
    "display.max_rows", None, "display.max_columns", None, "max_colwidth", None
):
    _tmp = pharmadb_predictions[
        (pharmadb_predictions["drug_name"] == "Niacin")
        & (pharmadb_predictions["disease"] == "coronary artery disease")
    ].sort_values(
        ["score_difference", "drug_name", "method"], ascending=[False, False, False]
    )
    display(_tmp.head(50))

# %% [markdown]
# For CAD, both methods assigned a positive score for niacin.

# %% tags=[]
find_differences("coronary artery disease")

# %% [markdown] tags=[]
# ## atherosclerosis

# %% tags=[]
# take a look at AT and niacin
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

# %% [markdown]
# For AT, the module-based method assigned a positive score, whereas the gene-based assigned one close to the mean, thus not giving a strong prediction for this pair.

# %% tags=[]
find_differences("atherosclerosis")

# %% tags=[]
