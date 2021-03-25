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

# %% [markdown]
# # Description

# %% [markdown]
# TODO

# %% [markdown]
# # Modules loading

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import HTML

# from entity import Trait
# from clustering.methods import ClusterInterpreter
# from data.recount2 import LVAnalysis
from data.cache import read_data

# from utils import generate_result_set_name
import conf

# %% [markdown]
# # Settings

# %% [markdown]
# # Data loading

# %% [markdown] tags=[]
# ## PhenomeXcan projection

# %%
input_filepath = Path(
    conf.RESULTS["PROJECTIONS_DIR"],
    "projection-smultixcan-efo_partial-mashr-zscores.pkl",
).resolve()
display(input_filepath)

# %% tags=[]
phenomexcan_data = pd.read_pickle(input_filepath).T

# %% tags=[]
phenomexcan_data.shape

# %% tags=[]
phenomexcan_data.head()

# %% [markdown]
# ## LVs enrichment on DEG from CRISPR screen

# %%
deg_enrich = pd.read_csv(
    "/home/miltondp/projects/labs/greenelab/phenoplier/base/data/crispr_screen/fsgea-all_lvs.tsv",
    sep="\t",
)

# %%
deg_enrich.shape

# %%
deg_enrich.head()

# %%
deg_enrich_max_idx = deg_enrich.groupby(["lv", "pathway"])["padj"].idxmax()

# %%
deg_enrich = deg_enrich.loc[deg_enrich_max_idx].reset_index(drop=True)
display(deg_enrich.shape)
display(deg_enrich.head())

# %% [markdown]
# ## MultiPLIER summary

# %%
multiplier_model_summary = read_data(conf.MULTIPLIER["MODEL_SUMMARY_FILE"])

# %%
multiplier_model_summary.shape

# %%
multiplier_model_summary.head()

# %% [markdown]
# # Analysis

# %%
df = deg_enrich[
    deg_enrich["pathway"].isin(
        (
            #             "gene_set_decrease_-2_and_-3",
            "gene_set_increase_2_and_3",
        )
    )
    & (deg_enrich["padj"] < 0.05)
]

# %%
df.shape


# %%
def _get_number(x):
    if "gene_set_decrease_" in x:
        return -1
    elif "gene_set_increase_" in x:
        return 1
    else:
        raise ValueError("Unknown")


# df = df.assign(pathway_number=df["pathway"].apply(lambda x: int(x.split("_set_")[1])))
df = df.assign(pathway_number=df["pathway"].apply(_get_number))

# df = df.assign(
#     pathway_number_abs=df["pathway"].apply(lambda x: np.abs(_get_number(x)))
# )

# %%
df = df.sort_values(["pathway_number", "padj"], ascending=[False, True])

# %%
df.shape

# %%
df

# %%
important_lvs = df["lv"].unique()

# %%
important_lvs.shape

# %%
important_lvs

# %% [markdown]
# # Summary

# %%
pathways = []
traits = []

for lv in important_lvs:
    _tmp = multiplier_model_summary[
        (multiplier_model_summary["LV index"] == lv[2:])
        & (
            (multiplier_model_summary["FDR"] < 0.05)
            | (multiplier_model_summary["AUC"] > 0.75)
        )
    ]
    pathways.extend(_tmp["pathway"].tolist())

    _tmp = phenomexcan_data[lv]
    _tmp = _tmp[_tmp > 0.0].sort_values(ascending=False)

    #     _tmp = _tmp.head(50)
    traits.append(_tmp)

# %%
pd.Series(pathways).value_counts().head(20)

# %%
_tmp = (
    pd.concat(traits)
    .reset_index()
    .groupby("index")
    .sum()
    .sort_values(0, ascending=False)
    .reset_index()
).rename(columns={"index": "trait", 0: "value"})

# %%
top_traits = _tmp.head(100)

# %%
with pd.option_context(
    "display.max_rows", None, "display.max_columns", None, "max_colwidth", None
):
    display(top_traits)

# %% [markdown]
# # Summary using trait categories

# %%
from entity import Trait

# %%
trait_code_to_trait_obj = [
    Trait.get_trait(full_code=t)
    if not Trait.is_efo_label(t)
    else Trait.get_traits_from_efo(t)
    for t in top_traits["trait"]
]

# %%
top_traits = top_traits.assign(
    category=[
        t.category
        if not isinstance(t, list)
        else t[0].category  # FIXME just taking the first one
        for t in trait_code_to_trait_obj
    ]
)

# for t in trait_code_to_trait_obj:
#     _tmp = multiplier_model_summary[
#         (multiplier_model_summary["LV index"] == lv[2:])
#         & (
#             (multiplier_model_summary["FDR"] < 0.05)
#             | (multiplier_model_summary["AUC"] > 0.75)
#         )
#     ]
#     pathways.extend(_tmp["pathway"].tolist())

#     _tmp = phenomexcan_data[lv]
#     _tmp = _tmp[_tmp > 0.0].sort_values(ascending=False)

#     _tmp = _tmp.head(50)

#     _tmp = _tmp.reset_index().rename(columns={lv: "lv", "index": "trait"})

#     _tmp["trait_category"] = [
#         trait_code_to_trait_obj[t].category
#         if not isinstance(trait_code_to_trait_obj[t], list)
#         else trait_code_to_trait_obj[t][0].category
#         for t in _tmp["trait"]
#     ]

#     traits.append(_tmp)

# %%
top_traits.shape

# %%
top_traits.head()

# %%
_tmp = (
    top_traits.groupby("category")
    .mean()
    .sort_values("value", ascending=False)
    .reset_index()
)

# %%
_tmp.head()

# %%
with pd.option_context(
    "display.max_rows", None, "display.max_columns", None, "max_colwidth", None
):
    display(_tmp)

# %%
for row_idx, row in _tmp.iterrows():
    category = row["category"]
    display(HTML(f"<h2>{category}</h2>"))

    _df = (
        top_traits[top_traits["category"] == category]
        .groupby("trait")["value"]
        .mean()
        .sort_values()
    )
    display(_df)

# %%
