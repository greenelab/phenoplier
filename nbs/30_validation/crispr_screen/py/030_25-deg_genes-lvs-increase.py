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
# TODO

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
from pathlib import Path

import pandas as pd
from IPython.display import HTML

from entity import Trait
from data.cache import read_data
import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
EXPERIMENT_NAME = "lv"
LIPIDS_GENE_SET = "gene_set_increase"

# %% tags=[]
OUTPUT_DIR = Path(
    conf.RESULTS["CRISPR_ANALYSES"]["BASE_DIR"], f"{EXPERIMENT_NAME}-{LIPIDS_GENE_SET}"
)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
display(OUTPUT_DIR)

# %% [markdown] tags=[]
# # Data loading

# %% [markdown] tags=[]
# ## PhenomeXcan projection

# %% tags=[]
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

# %% [markdown] tags=[]
# ## LVs enrichment on DEG from CRISPR screen

# %% tags=[]
deg_enrich = pd.read_csv(
    Path(conf.RESULTS["CRISPR_ANALYSES"]["BASE_DIR"], "fgsea-all_lvs.tsv").resolve(),
    sep="\t",
)

# %% tags=[]
deg_enrich.shape

# %% tags=[]
deg_enrich.head()

# %% [markdown] tags=[]
# ## MultiPLIER summary

# %% tags=[]
multiplier_model_summary = read_data(conf.MULTIPLIER["MODEL_SUMMARY_FILE"])

# %% tags=[]
multiplier_model_summary.shape

# %% tags=[]
multiplier_model_summary.head()

# %% [markdown] tags=[]
# # Analysis

# %% tags=[]
df = deg_enrich[
    deg_enrich["pathway"].isin((LIPIDS_GENE_SET,)) & (deg_enrich["padj"] < 0.05)
].sort_values("padj", ascending=True)

# %% tags=[]
df.shape

# %% tags=[]
df.head()

# %% tags=[]
important_lvs = df["lv"].unique()

# %% tags=[]
display(important_lvs.shape)
assert important_lvs.shape[0] == 27

# %% tags=[]
important_lvs

# %% [markdown] tags=[]
# # Get top traits and pathways

# %% tags=[]
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

    traits.append(_tmp)

# %% [markdown] tags=[]
# ## Pathways

# %% tags=[]
pathways_df = pd.Series(pathways).value_counts()
display(pathways_df)

# %% tags=[]
output_file = Path(OUTPUT_DIR, "pathways_counts.pkl").resolve()
display(output_file)

# %% tags=[]
pathways_df.to_pickle(output_file)

# %% [markdown] tags=[]
# ## Traits

# %% tags=[]
traits_df = (
    pd.concat(traits)
    .reset_index()
    .groupby("index")
    .sum()
    .sort_values(0, ascending=False)
    .reset_index()
).rename(columns={"index": "trait", 0: "value"})

# %% tags=[]
# add trait category
trait_code_to_trait_obj = [
    Trait.get_trait(full_code=t)
    if not Trait.is_efo_label(t)
    else Trait.get_traits_from_efo(t)
    for t in traits_df["trait"]
]

# %% tags=[]
traits_df = traits_df.assign(
    category=[
        t.category if not isinstance(t, list) else t[0].category
        for t in trait_code_to_trait_obj
    ]
)

# %% tags=[]
traits_df.shape

# %% tags=[]
traits_df.head()

# %% tags=[]
output_file = Path(OUTPUT_DIR, "traits.pkl").resolve()
display(output_file)

# %% tags=[]
traits_df.to_pickle(output_file)

# %% [markdown] tags=[]
# # Summary

# %% tags=[]
top_traits = traits_df.head(100)

# %% tags=[]
with pd.option_context(
    "display.max_rows", None, "display.max_columns", None, "max_colwidth", None
):
    display(top_traits)

# %% [markdown] tags=[]
# # Summary using trait categories

# %% tags=[]
top_traits_categories = (
    top_traits.groupby("category")
    .mean()
    .sort_values("value", ascending=False)
    .reset_index()
)

# %% tags=[]
top_traits_categories.head()

# %% tags=[]
with pd.option_context(
    "display.max_rows", None, "display.max_columns", None, "max_colwidth", None
):
    display(top_traits_categories)

# %% tags=[]
for row_idx, row in top_traits_categories.iterrows():
    category = row["category"]
    display(HTML(f"<h2>{category}</h2>"))

    _df = (
        top_traits[top_traits["category"] == category]
        .groupby("trait")["value"]
        .mean()
        .sort_values()
    )
    display(_df)

# %% tags=[]
