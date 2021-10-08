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
# In the previous notebook, we found that LV603 gene weight's are predictive of gene associations for neutrophil counts.
# In a real application, you would run the `GLSPhenoplier` for your trait of interested across all LVs in our models, and get the significant ones. Then you can see in which cell types the LV's genes are expressed, and this is what we are going to do in this notebook for LV603.
#
# To find the cell types associated with an LV, we'll use the matrix B (see our manuscript).
# We can connected samples in matrix B with gene expression metadata and explore which cell types are associated with the LV.
# However, metadata in gene expression datasets is usually hard to read, process and interpret.
# For example, for some samples that could be important for our LV, we might not have access to the cell types where the expriments were conducted. We'll show here what we can do to try to overcome this.

# %% [markdown]
# # Modules

# %%
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data.recount2 import LVAnalysis
from utils import chunker
import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=["parameters"]
LV_NAME = "LV603"

# %%
# LV_AXIS_THRESHOLD = 3.0
# N_TOP_SAMPLES = 400
# N_TOP_ATTRS = 25

# %%
OUTPUT_FIGURES_DIR = Path(conf.RESULTS_DIR, "demo", f"{LV_NAME.lower()}").resolve()
display(OUTPUT_FIGURES_DIR)
OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# %%
OUTPUT_CELL_TYPE_FILEPATH = OUTPUT_FIGURES_DIR / f"{LV_NAME.lower()}-cell_types.svg"
display(OUTPUT_CELL_TYPE_FILEPATH)

# %% [markdown] tags=[]
# # Load MultiPLIER summary

# %% tags=[]
multiplier_model_summary = pd.read_pickle(conf.MULTIPLIER["MODEL_SUMMARY_FILE"])

# %% tags=[]
multiplier_model_summary.shape

# %% tags=[]
multiplier_model_summary.head()

# %% [markdown] tags=[]
# # Load data

# %% [markdown] tags=[]
# ## Original data

# %% tags=[]
# INPUT_SUBSET = "z_score_std"

# %% tags=[]
# INPUT_STEM = "projection-smultixcan-efo_partial-mashr-zscores"

# %% tags=[]
# input_filepath = Path(
#     conf.RESULTS["DATA_TRANSFORMATIONS_DIR"],
#     INPUT_SUBSET,
#     f"{INPUT_SUBSET}-{INPUT_STEM}.pkl",
# ).resolve()
# display(input_filepath)

# assert input_filepath.exists(), "Input file does not exist"

# input_filepath_stem = input_filepath.stem
# display(input_filepath_stem)

# %% tags=[]
# data = pd.read_pickle(input_filepath)

# %% tags=[]
# data.shape

# %% tags=[]
# data.head()

# %% [markdown]
# ## LV data

# %%
lv_obj = LVAnalysis(LV_NAME)  # FIXME: does it work without data as argument?

# %%
lv_obj.lv_genes.head(20)

# %%
multiplier_model_summary[
    multiplier_model_summary["LV index"].isin((LV_NAME[2:],))
    & (
        (multiplier_model_summary["FDR"] < 0.05)
        & (multiplier_model_summary["AUC"] >= 0.75)
    )
]

# %%
lv_data = lv_obj.get_experiments_data()

# %%
lv_data.shape

# %%
lv_data.head()

# %% [markdown]
# # LV cell types analysis

# %% [markdown]
# ## Get top attributes

# %%
lv_attrs = lv_obj.get_attributes_variation_score()
display(lv_attrs.head(20))

# %%
# show those with cell type or tissue in their name
_tmp = pd.Series(lv_attrs.index)
lv_attrs[
    _tmp.str.match(
        "(?:cell.*(type|line)$)|(?:tissue$)|(?:tissue.*type$)",
        case=False,
        flags=re.IGNORECASE,
    ).values
].sort_values(ascending=False)

# %%
lv_attrs_selected = [
    "cell type",
    "celltype",
    "tissue",
]

# %%
_tmp = lv_data.loc[
    :,
    lv_attrs_selected + [LV_NAME],
]

# %%
_tmp_seq = list(chunker(_tmp.sort_values(LV_NAME, ascending=False), 25))

# %%
_tmp_seq[0]

# %%
# what is there in these projects?
lv_data.loc[["SRP015360"]].dropna(how="all", axis=1).sort_values(
    LV_NAME, ascending=False
).head(10)

# %% [markdown]
# MENTION HERE THAT YOU CAN GO TO THIS PLACE FOR THIS SRP: https://trace.ncbi.nlm.nih.gov/Traces/sra/?study=SRP015360
#
# And in this case we see it's neutrophils under different stimuli.

# %%
# TODO: select here in the order desired.
SELECTED_ATTRIBUTES = [
    "cell type",
    "celltype",
    "tissue",
]

# %%
# it has to be in the order desired for filling nans in the SELECTED_ATTRIBUTE
# SECOND_ATTRIBUTES = ["celltype", ]

# %% [markdown]
# ## Get plot data

# %%
plot_data = lv_data.loc[:, SELECTED_ATTRIBUTES + [LV_NAME]]

# %%
# if blank/nan, fill cell type column with tissue content
_new_column = plot_data[SELECTED_ATTRIBUTES].fillna(method="backfill", axis=1)[
    SELECTED_ATTRIBUTES[0]
]
plot_data[SELECTED_ATTRIBUTES[0]] = _new_column
plot_data = plot_data.drop(columns=SELECTED_ATTRIBUTES[1:])
plot_data = plot_data.fillna({SELECTED_ATTRIBUTES[0]: "NOT CATEGORIZED"})

# %%
plot_data = plot_data.sort_values(LV_NAME, ascending=False)

# %%
plot_data.head(20)

# %% [markdown]
# You can see that now all attributes ("celltype", "cell type" and "tissue") are combined under a single attribute named "cell type".
# For example, if you look at `SRP045500`, which had a value under "celltype" but an empty (NaN) value for attribute "cell type", that now the value was moved to "cell type" (it was unified into this single attribute).

# %% [markdown]
# ## Customize x-axis values

# %% [markdown]
# When cell type values are not very clear, customize their names by looking at their specific studies to know exactly what the authors meant.

# %%
final_plot_data = plot_data.replace(
    {
        SELECTED_ATTRIBUTES[0]: {
            # neutrophils:
            "primary human neutrophils": "Neutrophils",
            "Neutrophil isolated from peripheral blood": "Neutrophils",
            "Neutrophil": "Neutrophils",
            "neutrophils (Neu)": "Neutrophils",
            # granulocytes:
            "granulocyte": "Granulocytes",
            # monocytes:
            "primary human monocytes": "Monocytes",
            # whole blood:
            #             "whole blood": "Whole blood",  # uncomment this line to merge occurences of "whole blood" into "Whole blood"
            "Whole Blood": "Whole blood",
            # PBMC:
            "primary human PBMC": "PBMC",
            # B-cells:
            "primary human B cells": "B cells",
            # T-cells:
            "primary human T cells": "T cells",
            # epithelial cells:
            "epithelial cells (Epi)": "Epithelial cells",
            "primary human myeloid DC": "mDCs",
        }
    }
)

# %%
# # add also tissue information to these projects
# _srp_code = "SRP061881"
# _tmp = final_plot_data.loc[(_srp_code,)].apply(
#     lambda x: lv_data.loc[(_srp_code, x.name), "cell type"]
#     + f" ({lv_data.loc[(_srp_code, x.name), 'tissue']})",
#     axis=1,
# )
# final_plot_data.loc[(_srp_code, _tmp.index), SELECTED_ATTRIBUTE] = _tmp.values


# _srp_code = "SRP056049"
# _tmp = final_plot_data.loc[(_srp_code,)].apply(
#     lambda x: lv_data.loc[(_srp_code, x.name), "cell population"]
#     + f" ({lv_data.loc[(_srp_code, x.name), 'diagnosis']})",
#     axis=1,
# )
# final_plot_data.loc[(_srp_code, _tmp.index), SELECTED_ATTRIBUTE] = _tmp.values


# _srp_code = "SRP059057"
# _tmp = final_plot_data.loc[(_srp_code,)].apply(
#     lambda x: lv_data.loc[(_srp_code, x.name), "cell stimulation"]
#     + f" ({lv_data.loc[(_srp_code, x.name), 'phenotype']})",
#     axis=1,
# )
# final_plot_data.loc[(_srp_code, _tmp.index), SELECTED_ATTRIBUTE] = _tmp.values

# %%
# # all samples from SRP049593 are fibroblasts
# final_plot_data[SELECTED_ATTRIBUTES[0]] = final_plot_data.apply(
#     lambda x: "Myeloma cells" if x.name[0] in ("SRP027015",) else x["cell type"],
#     axis=1,
# )

# %%
# take the top samples only
final_plot_data = final_plot_data.sort_values(
    LV_NAME, ascending=False
)  # [:N_TOP_SAMPLES]

# %% [markdown]
# ## Threshold LV values

# %%
# final_plot_data.loc[
#     final_plot_data[LV_NAME] > LV_AXIS_THRESHOLD, LV_NAME
# ] = LV_AXIS_THRESHOLD

# %% [markdown]
# ## Delete samples with no tissue/cell type information

# %%
# final_plot_data = final_plot_data[
#     final_plot_data[SELECTED_ATTRIBUTE] != "NOT CATEGORIZED"
# ]

# %% [markdown]
# ## Set x-axis order

# %%
N_TOP_ATTRS = 10

# %%
attr_order = (
    final_plot_data.groupby(SELECTED_ATTRIBUTES[0])
    .max()
    .sort_values(LV_NAME, ascending=False)
    .index[:N_TOP_ATTRS]
    .tolist()
)

# %%
len(attr_order)

# %%
attr_order[:5]

# %% [markdown]
# ## Plot

# %%
with sns.plotting_context("paper", font_scale=1.5), sns.axes_style("whitegrid"):
    sns.catplot(
        data=final_plot_data,
        y=LV_NAME,
        x=SELECTED_ATTRIBUTES[0],
        order=attr_order,
        kind="strip",
        height=5,
        aspect=3,
    )
    plt.xticks(rotation=45, horizontalalignment="right")

#     plt.savefig(
#         OUTPUT_CELL_TYPE_FILEPATH,
#         bbox_inches="tight",
#         facecolor="white",
#     )

# %% [markdown]
# # Debug

# %%
with pd.option_context(
    "display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None
):
    _tmp = final_plot_data[
        final_plot_data[SELECTED_ATTRIBUTES[0]].str.contains("NOT CAT")
    ]
    display(_tmp.head(20))

# %%
# what is there in these projects?
lv_data.loc[["SRP015360"]].dropna(how="all", axis=1).sort_values(
    LV_NAME, ascending=False
).head(60)

# %%
