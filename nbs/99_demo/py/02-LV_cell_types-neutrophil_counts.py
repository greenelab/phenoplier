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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Description

# %% [markdown]
# In the previous notebook, we found that LV603 gene weight's are predictive of gene associations for neutrophil counts.
# In a real application, you would run the `GLSPhenoplier` for your trait of interested across all LVs in our models, and get the significant ones. Then you can see in which cell types the LVs' genes are expressed, and this is what we are going to do in this notebook for LV603.
#
# To find the cell types associated with an LV, we'll use matrix B (see our [manuscript](https://greenelab.github.io/phenoplier_manuscript/#phenoplier-an-integration-framework-based-on-gene-co-expression-patterns)).
# We can link RNA-seq samples in matrix B with gene expression metadata and explore which cell types are associated with the LV.
# However, metadata in gene expression datasets is usually hard to read, process and interpret, and many times important attributes (such as `tissue` or `cell type` are missing).
# We'll show here what we can do to try to overcome this.

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

# %% [markdown] tags=[]
# Specify below the LV id you are interested in:

# %% tags=["parameters"]
LV_NAME = "LV603"

# %% [markdown] tags=[]
# # Paths

# %% [markdown] tags=[]
# These are paths to folder and files where we'll save our figures.

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
# # Load LV data

# %% [markdown] tags=[]
# You can use the `LVAnalysis` class to explore an LV.

# %%
lv_obj = LVAnalysis(LV_NAME)

# %% [markdown] tags=[]
# Here I show the top 20 genes for our LV. You can see gene symbols, the LV weight (in column `LV603`) and the cytoband.

# %%
lv_obj.lv_genes.head(20)

# %% [markdown]
# Show the pathways our LV is aligned to (neutrophils):

# %%
multiplier_model_summary[
    multiplier_model_summary["LV index"].isin((LV_NAME[2:],))
    & (
        (multiplier_model_summary["FDR"] < 0.05)
        & (multiplier_model_summary["AUC"] >= 0.75)
    )
]

# %% [markdown]
# Here I load the LV metadata.
# This is, for each sample from our matrix B, we load its metadata.
# The `LVAnalysis` class takes care of downloading all necessary files from recount2.

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

# %% [markdown]
# Here I quickly show the top attributes in our metadata for which we have the largest LV variance.
# This is just to have an idea of the variance across different attributes

# %%
lv_attrs = lv_obj.get_attributes_variation_score()
display(lv_attrs.head(20))

# %% [markdown]
# Since we are interested in cell types and tissues, the code below will find which attributes contain "cell type" or "tissue", so we can select from these results.

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

# %% [markdown]
# Select from the output above those you are interested in. Usually all of them.

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

# %% [markdown]
# Now that we selected attributes that might be providing information about the cell type, we can see which are the top ones.
#
# If you change the number in the cell below, you can paginate over the entire set of results; for example, with `_tmp_seq[1]` you'll see the next "page" in a descending order with respect to the LV value. So the ones at the top are the most important ones.

# %%
_tmp_seq[0]

# %% [markdown]
# You can see above that the RNA-seq samples are from neutrophils or granulocytes.
# If we move to the next "page", we'll see this:

# %%
_tmp_seq[1]

# %% [markdown]
# Although the attributes we selected (`cell type`, `celltype` and `tissue`) seem to be enough for most of the RNA-seq samples to get their cell types, this is not the case for some of them. For example, we have empty values (NaN) for `SRP015360`.
#
# You can explore the metadata provided for a particular SPR with the code below. Let's see what we have in `SRP015360`:

# %%
# list the top 10 samples from this project
lv_data.loc[["SRP015360"]].dropna(how="all", axis=1).sort_values(
    LV_NAME, ascending=False
).head(10)

# %% [markdown]
# So for `SRP015360` we only have three attributes: `age`, `Sex` and `treatment`. No cell type information. Here you can start grasping the challenges in analyzing this data.
#
# If you are really interested in this metadata for this project, you can use this URL:
# ```
# https://trace.ncbi.nlm.nih.gov/Traces/sra/?study=SRP015360
# ```
# where you can put the SRP code you want. If you check the URL above, you'll see that these samples were taken from neutrophils as well (treated with different stimuli), although we don't have metadata about the cell type/tissue.

# %% [markdown]
# Ok, we do not want several attributes conveying the same information (cell types), just a single one, so here we'll combine the attributes we found before with a particular _order_.
# In this case I select `cell type` _first_ (the order is important), `celltype` as second, and `tissue` as third/last.
# This means that I will use the information in `cell type` first, but if this is empty (as in `SRP045500`), we'll try with `celltype`, and so on.

# %%
SELECTED_ATTRIBUTES = [
    "cell type",
    "celltype",
    "tissue",
]

# %% [markdown]
# ## Get a single attribute with "cell type/tissue" information

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
#
# If you see "NOT CATEGORIZED", it means that none of the attributes you used had a value (they were all empty or NaN). In this case you would go the URL before and read the article/RNA-seq data description to find out the cell type or tissue (or whatever attribute your are interested in).

# %% [markdown]
# ## Standardize cell type names

# %% [markdown]
# When cell type values are different but represent the same cell type, we unify them:

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

# %% [markdown]
# The code below is more advanced, but you can use it to customize even further your final attribute value (for example, if you want to use the cell type, but put the treatment name with parenthesis).

# %%
# # add also tissue information to these projects
# _srp_code = "SRP061881"
# _tmp = final_plot_data.loc[(_srp_code,)].apply(
#     lambda x: lv_data.loc[(_srp_code, x.name), "cell type"]
#     + f" ({lv_data.loc[(_srp_code, x.name), 'tissue']})",
#     axis=1,
# )
# final_plot_data.loc[(_srp_code, _tmp.index), SELECTED_ATTRIBUTES[0]] = _tmp.values

# %%
# # all samples from SRP015360 are neutrophils
# final_plot_data[SELECTED_ATTRIBUTES[0]] = final_plot_data.apply(
#     lambda x: "Neutrophils" if x.name[0] in ("SRP015360",) else x[SELECTED_ATTRIBUTES[0]],
#     axis=1,
# )

# %%
final_plot_data = final_plot_data.sort_values(LV_NAME, ascending=False)

# %% [markdown]
# ## Threshold LV values

# %% [markdown]
# The code below is useful in case you want to put a threshold to the plot (for example, you have same samples with very high values).

# %%
# final_plot_data.loc[
#     final_plot_data[LV_NAME] > LV_AXIS_THRESHOLD, LV_NAME
# ] = LV_AXIS_THRESHOLD

# %% [markdown]
# ## Delete samples with no tissue/cell type information

# %% [markdown]
# If you are not interested in samples for which we couldn't find cell type/tissue information, you can delete them by uncommenting the code below.
# However, if you want, you can take a look at those later (we'll show you how at the end).

# %%
# final_plot_data = final_plot_data[
#     final_plot_data[SELECTED_ATTRIBUTE] != "NOT CATEGORIZED"
# ]

# %% [markdown]
# ## Set top cell types to show

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
# # Plot

# %% [markdown]
# Now we create the plot with RNA-seq samples as points, cell types in the x-axis, and the LV value in the y-axis.
# This shows you in which cell types genes in LV603 are primirily expressed.

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

    # You can save the figure if you uncomment the code below
    # the figure will be created under ${DATA_FOLDER}/results/demo

#     plt.savefig(
#         OUTPUT_CELL_TYPE_FILEPATH,
#         bbox_inches="tight",
#         facecolor="white",
#     )

# %% [markdown]
# # Debug

# %% [markdown]
# Now that you have the figure for your LV, you can debug some things.
# For example, you'll see above that some samples are "NOT CATEGORIZED".
# Let's see which are those:

# %%
with pd.option_context(
    "display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None
):
    _tmp = final_plot_data[
        final_plot_data[SELECTED_ATTRIBUTES[0]].str.contains("NOT CAT")
    ]
    display(_tmp.head(20))

# %% [markdown]
# We don't have information about cell type/tissue for these.
# But we can manually go to the URL showed before and check out.
# For example, for `SRP015360` we can open our browser with this URL:
# https://trace.ncbi.nlm.nih.gov/Traces/sra/?study=SRP015360, and see that the samples were taken from neutrophils.
#
# If you need to see which information is provided by each SRP, you can again use the code below:

# %%
# what is there in these projects?
lv_data.loc[["SRP015360"]].dropna(how="all", axis=1).sort_values(
    LV_NAME, ascending=False
).head(60)

# %% [markdown]
# Now you can go back to section `Standardize cell type names` in this notebook and adjust according to this findings to improve the figure.
# Once you are happy, you can remove the `NOT CATEGORIZED` items by uncommenting the code in `Delete samples with no tissue/cell type information`.

# %%
