# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
#     formats: ipynb,py//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Description

# %% [markdown]
# Generates a plot with the top conditions/experiments for LV603.

# %% [markdown]
# # Modules loading

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from IPython.display import display
from pathlib import Path
import re

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import conf
from data.cache import read_data
from data.recount2 import LVAnalysis

# %%
assert (
    conf.MANUSCRIPT["BASE_DIR"] is not None
), "The manuscript directory was not configured"

display(conf.MANUSCRIPT["BASE_DIR"])

# %% [markdown]
# # Settings

# %%
LV_NUMBER_SELECTED = 603
LV_NAME_SELECTED = f"LV{LV_NUMBER_SELECTED}"
display(LV_NAME_SELECTED)

# %%
OUTPUT_FIGURES_DIR = Path(conf.MANUSCRIPT["FIGURES_DIR"], "entire_process").resolve()
display(OUTPUT_FIGURES_DIR)
OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# # Load MultiPLIER summary

# %%
multiplier_model_summary = read_data(conf.MULTIPLIER["MODEL_SUMMARY_FILE"])

# %%
multiplier_model_summary.shape

# %%
multiplier_model_summary.head()

# %% [markdown]
# # Load S-MultiXcan projection (`z_score_std`)

# %% tags=[]
INPUT_SUBSET = "z_score_std"

# %% tags=[]
INPUT_STEM = "projection-smultixcan-efo_partial-mashr-zscores"

# %% tags=[]
input_filepath = Path(
    conf.RESULTS["DATA_TRANSFORMATIONS_DIR"],
    INPUT_SUBSET,
    f"{INPUT_SUBSET}-{INPUT_STEM}.pkl",
).resolve()
display(input_filepath)

assert input_filepath.exists(), "Input file does not exist"

input_filepath_stem = input_filepath.stem
display(input_filepath_stem)

# %% tags=[]
data = read_data(input_filepath)

# %% tags=[]
data.shape

# %% tags=[]
data.head()

# %% [markdown]
# # LV analysis

# %%
lv_obj = lv_exp = LVAnalysis(LV_NAME_SELECTED, data)

# %%
lv_gene_sets = multiplier_model_summary[
    multiplier_model_summary["LV index"].isin((str(LV_NUMBER_SELECTED),))
    & (
        (multiplier_model_summary["FDR"] < 0.05)
        | (multiplier_model_summary["AUC"] >= 0.75)
    )
]
display(lv_gene_sets)

# %% [markdown]
# ## Traits

# %%
lv_obj.lv_traits.shape

# %%
lv_obj.lv_traits.head(20)

# %% [markdown]
# ## Genes

# %%
lv_obj.lv_genes.shape

# %%
lv_obj.lv_genes.head(10)

# %% [markdown]
# ## Conditions

# %%
lv_obj.lv_conds.shape

# %%
lv_obj.lv_conds.head()

# %% [markdown]
# # Cell type and tissue attributes

# %%
lv_attrs = lv_obj.get_attributes_variation_score()

_tmp = pd.Series(lv_attrs.index)

lv_attrs = lv_attrs[
    _tmp.str.match(
        "(?:cell[\W]*type$)|(?:tissue$)|(?:tissue[\W]*type$)",
        case=False,
        flags=re.IGNORECASE,
    ).values
].sort_values(ascending=False)

display(lv_attrs)

# %% [markdown]
# ## LV data

# %%
lv_data = lv_obj.get_experiments_data()

# %%
lv_data.shape

# %%
lv_data.head()

# %% [markdown]
# ## Prepare experiments data

# %%
lv_data = lv_data[lv_attrs.index.tolist() + [LV_NAME_SELECTED]]

# %%
# merge "cell type" with "celltype"
lv_data = lv_data.fillna(
    {
        "cell type": lv_data["celltype"],
    }
)

# %%
imp_f = "cell type"
assert imp_f in lv_attrs.index

# %%
features = [imp_f, LV_NAME_SELECTED]

# %%
lv_data.shape

# %%
lv_data = lv_data[features].dropna()

# %%
imp_f_old = imp_f
imp_f = "Cell type"
imp_f_short = "cell_type"
lv_data = lv_data.rename(columns={imp_f_old: imp_f})

# %%
lv_data.shape

# %%
lv_data.head()

# %%
lv_data.index.get_level_values("project").unique()

# %% [markdown]
# # Figure: conditions/experiments summary

# %% [markdown]
# ## Settings

# %%
top_x_values = 7

# %% [markdown]
# ## Use uniform cell types

# %%
# lv_data[lv_data[imp_f].str.lower().str.contains("mye")]  # ["Cell type"].iloc[0]

# %%
plot_data = lv_data.replace(
    {
        imp_f: {
            # neutrophils
            "primary human neutrophils": "Neutrophils",
            "Neutrophil isolated from peripheral blood": "Neutrophils",
            "Neutrophil": "Neutrophils",
            "neutrophils (Neu)": "Neutrophils",
            # granulocytes
            "granulocyte": "Granulocytes",
            # monocytes
            "primary human monocytes": "Monocytes",
            # whole blood
            "Whole Blood": "Whole blood",
            # PBMC
            "primary human PBMC": "PBMC",
            # B-cells
            "primary human B cells": "B cells",
            # T-cells
            "primary human T cells": "T cells",
            #             "naive CD4+ T-cells": "Naive CD4+ T-cells",
            # epithelial cells
            "epithelial cells (Epi)": "Epithelial cells",
            "primary human myeloid DC": "mDCs",
        }
    }
)

# %% [markdown]
# ## Plot

# %%
with sns.plotting_context("paper", font_scale=5.0):
    cat_order = plot_data.groupby(imp_f).median().squeeze()

    cat_order = cat_order.sort_values(ascending=False)
    if top_x_values is not None:
        cat_order = cat_order.head(top_x_values)

    cat_order = cat_order.index

    g = sns.catplot(
        data=plot_data,
        x=imp_f,
        y=LV_NAME_SELECTED,
        order=cat_order,
        kind="box",
        linewidth=1,
        height=10,
        aspect=1.5,
    )
    g.ax.set_ylabel(f"$\mathbf{{B}}_{{\mathrm{{LV}}{LV_NUMBER_SELECTED}}}$")
    g.set_xticklabels(g.ax.get_xticklabels(), rotation=45, horizontalalignment="right")
    # g.set_yticklabels([])
    g.ax.set_xlabel("")

    output_filepath = OUTPUT_FIGURES_DIR / f"lv{LV_NUMBER_SELECTED}_{imp_f_short}.pdf"
    display(output_filepath)
    plt.savefig(
        output_filepath,
        bbox_inches="tight",
    )

# %% [markdown]
# ## Data stats

# %%
imp_f

# %%
# Get top cell types sorted by median value, then get those cell type names
tmp = plot_data.groupby(imp_f).median().sort_values(by=LV_NAME_SELECTED, ascending=False)
display(tmp)

cell_types = tmp.index
display(cell_types)

# %%
# Get the sample size by cell type; this is in response to editorial comments about Figure 1c
plot_data.groupby(imp_f).count().loc[cell_types]

# %%
