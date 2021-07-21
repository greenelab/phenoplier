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
import pickle
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# from IPython.display import HTML
from tqdm import tqdm

from clustering.methods import ClusterInterpreter
from data.recount2 import LVAnalysis
from data.cache import read_data
import conf

# %% [markdown]
# # Settings

# %% tags=["parameters"]
SHORT_TRAIT_NAME = "ICD10_I70_Atherosclerosis"
FULL_TRAIT_NAME = "I70-Diagnoses_main_ICD10_I70_Atherosclerosis"
N_TOP_LVS = 50

# %% [markdown]
# # Paths

# %% tags=[]
OUTPUT_DIR = conf.RESULTS["DRUG_DISEASE_ANALYSES"] / "lincs" / "analyses"
display(OUTPUT_DIR)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# %% tags=[]
OUTPUT_FIGURES_DIR = Path(
    conf.MANUSCRIPT["FIGURES_DIR"], "drug_disease_prediction"
).resolve()
display(OUTPUT_FIGURES_DIR)
OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# %%
OUTPUT_FILE = OUTPUT_DIR / "cardiovascular-niacin.h5"
display(OUTPUT_FILE)

# %% [markdown]
# # Load data

# %%
output_file = OUTPUT_DIR / "cardiovascular-niacin.h5"
display(output_file)

# %%
with pd.HDFStore(output_file, mode="r", complevel=4) as store:
    top_lvs = store[f"traits/{SHORT_TRAIT_NAME}/top_lvs"]
    cell_types_data = store[f"traits/{SHORT_TRAIT_NAME}/cell_types"]
    tissues_data = store[f"traits/{SHORT_TRAIT_NAME}/tissues"]

# %%
top_lvs = top_lvs.head(N_TOP_LVS)

# %%
top_lvs.shape

# %%
top_lvs.head()

# %%
cell_types_data.shape

# %%
cell_types_data.head()

# %%
tissues_data.shape

# %%
tissues_data.head()

# %% [markdown]
# # LVs selection

# %%
# N_TOP_LVS = 50

# %% [markdown]
# # Cell types

# %%
# def _get_lv_rank(data):
#     data = data.copy()
#     data["lv"] = data["lv"].rank()
#     return data

# %%
df = cell_types_data.copy()
# df = pd.concat(cell_type_dfs[:N_TOP_LVS], ignore_index=True)
# df = pd.concat([_get_lv_rank(x) for x in cell_type_dfs[:N_TOP_LVS]], ignore_index=True)

# %%
df = df[df["lv_name"].isin(top_lvs.index)]

# %%
df.shape

# %%
df.head()

# %%
# this is an attempt to weight cell types by the top lvs from drug-disease

# _tmp = pd.merge(df, top_lvs.rename("lv_weight"), left_on="lv_name", right_index=True)
# _tmp = _tmp.assign(value=(_tmp["lv"] * _tmp["lv_weight"]))
# df = _tmp[["attr", "value", "lv"]]

# %%
# The PBMCs entry is related only to those samples treated with HSV-1
# see https://trace.ncbi.nlm.nih.gov/Traces/sra/?study=SRP045569
# here I keep only those, since the rest (sham) have almost zero expression
df = df[~((df["attr"] == "PBMCs") & (df["lv"] < 0.05))]

# https://trace.ncbi.nlm.nih.gov/Traces/sra/?study=SRP062958
df = df[~((df["attr"] == "peripheral blood monocytes") & (df["lv"] < 0.00))]

# https://trace.ncbi.nlm.nih.gov/Traces/sra/?study=SRP048804
df = df[~((df["attr"] == "glioblastoma cell line") & (df["lv"] < 0.00))]

# https://trace.ncbi.nlm.nih.gov/Traces/sra/?study=SRP066356
df = df[~((df["attr"] == "Monocyte") & (df["lv"] < 0.00))]

# https://trace.ncbi.nlm.nih.gov/Traces/sra/?study=SRP015670
df = df[~((df["attr"] == "monocyte-derived macrophages") & (df["lv"] < 0.00))]

# %%
df.groupby("attr").median().squeeze().sort_values(ascending=False).head(25)
# df[["attr", "value"]].groupby("attr").median().squeeze().sort_values(ascending=False).head(40)
# df.groupby("attr").max().sort_values("lv", ascending=False).head(25)

# %%
df = df.replace(
    {
        "attr": {
            "CD14 cells": "CD14+ cells",
            "M1-polarized HMDM": "M1 macrophages\n(HMDM)",
            "M1-polarized IPSDM": "M1 macrophages\n(IPSDM)",
            "PBMCs": "PBMCs (HSV)",
            "peripheral blood monocytes": "Monocytes (IFNa)",
            "LHSAR overexpressed with HOXB13": "Prostate epithelial cells (LHSAR)",
            "LHSAR overexpressed with HOXB13 and FOXA1": "Prostate epithelial cells (LHSAR)",
            "LHSAR overexpressed with LacZ": "Prostate epithelial cells (LHSAR)",
            "Primary Monocytes(BC8)": "Primary monocytes",
            "Primary Monocytes(BC9)": "Primary monocytes",
            "Primary Monocytes(BC12)": "Primary monocytes",
            "Primary Monocytes(BC11)": "Primary monocytes",
            "glioblastoma cell line": "Glioblastoma (GBM1A cell line)",
            "Monocyte": "Monocytes",
            "monocyte-derived macrophages": "Monocyte-derived\nmacrophages (WNV)",
            "Tongue squamous cell carcinoma": "Tongue squamous\ncell carcinoma",
            
            "CD4+CD25highCD127low/- Treg cells": "Regulatory T cells (Treg)",
            
            "WAT": "White adipose tissue",
            "BAT": "Brown adipose tissue",
            "human adipose-derived stem cells": "Adipose-derived stem cells",
            
            "neural precursor cell derived neuronal like cells": "Neural precursor cell",
            "Neural crest cells (hNCC) derived from H9 ESC": "Neural crest cells",
            "NPC": "Neural progenitor cell",
        }
    }
)

# %%
cat_order = df.groupby("attr").median().squeeze()
cat_order = cat_order.sort_values(ascending=False)
cat_order = cat_order.head(20)
cat_order = cat_order.index

# %%
with sns.plotting_context("paper", font_scale=2.5), sns.axes_style("whitegrid"):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax = sns.boxplot(
        data=df,
        x="attr",
        y="lv",
        order=cat_order,
        linewidth=None,
        ax=ax,
    )
    
    ax.set_xlabel("Cell type")
    ax.set_ylabel("LV value")
    plt.xticks(rotation=45, horizontalalignment="right")

    output_filepath = OUTPUT_FIGURES_DIR / f"niacin-{SHORT_TRAIT_NAME}-cell_types.svg"
    display(output_filepath)
    plt.savefig(
        output_filepath,
        bbox_inches="tight",
    )

# %% [markdown]
# # Debug

# %%
df[df["attr"].str.contains("NPC")]

# %%
LV_NAME = "LV879"

# %%
lv_obj = LVAnalysis(LV_NAME, None)

# %%
lv_data = lv_obj.get_experiments_data()

# %%
lv_data.shape

# %%
lv_data.head()

# %%
import numpy as np

# %%
mask = np.column_stack([lv_data[col].str.contains("^NPC$", na=False) for col in lv_data if col != LV_NAME])

# %%
lv_data.loc[mask.any(axis=1)]

# %%
# what is there in these projects?
lv_data.loc[["SRP017684"]].dropna(how="all", axis=1).sort_values(
    LV_NAME, ascending=False
).head(10)

# %%
_tmp = lv_data[["cell type", "LV116"]].dropna()

# %%
# _tmp[_tmp["cell type"].str.contains("M1")]

# %%
# _tmp[_tmp["cell type"].str.contains("CD14")]

# %%

# %% [markdown]
# # Tissues

# %%
# df = tissues_data.copy()

# %%
# df.shape

# %%
# df.head()

# %%
# df.groupby("attr").median().squeeze().sort_values(ascending=False).head(25)
# # df.groupby("attr").max().sort_values("lv", ascending=False).head(25)

# %% [markdown]
# # LV analysis

# %%
# lv_obj = LVAnalysis("LV116", data)

# %%
# lv_obj.lv_genes.head(20)

# %%
# lv_data = lv_obj.get_experiments_data()

# %%
# lv_data.shape

# %%
# _tmp = lv_data[["cell type", "LV116"]].dropna()

# %%
# _tmp[_tmp["cell type"].str.contains("M1")]

# %%
# _tmp[_tmp["cell type"].str.contains("CD14")]

# %%
