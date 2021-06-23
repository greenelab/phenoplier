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
# This notebook contains the interpretation of a cluster (which features/latent variables in the original data are useful to distinguish traits in the cluster).
#
# See section [LV analysis](#lv_analysis) below

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
# PARTITION_K = 26
# PARTITION_CLUSTER_ID = 18

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

# %% [markdown]
# # Load MultiPLIER summary

# %%
multiplier_model_summary = read_data(conf.MULTIPLIER["MODEL_SUMMARY_FILE"])

# %%
multiplier_model_summary.shape

# %%
multiplier_model_summary.head()

# %% [markdown]
# # Load data

# %% [markdown]
# ## Original data

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
data = pd.read_pickle(input_filepath)

# %% tags=[]
data.shape

# %% tags=[]
data.head()

# %% [markdown]
# ## Clustering partitions

# %%
# CONSENSUS_CLUSTERING_DIR = Path(
#     conf.RESULTS["CLUSTERING_DIR"], "consensus_clustering"
# ).resolve()

# display(CONSENSUS_CLUSTERING_DIR)

# %%
# input_file = Path(CONSENSUS_CLUSTERING_DIR, "best_partitions_by_k.pkl").resolve()
# display(input_file)

# %%
# best_partitions = pd.read_pickle(input_file)

# %%
# best_partitions.shape

# %%
# best_partitions.head()

# %% [markdown]
# # Functions

# %%
# def show_cluster_stats(data, partition, cluster):
#     cluster_traits = data[partition == cluster].index
#     display(f"Cluster '{cluster}' has {len(cluster_traits)} traits")
#     display(cluster_traits)

# %% [markdown]
# # LV analysis
# <a id="lv_analysis"></a>

# %% [markdown]
# ## Associated traits

# %%
# display(best_partitions.loc[PARTITION_K])
# part = best_partitions.loc[PARTITION_K, "partition"]

# %%
# show_cluster_stats(data, part, PARTITION_CLUSTER_ID)

# %% [markdown]
# ## Associated latent variables

# %%
# ci = ClusterInterpreter(
#     threshold=1.0,
#     max_features=20,
#     max_features_to_explore=100,
# )

# %%
# ci.fit(data, part, PARTITION_CLUSTER_ID)

# %%
# ci.features_

# %%
# # save interpreter instance
# output_dir = Path(
#     conf.RESULTS["CLUSTERING_INTERPRETATION"]["BASE_DIR"],
#     "cluster_lvs",
#     f"part{PARTITION_K}",
# )
# output_dir.mkdir(exist_ok=True, parents=True)

# %%
# output_file = Path(
#     output_dir, f"cluster_interpreter-part{PARTITION_K}_k{PARTITION_CLUSTER_ID}.pkl"
# )
# display(output_file)

# %%
# ci.features_.to_pickle(output_file)

# %% [markdown]
# ## Top attributes

# %%
output_file = OUTPUT_DIR / "cardiovascular-niacin.h5"
display(output_file)

# %%
with pd.HDFStore(output_file, mode="r") as store:
    traits_module_tissue_data = store["traits_module_tissue_data"]
    drug_data = store["drug_data"]
    drug_trait_predictions = store["drug_trait_predictions"]

# %%
top_lvs = drug_trait_predictions["CARDIoGRAM_C4D_CAD_ADDITIVE"].sort_values(
    ascending=True
)
top_lvs = top_lvs[top_lvs < 0.0]

# %%
top_lvs.shape

# %%
top_lvs

# %%
lvs_list = top_lvs.index.tolist()

# %%
len(lvs_list)

# %%
lvs_list[:10]


# %%
def _my_func(x):
    _cols = [c for c in x.index if not c.startswith("LV")]
    _tmp = x[_cols].dropna()
    if _tmp.shape[0] > 0:
        return _tmp.iloc[0]

    return None


# %%
cell_type_dfs = []
tissue_dfs = []

pbar = tqdm(lvs_list[:50])
for lv_name in pbar:
    pbar.set_description(lv_name)

    #     lv_name = lv_info["name"]
    lv_obj = LVAnalysis(lv_name, data)

    #     # show lv prior knowledge match (pathways)
    #     lv_pathways = multiplier_model_summary[
    #         multiplier_model_summary["LV index"].isin((lv_name[2:],))
    #         & (
    #             (multiplier_model_summary["FDR"] < 0.05)
    #             | (multiplier_model_summary["AUC"] >= 0.75)
    #         )
    #     ]
    #     display(lv_pathways)

    lv_data = lv_obj.get_experiments_data()

    #     display("")
    #     display(lv_obj.lv_traits.head(20))
    #     display("")
    #     display(lv_obj.lv_genes.head(10))

    # get cell type attributes
    #     lv_attrs = lv_obj.get_attributes_variation_score()
    lv_attrs = pd.Series(lv_data.columns.tolist())
    lv_attrs = lv_attrs[
        lv_attrs.str.match(
            "(?:cell[^\w]*type$)",
            case=False,
            flags=re.IGNORECASE,
        ).values
    ].sort_values(ascending=False)
    display(lv_attrs)

    lv_attrs_data = lv_data[lv_attrs.tolist() + [lv_name]]
    lv_attrs_data = lv_attrs_data.assign(attr=lv_attrs_data.apply(_my_func, axis=1))
    lv_attrs_data = lv_attrs_data.drop(columns=lv_attrs.tolist())
    lv_attrs_data = lv_attrs_data.dropna().sort_values(lv_name, ascending=False)
    lv_attrs_data = lv_attrs_data.rename(columns={lv_name: "lv"})
    lv_attrs_data = lv_attrs_data.assign(lv_name=lv_name)
    cell_type_dfs.append(lv_attrs_data)

    # get tissue attributes
    lv_attrs = pd.Series(lv_data.columns.tolist())
    lv_attrs = lv_attrs[
        lv_attrs.str.match(
            "(?:tissue$)|(?:tissue[^\w]*type$)",
            case=False,
            flags=re.IGNORECASE,
        ).values
    ].sort_values(ascending=False)
    display(lv_attrs)

    lv_attrs_data = lv_data[lv_attrs.tolist() + [lv_name]]
    lv_attrs_data = lv_attrs_data.assign(attr=lv_attrs_data.apply(_my_func, axis=1))
    lv_attrs_data = lv_attrs_data.drop(columns=lv_attrs.tolist())
    lv_attrs_data = lv_attrs_data.dropna().sort_values(lv_name, ascending=False)
    lv_attrs_data = lv_attrs_data.rename(columns={lv_name: "lv"})
    lv_attrs_data = lv_attrs_data.assign(lv_name=lv_name)
    tissue_dfs.append(lv_attrs_data)

# %% [markdown]
# # LVs selection

# %%
N_TOP_LVS = 10


# %% [markdown]
# # Cell types

# %%
def _get_lv_rank(data):
    data = data.copy()
    data["lv"] = data["lv"].rank()
    return data


# %%
df = pd.concat(cell_type_dfs[:N_TOP_LVS], ignore_index=True)
# df = pd.concat([_get_lv_rank(x) for x in cell_type_dfs[:N_TOP_LVS]], ignore_index=True)

# %%
# https://trace.ncbi.nlm.nih.gov/Traces/sra/?study=SRP037775
df = df[~((df["attr"] == "HER2-positive breast cancer") & (df["lv"] < 0.10))]

# The PBMCs entry is related only to those samples treated with HSV-1
# see https://trace.ncbi.nlm.nih.gov/Traces/sra/?study=SRP045569
# here I keep only those, since the rest (sham) have almost zero expression
df = df[~((df["attr"] == "PBMCs") & (df["lv"] < 0.05))]

# https://trace.ncbi.nlm.nih.gov/Traces/sra/?study=SRP062958
# df = df[~((df["attr"] == "peripheral blood monocytes") & (df["lv"] < 0.00))]

# https://trace.ncbi.nlm.nih.gov/Traces/sra/?study=SRP048804
# df = df[~((df["attr"] == "glioblastoma cell line") & (df["lv"] < 0.00))]

# https://trace.ncbi.nlm.nih.gov/Traces/sra/?study=SRP066356
# df = df[~((df["attr"] == "Monocyte") & (df["lv"] < 0.00))]

# https://trace.ncbi.nlm.nih.gov/Traces/sra/?study=SRP015670
# df = df[~((df["attr"] == "monocyte-derived macrophages") & (df["lv"] < 0.00))]

# %%
# df = df[~df["attr"].str.lower().str.contains("cancer|glioblastoma|carcinoma")]

# %%
df.groupby("attr").median().squeeze().sort_values(ascending=False).head(25)

# %%
df = df.replace(
    {
        "attr": {
            "HER2-positive breast cancer": "HER2-positive breast cancer\n(no trastuzumab resistance)",
            #             "": "",
            "Rapamycin treated fibroblast": "Fibroblasts\n(Rapamycin treated)",
            "": "",
            #             "CD14 cells": "CD14+ cells",
            "M1-polarized HMDM": "M1 macrophages\n(HMDM)",
            "M1-polarized IPSDM": "M1 macrophages\n(IPSDM)",
            "PBMCs": "PBMCs (HSV)",
            #             "peripheral blood monocytes": "Monocytes (IFNa)",
            "LHSAR overexpressed with HOXB13": "Prostate epithelial cells\n(LHSAR overexpressed\nwith HOXB13)",
            "LHSAR overexpressed with HOXB13 and FOXA1": "Prostate epithelial cells (LHSAR)",
            "LHSAR overexpressed with LacZ": "Prostate epithelial cells (LHSAR)",
            #             "Primary Monocytes(BC8)": "Primary monocytes",
            #             "Primary Monocytes(BC9)": "Primary monocytes",
            #             "Primary Monocytes(BC12)": "Primary monocytes",
            #             "Primary Monocytes(BC11)": "Primary monocytes",
            #             "glioblastoma cell line": "Glioblastoma\n(GBM1A cell line)",
            #             "Monocyte": "Monocytes",
            #             "monocyte-derived macrophages": "Monocyte-derived\nmacrophages (WNV)",
            #             "Tongue squamous cell carcinoma": "Tongue squamous\ncell carcinoma",
        }
    }
)

# %%
cat_order

# %%
cat_order = df[["attr", "lv"]].groupby("attr").median().squeeze()
cat_order = cat_order.sort_values(ascending=False)
cat_order = cat_order.head(10)
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

    output_filepath = OUTPUT_FIGURES_DIR / "niacin-cad-modules_cell_types.svg"
    display(output_filepath)
    plt.savefig(
        output_filepath,
        bbox_inches="tight",
    )

# %%
df_tmp = pd.concat(cell_type_dfs[:N_TOP_LVS], ignore_index=False).reset_index()

# %%
df_tmp[df_tmp["attr"].str.contains("Neutrophils")]

# %%
lv_name = "LV116"
lv_obj = LVAnalysis(lv_name, data)

# %%
lv_obj.lv_genes.head(10)

# %%
lv_data = lv_obj.get_experiments_data()

# %%
lv_data.shape

# %%
lv_data.loc[["SRP045500"]].dropna(how="all", axis=1).sort_values(
    lv_name, ascending=False
).head(60)

# %%

# %% [markdown]
# # Tissues

# %%
df = pd.concat(tissue_dfs[:N_TOP_LVS], ignore_index=True)
# df = pd.concat([_get_lv_rank(x) for x in tissue_dfs[:N_TOP_LVS]], ignore_index=True)

# %%
df.groupby("attr").mean().squeeze().sort_values(ascending=False).head(50)

# %% [markdown]
# # LV analysis

# %%
lv_obj = LVAnalysis("LV116", data)

# %%
lv_obj.lv_genes.head(20)

# %%
lv_data = lv_obj.get_experiments_data()

# %%
lv_data.shape

# %%
_tmp = lv_data[["cell type", "LV116"]].dropna()

# %%
_tmp[_tmp["cell type"].str.contains("M1")]

# %%
_tmp[_tmp["cell type"].str.contains("CD14")]

# %%

# %%

# %%
lv_obj2 = LVAnalysis("LV881", data)

# %%
lv_obj2.lv_genes.head(20)

# %%
lv_attrs = lv_obj.get_attributes_variation_score()
_tmp = pd.Series(lv_attrs.index)
lv_attrs = lv_attrs[
    _tmp.str.match(
        "(?:cell.+type$)|(?:tissue$)|(?:tissue.+type$)",
        case=False,
        flags=re.IGNORECASE,
    ).values
].sort_values(ascending=False)
display(lv_attrs)

# %%
with sns.plotting_context("paper", font_scale=1.0), sns.axes_style("whitegrid"):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax = lv_obj2.plot_attribute("tissue", top_x_values=20)

# %%
with sns.plotting_context("paper", font_scale=1.0), sns.axes_style("whitegrid"):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax = lv_obj2.plot_attribute("cell type", top_x_values=20)

# %%
