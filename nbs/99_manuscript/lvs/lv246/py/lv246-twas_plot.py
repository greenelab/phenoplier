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
# COMPLETE
# Generates a plot with TWAS associations using the top genes and traits for LV246 (related to the hypertension cluster).

# %% [markdown]
# # Modules loading

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import conf
from data.cache import read_data
from utils import generate_result_set_name
from data.recount2 import LVAnalysis

# %% [markdown]
# # Settings

# %%
LV_NUMBER_SELECTED = 246
LV_NAME_SELECTED = f"LV{LV_NUMBER_SELECTED}"
display(LV_NAME_SELECTED)

# %%
OUTPUT_FIGURES_DIR = Path(
    conf.MANUSCRIPT["FIGURES_DIR"], "lvs_analysis", f"lv{LV_NUMBER_SELECTED}"
).resolve()
display(OUTPUT_FIGURES_DIR)
OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# # Data loading

# %% [markdown]
# ## MultiPLIER summary

# %%
multiplier_model_summary = read_data(conf.MULTIPLIER["MODEL_SUMMARY_FILE"])

# %%
multiplier_model_summary.shape

# %%
multiplier_model_summary.head()

# %% [markdown]
# ## PhenomeXcan results

# %% [markdown]
# ### S-MultiXcan

# %%
smultixcan_results_filename = conf.PHENOMEXCAN[
    "SMULTIXCAN_EFO_PARTIAL_MASHR_PVALUES_FILE"
]
display(smultixcan_results_filename)

# %%
smultixcan_results = pd.read_pickle(smultixcan_results_filename)

# %%
smultixcan_results.shape

# %%
smultixcan_results.head()

# %% [markdown]
# ### fastENLOC

# %%
fastenloc_results_filename = conf.PHENOMEXCAN["FASTENLOC_EFO_PARTIAL_TORUS_RCP_FILE"]
display(fastenloc_results_filename)

# %%
fastenloc_results = pd.read_pickle(fastenloc_results_filename)

# %%
fastenloc_results.shape

# %%
fastenloc_results.head()

# %% [markdown]
# ## S-MultiXcan projection (`z_score_std`)

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

# %%
lv_traits = lv_obj.lv_traits.copy()

# %% [markdown]
# ## Genes

# %%
lv_obj.lv_genes.shape

# %%
lv_obj.lv_genes.head(10)

# %%
lv_genes = (
    lv_obj.lv_genes[["gene_name", LV_NAME_SELECTED]].set_index("gene_name").squeeze()
)

# %%
lv_genes

# %%
" ".join(lv_genes.index[:20])

# %% [markdown]
# # Prepare data for plot

# %%
from entity import Gene, Trait

# %% [markdown]
# ## Select top genes

# %%
lv_top_genes = lv_genes.head(40).rename(index=Gene.GENE_NAME_TO_ID_MAP)

# %%
lv_top_genes

# %%
# remove genes not present in S-MultiXcan
lv_top_genes = lv_top_genes.loc[
    [g for g in lv_top_genes.index if g in Gene.GENE_ID_TO_NAME_MAP]
]
# assert lv_top_genes.shape[0] == 30, lv_top_genes.shape

# %% [markdown]
# ## Select top traits

# %%
lv_top_traits = lv_traits.head(20)  # df["trait"]

# %%
lv_top_traits.shape

# %%
lv_top_traits.head()

# %%
# Remove some
lv_top_traits = lv_top_traits.drop(
    [
        #         "1737-Childhood_sunburn_occasions",
        "20095_3-Size_of_white_wine_glass_drunk_large_250ml",
        #         "1747_2-Hair_colour_natural_before_greying_Red",
        "TRAUMBRAIN_NONCONCUS-severe_traumatic_brain_injury_does_not_include_concussion",
        #         "3143_raw-Ankle_spacing_width",
        "4119_raw-Ankle_spacing_width_right",
        "4100_raw-Ankle_spacing_width_left",
        "5109_raw-6mm_asymmetry_angle_right",
        "20107_100-Illnesses_of_father_None_of_the_above_group_1",
    ]
)

# %% [markdown]
# ## Subset S-MultiXcan results

# %%
data_subset = smultixcan_results.loc[lv_top_genes.index, lv_top_traits.index].T

# %%
data_subset = data_subset.apply(lambda x: -np.log10(x))

# %%
data_subset.shape

# %%
data_subset = data_subset.rename(columns=Gene.GENE_ID_TO_NAME_MAP)

# %%
data_subset

# %%
data_subset["trait_description"] = data_subset.apply(
    lambda x: Trait.get_trait(full_code=x.name).description
    if not Trait.is_efo_label(x.name)
    else x.name,
    axis=1,
)

# %%
# calculate mean of traits with the same description
data_subset = data_subset.groupby("trait_description").mean()

# %%
data_subset.shape

# %%
data_subset.head()

# %%
plot_data = data_subset.reset_index().set_index("trait_description")

# %%
plot_data

# %%
_old_columns = plot_data.index
display(_old_columns.shape)
display(_old_columns)

# %%
# reorder columns
plot_data = plot_data.loc[
    [
        "Ankle spacing width",
        "Basophill percentage",
        "Mean platelet (thrombocyte) volume",
        "Mean reticulocyte volume",
        "Red blood cell (erythrocyte) distribution width",
        "Ease of skin tanning",
        "Childhood sunburn occasions",
        "Hair colour (natural, before greying): Red",
        "Pulse rate, automated reading",
        "CH2DB NMR",
        "HDL Cholesterol NMR",
        "hypercholesterolemia",
        "Medication for cholesterol, blood pressure, diabetes, or take exogenous hormones: Cholesterol lowering medication",
        "Illnesses of father: Heart disease",
    ]
]

# %%
assert set(plot_data.index) == set(_old_columns)

# %% [markdown]
# ## Subset fastENLOC results

# %%
data_subset2 = (
    fastenloc_results.loc[lv_top_genes.index, lv_top_traits.index].fillna(0.0).T * 100.0
)

# %%
data_subset2.shape

# %%
data_subset2 = data_subset2.rename(columns=Gene.GENE_ID_TO_NAME_MAP)

# %%
data_subset2["trait_description"] = data_subset2.apply(
    lambda x: Trait.get_trait(full_code=x.name).description
    if not Trait.is_efo_label(x.name)
    else x.name,
    axis=1,
)

# %%
# calculate mean of traits with the same description
data_subset2 = data_subset2.groupby("trait_description").mean()
# _tmp = _tmp.assign(cluster_name=_tmp.apply(lambda x: _cluster_column[x.name], axis=1))

# %%
data_subset2.shape

# %%
data_subset2 = data_subset2.loc[plot_data.index, plot_data.columns]

# %%
data_subset2.shape

# %%
data_subset2.head()

# %%
plot_data2 = data_subset2  # .reset_index().set_index("trait_description")

# %%
plot_data2.shape

# %%
pd.Series(plot_data2.values.flatten()).quantile(
    [0.01, 0.05, 0.10, 0.15, 0.90, 0.95, 0.97, 0.98, 0.99]
)

# %%
pd.Series(plot_data2.values.flatten()).describe()

# %%
fastenloc_heatmap_data = plot_data2.T  # .drop(columns=["cluster_name", "color"]).T

# %%
fastenloc_heatmap_data.shape

# %%
fastenloc_heatmap_data[fastenloc_heatmap_data > 100.0] = 100.0

# %%
fastenloc_heatmap_data.head()

# %%
fastenloc_heatmap_data = (
    fastenloc_heatmap_data.reset_index(drop=True).T.reset_index(drop=True).T
)

# %%
fastenloc_heatmap_data = (
    fastenloc_heatmap_data.rename_axis(index="genes", columns="traits")
    .unstack()
    .rename("rcp")
    .reset_index()
)

# %%
fastenloc_heatmap_data.head()

# %%
_tmp_heatmap_multixcan2 = plot_data.loc[plot_data2.index, plot_data2.columns].T

# %%
fastenloc_heatmap_data = fastenloc_heatmap_data.assign(
    color=fastenloc_heatmap_data.apply(
        lambda x: "white"
        if _tmp_heatmap_multixcan2.iloc[int(x.genes), int(x.traits)] >= 8
        else "black",
        axis=1,
    )
)

# %%
fastenloc_heatmap_data = fastenloc_heatmap_data.assign(
    marker_size=fastenloc_heatmap_data.apply(
        lambda x: x.rcp if x.rcp >= 1.0 else 0.0, axis=1
    )
)

# %%
n_power = 1.30

fastenloc_heatmap_data["marker_size"] = np.power(
    fastenloc_heatmap_data["marker_size"], n_power
)

# %%
fastenloc_heatmap_data["traits"] += 0.5
fastenloc_heatmap_data["genes"] += 0.5

# %%
fastenloc_heatmap_data  # .groupby('color').count()

# %% [markdown]
# # Plot

# %%
_trait_renames = {
    #     "platelet count": "Platelet count",
    #     "Red blood cell (erythrocyte) count": "Erythrocyte count",
    #     "erythrocyte count": "Erythrocyte count",
    "Red blood cell (erythrocyte) distribution width": "Erythrocyte distribution width",
    "Hair colour (natural, before greying): Red": "Hair color: red",
    #     "reticulocyte count": "Reticulocyte count",
    #     "monocyte count": "Monocyte count",
    #     "lymphocyte count": "Lymphocyte count",
    #     "eosinophil count": "Eosinophill count",
    #     "neutrophil count": "Neutrophil count",
    #     "Neutrophill count": "Neutrophil count",
    "Illnesses of father: Heart disease": "Heart disease (father)",
    "HDL Cholesterol NMR": "HDL Cholesterol",
    "CH2DB NMR": "CH2DB (lipids)",
    #     "White blood cell (leukocyte) count": "Leukocyte count",
    #     "leukocyte count": "Leukocyte count",
    #     "granulocyte count": "Granulocyte count",
    #     "myeloid white cell count": "Myeloid count",
    #     "sum of basophil and neutrophil counts": "Basophil+neutrophil counts",
    #     "sum of neutrophil and eosinophil counts": "Neutrophil+eosinophil counts",
    #     "sum of eosinophil and basophil counts": "Eosinophil+basophil counts",
    "Mean platelet (thrombocyte) volume": "Mean platelet volume",
    "Medication for cholesterol, blood pressure, diabetes, or take exogenous hormones: Cholesterol lowering medication": "Cholesterol medication",
    "hypercholesterolemia": "High-cholesterol",
    "Pulse rate, automated reading": "Pulse rate",
}

# %%
lipids_traits = [
    "Pulse rate",
    "CH2DB (lipids)",
    "HDL Cholesterol",
    "High-cholesterol",
    "Cholesterol medication",
    "Heart disease (father)",
]

# %%
plot_data = _tmp_heatmap_multixcan2.rename(columns=_trait_renames)

# %%
rc = {
    #     "font.size": 9,
    #     "xtick.labelsize": 10,
    #     "ytick.labelsize": 19,
}

with sns.plotting_context("paper", font_scale=2.50, rc=rc):
    g = sns.clustermap(
        data=plot_data,
        vmin=0.0,
        vmax=10.0,
        row_cluster=False,
        col_cluster=False,
        dendrogram_ratio=0.20,
        xticklabels=True,
        yticklabels=True,
        figsize=(10, 19),
        linewidths=0.25,
        cmap="Blues",
        cbar_pos=(0.95, 0.64, 0.05, 0.10),
    )

    g.ax_heatmap.set_xlabel(None)
    g.ax_heatmap.set_ylabel(None)

    g.ax_heatmap.get_xaxis().set_ticklabels(
        g.ax_heatmap.get_xaxis().get_ticklabels(),
        rotation=45,
        horizontalalignment="right",
    )

    for l in g.ax_heatmap.get_yaxis().get_ticklabels():
        l.set_style("italic")
        if l.get_text() in ("DGAT2", "ACACA"):
            l.set_fontweight("bold")

    for l in g.ax_heatmap.get_xaxis().get_ticklabels():
        if l.get_text() in lipids_traits:
            l.set_fontweight("bold")

    g.ax_heatmap.scatter(
        fastenloc_heatmap_data["traits"].tolist(),
        fastenloc_heatmap_data["genes"].tolist(),
        marker=".",
        s=fastenloc_heatmap_data["marker_size"].tolist(),
        color=fastenloc_heatmap_data["color"].tolist(),
    )

    for rcp in [10, 25, 50, 75, 100]:
        plt.scatter(
            [],
            [],
            marker=".",
            c="k",
            alpha=0.3,
            s=np.power(rcp, n_power),
            label=str(rcp) + "%",
        )
    leg = plt.legend(
        scatterpoints=1,
        frameon=False,
        labelspacing=1,
        loc="center right",
        bbox_to_anchor=(1.90, -2.0, 0.5, 0.5),
        title="RCP.",
        title_fontsize=18,  # rc['ytick.labelsize'],
        fontsize=14,  # rc['ytick.labelsize'],
    )
    plt.setp(leg.get_title(), multialignment="center")

    g.ax_cbar.set_title("MultiXcan\n-log($p$)", fontdict={"fontsize": 18})
    g.ax_cbar.tick_params(labelsize=14)

    output_filepath = OUTPUT_FIGURES_DIR / f"lv{LV_NUMBER_SELECTED}-twas_plot.svg"
    display(output_filepath)
    plt.savefig(
        output_filepath,
        #         dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )

# %% [markdown]
# # Show p-values for some genes

# %%
_tmp = plot_data.loc["DGAT2", lipids_traits].sort_values()
_tmp = np.power(10, -_tmp)
display(_tmp)

# %%
_tmp = plot_data.loc["ACACA", lipids_traits].sort_values()
_tmp = np.power(10, -_tmp)
display(_tmp)

# %%
