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
# This notebook generates a heatmap using a subset of the $\hat{\mathbf{M}}$ matrix (projectiong of S-MultiXcan results into latent gene expression representation). Traits in columns (organized by clusters of traits) and LVs in rows. The LVs shown are those detected to be driving clusters _and_ well-aligned (FDR < 0.05) with pathways (from the MultiPLIER models).

# %% [markdown]
# # Environment variables

# %%
from IPython.display import display

import conf

N_JOBS = min((conf.GENERAL["N_JOBS"], 3))
display(N_JOBS)

# %%
# %env MKL_NUM_THREADS=$N_JOBS
# %env OPEN_BLAS_NUM_THREADS=$N_JOBS
# %env NUMEXPR_NUM_THREADS=$N_JOBS
# %env OMP_NUM_THREADS=$N_JOBS

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
from entity import Trait
from data.cache import read_data

# %% [markdown]
# # Settings

# %%
OUTPUT_FIGURES_DIR = Path(conf.MANUSCRIPT["FIGURES_DIR"], "clustering").resolve()
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

# %%
well_aligned_lvs = multiplier_model_summary[
    (multiplier_model_summary["FDR"] < 0.05) | (multiplier_model_summary["AUC"] >= 0.75)
]

display(well_aligned_lvs.shape)
display(well_aligned_lvs.head())

# %% [markdown]
# ## S-MultiXcan results

# %%
smultixcan_results_filename = conf.PHENOMEXCAN[
    "SMULTIXCAN_EFO_PARTIAL_MASHR_ZSCORES_FILE"
]
display(smultixcan_results_filename)

# %%
smultixcan_results = pd.read_pickle(smultixcan_results_filename)

# %%
smultixcan_results.shape

# %%
smultixcan_results.head()

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
# ## Clustering results

# %%
CONSENSUS_CLUSTERING_DIR = Path(
    conf.RESULTS["CLUSTERING_DIR"], "consensus_clustering"
).resolve()

display(CONSENSUS_CLUSTERING_DIR)

# %%
input_file = Path(CONSENSUS_CLUSTERING_DIR, "best_partitions_by_k.pkl").resolve()
display(input_file)

# %%
best_partitions = pd.read_pickle(input_file)

# %%
# keep selected partitions only
best_partitions = best_partitions[best_partitions["selected"]]

# %%
best_partitions.shape

# %%
best_partitions.head()

# %% [markdown]
# # Get all clusters information

# %%
partition_k = 29

# %%
part = best_partitions.loc[partition_k, "partition"]

# %%
part.shape

# %%
part_cluster_size = pd.Series(part).value_counts()
display(part_cluster_size)

# %%
clusters_labels = {
    12: "Asthma",
    27: "Well-being",
    22: "Nutrients",
    15: "Heel bone",
    4: "Reticulocytes",
    2: "Red blood volumes",
    5: "Erythrocytes",
    23: "Read blood DW",
    20: "BMI",
    18: "Spirometry",
    19: "Height",
    10: "Keratometry",
    1: "Platelets",
    6: "Skin/hair color",
    13: "Autoimmune #13",
    26: "Autoimmune #26",
    8: "Autoimmune #8",
    17: "HTN/high cholesterol",
    25: "Other traits #25",
    21: "Eczema/IBD/SCZ",
    28: "Env. factors #28",
    11: "CAD/breast cancer",
    16: "Lipids/AD/Chronotype",
    14: "Other heart diseases",
    7: "Monocytes",
    24: "Lymphocytes",
    9: "Neutrophils",
    3: "Eosinophils",
}

# %%
all_clusters = []

for c_idx in part_cluster_size[part_cluster_size < 50].index:
    _df = pd.DataFrame(
        {
            "trait": data.index[part == c_idx].tolist(),
            "cluster_label": clusters_labels[c_idx],
            "cluster_idx": c_idx,
        }
    )

    all_clusters.append(_df)

# %%
# combine all
df = pd.concat(all_clusters, ignore_index=True)

# %%
df.shape

# %%
df.head()

# %% [markdown]
# # Get top LVs for each cluster

# %%
CLUSTER_LV_DIR = (
    conf.RESULTS["CLUSTERING_INTERPRETATION"]["BASE_DIR"]
    / "cluster_lvs"
    / f"part{partition_k}"
)
display(CLUSTER_LV_DIR)


# %%
def _get_lvs_data(cluster_idx):
    cluster_lvs = pd.read_pickle(
        CLUSTER_LV_DIR / f"cluster_interpreter-part{partition_k}_k{cluster_idx}.pkl"
    )

    # keep well-aligned lvs only
    cluster_lvs = cluster_lvs[
        cluster_lvs.index.astype(str).isin(well_aligned_lvs["LV index"])
    ]

    cluster_lvs["cluster"] = cluster_idx

    return pd.Series([set(cluster_lvs["name"]), cluster_lvs])


# %%
clusters_df = df.groupby("cluster_idx")[["cluster_label", "cluster_idx"]].head(1)

# add top lvs for each cluster
clusters_df[["lvs_names", "lvs_data"]] = clusters_df["cluster_idx"].apply(_get_lvs_data)

clusters_df = clusters_df.set_index("cluster_label")
assert clusters_df.index.is_unique

display(clusters_df)

# %%
lvs_data = (
    pd.concat(clusters_df["lvs_data"].tolist())
    .reset_index()
    .set_index(["cluster", "idx"])
)

# %%
lvs_data.shape

# %%
lvs_data.head(10)

# %%
_traits = df["trait"]
_lvs = set(lvs_data["name"])

# %%
_lvs

# %%
data_subset = data.loc[_traits, _lvs].rename_axis("trait")
data_subset = data_subset.assign(
    cluster_name=df.set_index("trait")["cluster_label"].astype("category")
)

# %%
data_subset.shape

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
data_subset

# %%
data_subset = data_subset.replace(
    {
        "trait_description": {
            "platelet count": "Platelet count",
            "Red blood cell (erythrocyte) count": "Erythrocyte count",
            "erythrocyte count": "Erythrocyte count",
            "Red blood cell (erythrocyte) distribution width": "Erythrocyte distribution width",
            "reticulocyte count": "Reticulocyte count",
        }
    }
)

# %%
_cluster_column = data_subset.set_index("trait_description")["cluster_name"].to_dict()

# %%
_cluster_column

# %%
data_subset

# %%
# calculate mean of traits with the same description
_tmp = data_subset.groupby("trait_description").mean()
_tmp = _tmp.assign(cluster_name=_tmp.apply(lambda x: _cluster_column[x.name], axis=1))

# %%
_tmp.shape

# %%
data_subset = _tmp.sort_values("cluster_name")

# %%
data_subset.shape

# %%
data_subset.head()

# %%
data_subset.describe()

# %%
data_subset["cluster_name"].unique()

# %%
# this is the order from the clustering tree
cluster_order = [
    "Asthma",
    #     "Well-being",
    #     "Nutrients",
    "Heel bone",
    "Reticulocytes",
    "Red blood volumes",
    "Erythrocytes",
    "Read blood DW",
    "BMI",
    "Spirometry",
    "Height",
    "Keratometry",
    "Platelets",
    "Skin/hair color",
    "Autoimmune #13",
    "Autoimmune #26",
    "Autoimmune #8",
    "HTN/high cholesterol",
    #     "Other traits #25",
    "Eczema/IBD/SCZ",
    #     "Other traits #28",
    "CAD/breast cancer",
    "Lipids/AD/Chronotype",
    "Other heart diseases",
    "Monocytes",
    "Lymphocytes",
    "Neutrophils",
    "Eosinophils",
]

# %%
# _cluster = data_subset["cluster_name"].unique()
lut = dict(zip(cluster_order, sns.color_palette("Paired", len(cluster_order))))

# %%
row_colors = data_subset[["cluster_name"]].assign(
    color=data_subset["cluster_name"].map(lut)
)

# %%
row_colors.head(20)

# %% [markdown]
# # Plot data

# %%
_tmp = data_subset.reset_index().set_index(["cluster_name", "trait_description"])

# %%
plot_data = _tmp.loc[cluster_order].droplevel("cluster_name")

# %%
plot_data = plot_data.assign(
    color=row_colors["color"], cluster_name=row_colors["cluster_name"]
)

# %%
plot_data.shape

# %%
plot_data.head()

# %%
_tmp = plot_data.groupby("cluster_name").count().iloc[:, 0]

# %%
_tmp = _tmp.loc[cluster_order]

# %%
_tmp

# %%
_tmp.cumsum()

# %%
cluster_order_start_idx = np.array([0.5] + _tmp.cumsum()[:-1].tolist())
# cluster_order = _tmp.index.tolist()

# %%
cluster_order_start_idx

# %%
# slightly change the "Blood DW" and "Reticulocytes" index since they overlap a bit in the figure
cluster_label_order_start_idx = cluster_order_start_idx.copy()
# cluster_label_order_start_idx[10] = cluster_label_order_start_idx[10] - 0.2
cluster_label_order_start_idx[4] = cluster_label_order_start_idx[4] - 1.0
cluster_label_order_start_idx[6] = cluster_label_order_start_idx[6] + 2.0

cluster_label_order_start_idx[13] = cluster_label_order_start_idx[13] - 1.0
cluster_label_order_start_idx[15] = cluster_label_order_start_idx[15] + 2.0

# %%
cluster_label_order_start_idx

# %%
cluster_order_start_idx.shape

# %%
len(cluster_order)

# %%
pd.Series(plot_data.drop(columns=["cluster_name", "color"]).values.flatten()).quantile(
    [0.90, 0.95, 0.97, 0.98, 0.99]
)

# %%
pd.Series(plot_data.drop(columns=["cluster_name", "color"]).values.flatten()).describe()


# %% [markdown]
# # Plotting functions

# %%
def _get_lv_pathway(lv_name):
    p = (
        well_aligned_lvs_df.loc[lv_name]
        .sort_values("FDR")
        .index.get_level_values("pathway")
    )
    return ", ".join(p)


def _get_lv_label(lv_name, n_top=2):
    lv_info = all_results[lv_name]

    cell_types_label = ""
    tissues_label = ""

    if "attr" in lv_info:
        cell_types_label = ", ".join(lv_info["attr"][:n_top])

    if "tissue" in lv_info:
        tissues_label = ", ".join(lv_info["tissue"][:n_top])

    final_label = ""

    if cell_types_label is not None:
        final_label += cell_types_label

    if tissues_label is not None:
        final_label += f" | {tissues_label}"

    return final_label


# %%
def plot_clustermap(rc, lv_labeling_function, output_filepath):
    with sns.plotting_context("paper", font_scale=1.25, rc=rc):
        g = sns.clustermap(
            data=plot_data.drop(columns=["cluster_name", "color"]).T,
            col_colors=plot_data["color"].rename("Cluster"),
            vmin=0.0,
            vmax=7.0,
            row_cluster=True,
            col_cluster=False,
            cmap="YlOrBr",
            dendrogram_ratio=0.05,
            cbar_pos=None,  # (-0.06, 0.8, 0.05, 0.18),
            xticklabels=False,
            yticklabels=True,
            figsize=(18, 13),
            method="complete",
            linewidths=0.25,
            rasterized=True,
        )

        g.ax_heatmap.set_xlabel("")
        g.ax_heatmap.set_ylabel("")

        g.ax_col_colors.set_xticks(cluster_label_order_start_idx)
        g.ax_col_colors.set_xticklabels(
            cluster_order,
            rotation=45,
            horizontalalignment="left",
        )
        g.ax_col_colors.xaxis.set_tick_params(size=0)
        g.ax_col_colors.xaxis.tick_top()

        g.ax_heatmap.get_xaxis().set_ticklabels(
            g.ax_heatmap.get_xaxis().get_ticklabels(),
            rotation=45,
            horizontalalignment="right",
        )

        if lv_labeling_function is not None:
            new_y_labels = []
            for t in g.ax_heatmap.get_yaxis().get_ticklabels():
                t.set_text(f"{t.get_text()}: {lv_labeling_function(t.get_text())}")
                t.set_fontsize(0.75 * t.get_fontsize())
                new_y_labels.append(t)

            g.ax_heatmap.get_yaxis().set_ticklabels(
                new_y_labels,
            )

        g.ax_heatmap.vlines(
            cluster_order_start_idx[1:], *g.ax_heatmap.get_ylim(), colors="black"
        )

        display(output_filepath)
        plt.savefig(
            output_filepath,
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
        )


# %% [markdown]
# # Simple plot

# %%
rc = {
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 8,
}

plot_clustermap(rc, None, OUTPUT_FIGURES_DIR / "global_clustermap-plain.svg")

# %% [markdown]
# # Take top cell types/tissues in LVs

# %% [markdown]
# The code below is not used now. The idea is to describe each LV taking the top two or three cell types/tissues.

# %%
# import re
# from collections import defaultdict

# from tqdm import tqdm

# from data.recount2 import LVAnalysis

# %%
# well_aligned_lvs_df = (
#     well_aligned_lvs.assign(
#         lv_name=well_aligned_lvs["LV index"].apply(lambda x: f"LV{x}")
#     )
#     .set_index(["lv_name", "pathway"])
#     .sort_index()
# )

# %%
# well_aligned_lvs_df.loc["LV960"].index

# %%
# selected_lvs = plot_data.drop(columns=["cluster_name", "color"]).columns.tolist()

# %%
# selected_lvs[:10]

# %%
# def _my_func(x):
#     _cols = [c for c in x.index if not c.startswith("LV")]
#     _tmp = x[_cols].dropna()
#     if _tmp.shape[0] > 0:
#         return _tmp.iloc[0]

#     return None

# %%
# import warnings
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from utils import chunker

# %%
# cell_types_tissues_replacements = {
#     # Neutrophils
#     "primary human neutrophils": "Neutrophils",
#     "Neutrophil isolated from peripheral blood": "Neutrophils",
#     "Neutrophil": "Neutrophils",
#     "neutrophils (Neu)": "Neutrophils",
#     # granulocytes
#     "granulocyte": "Granulocytes",
#     # monocytes
#     "primary human monocytes": "Monocytes",
#     # whole blood
#     "Whole Blood": "Whole blood",
#     # PBMC
#     "primary human PBMC": "PBMC",
#     # B-cells
#     "primary human B cells": "B cells",
#     # T-cells
#     "primary human T cells": "T cells",
#     # epithelial cells
#     "epithelial cells (Epi)": "Epithelial cells",
#     "primary human myeloid DC": "mDCs",
# }

# %%
# def _get_combined_results(phenos):
#     lvs_info = defaultdict(dict)

#     for lv_name in phenos:  # tqdm(["LV136", "LV538"])::
#         #         print(lv_name, end=", ", flush=True)

#         lvs_info[lv_name]["pathways"] = well_aligned_lvs_df.loc[lv_name].index.tolist()

#         lv_obj = LVAnalysis(lv_name, data, debug=False)
#         lv_data = lv_obj.get_experiments_data()

#         # tissue
#         lv_attrs = lv_obj.get_attributes_variation_score()
#         _tmp = pd.Series(lv_attrs.index)
#         lv_attrs = lv_attrs[
#             _tmp.str.match(
#                 "(?:tissue$)|(?:tissue[^\w]*type$)",
#                 case=False,
#                 flags=re.IGNORECASE,
#             ).values
#         ].sort_values(ascending=False)

#         lv_attrs_data = lv_data[lv_attrs.index.tolist() + [lv_name]]
#         lv_attrs_data = lv_attrs_data.assign(
#             tissue_final=lv_attrs_data.apply(_my_func, axis=1)
#         )
#         lv_attrs_data = lv_attrs_data.drop(columns=lv_attrs.index)
#         #     lv_attrs_data = lv_attrs_data[lv_attrs_data[lv_name] > 0.0].dropna()
#         lv_attrs_data = lv_attrs_data.dropna().sort_values(lv_name, ascending=False)
#         lv_attrs_data = lv_attrs_data.replace(
#             {
#                 "tissue_final": cell_types_tissues_replacements,
#             }
#         )

#         #     for lva in lv_attrs.index:
#         #         lv_attr_data = lv_data[[lva, lv_name]].dropna().sort_values(
#         #             lv_name, ascending=False
#         #         )

#         #         lv_attr_data = lv_attr_data[lv_attr_data[lv_name] > 0.0]

#         _tmp = lv_attrs_data.groupby("tissue_final").median().squeeze()
#         if not isinstance(_tmp, float):
#             _tmp = _tmp.sort_values(ascending=False)
#             lvs_info[lv_name]["tissue"] = _tmp.head(10).index.tolist()

#         # cell type
#         lv_attrs = lv_obj.get_attributes_variation_score()
#         _tmp = pd.Series(lv_attrs.index)
#         lv_attrs = lv_attrs[
#             _tmp.str.match(
#                 "(?:cell[^\w]*type$)",
#                 case=False,
#                 flags=re.IGNORECASE,
#             ).values
#         ].sort_values(ascending=False)

#         lv_attrs_data = lv_data[lv_attrs.index.tolist() + [lv_name]]
#         lv_attrs_data = lv_attrs_data.assign(attr=lv_attrs_data.apply(_my_func, axis=1))
#         lv_attrs_data = lv_attrs_data.drop(columns=lv_attrs.index)
#         #     lv_attrs_data = lv_attrs_data[lv_attrs_data[lv_name] > 0.0].dropna()
#         lv_attrs_data = lv_attrs_data.dropna().sort_values(lv_name, ascending=False)
#         lv_attrs_data = lv_attrs_data.replace(
#             {
#                 "attr": cell_types_tissues_replacements,
#             }
#         )

#         #     for lva in lv_attrs.index:
#         #         lv_attr_data = lv_data[[lva, lv_name]].dropna().sort_values(
#         #             lv_name, ascending=False
#         #         )

#         #         lv_attr_data = lv_attr_data[lv_attr_data[lv_name] > 0.0]

#         _tmp = lv_attrs_data.groupby("attr").median().squeeze()
#         if not isinstance(_tmp, float):
#             _tmp = _tmp.sort_values(ascending=False)
#             lvs_info[lv_name]["attr"] = _tmp.head(10).index.tolist()

#     return lvs_info

# %%
# # phenotype_chunks = chunker(selected_lvs, int(len(selected_lvs) / conf.GENERAL["N_JOBS"]))
# phenotype_chunks = list(chunker(selected_lvs, 5))

# %%
# len(phenotype_chunks)

# %%
# phenotype_chunks[:4]

# %%
# # def _run(tissue, column, phenotype_chunks, n_jobs=conf.N_JOBS_HIGH):
# all_results = {}

# with warnings.catch_warnings():
#     warnings.filterwarnings("ignore", category=UserWarning)

#     with ProcessPoolExecutor(max_workers=N_JOBS) as executor:
#         tasks = [
#             executor.submit(_get_combined_results, chunk) for chunk in phenotype_chunks
#         ]
#         for future in tqdm(as_completed(tasks), total=len(phenotype_chunks)):
#             res = future.result()
#             all_results.update(res)

# %%
# all_results["LV746"]

# %%
# all_results["LV70"]

# %%
# all_results["LV136"]

# %%
# all_results["LV538"]


# %% [markdown]
# ## Plot with cell types

# %%
# _get_lv_label("LV140")

# %%
# # with cell types as labels
# rc = {
#     #     "font.size": 9,
#     #     "xtick.labelsize": 10,
#     #     "ytick.labelsize": 19,
# }

# plot_clustermap(
#     rc, _get_lv_label, OUTPUT_FIGURES_DIR / "global_clustermap-cell_types.svg"
# )

# %% [markdown]
# ## Plot with pathways

# %%
# _get_lv_pathway("LV603")

# %%
# # with pathways as labels
# rc = {
#     #     "font.size": 9,
#     #     "xtick.labelsize": 10,
#     #     "ytick.labelsize": 19,
# }

# plot_clustermap(
#     rc, _get_lv_pathway, OUTPUT_FIGURES_DIR / "global_clustermap-pathways.svg"
# )

# %%
