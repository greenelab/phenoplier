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

# %% [markdown] tags=[]
# # Description

# %% [markdown] tags=[]
# It generates a figure with the agreement of the final consensus partitions with the ensemble using three scenarios:
# * The real data (using PhenomeXcan TWAS results projected into the LV space).
# * Two null scenarios where structure in data is destroyed to assess the significance of clustering results in the real data.

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
from pathlib import Path
from IPython.display import display

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import conf

# %% [markdown]
# # Settings

# %%
OUTPUT_FIGURES_DIR = Path(conf.MANUSCRIPT["FIGURES_DIR"], "clustering").resolve()
display(OUTPUT_FIGURES_DIR)
OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown] tags=[]
# # Load best partitions

# %%
scenarios_results = {}

# %% [markdown] tags=[]
# ## Runs on real data

# %%
CONSENSUS_CLUSTERING_DIR = Path(
    conf.RESULTS["CLUSTERING_DIR"], "consensus_clustering"
).resolve()

display(CONSENSUS_CLUSTERING_DIR)

# %% tags=[]
input_file = Path(CONSENSUS_CLUSTERING_DIR, "best_partitions_by_k.pkl").resolve()
display(input_file)

# %% tags=[]
best_partitions = pd.read_pickle(input_file)

# %% tags=[]
best_partitions.shape

# %% tags=[]
best_partitions.head()

# %%
best_partitions["selected"].value_counts()

# %%
# get threshold
best_parts_stats = best_partitions["ari_median"].describe()
display(best_parts_stats)

# %%
best_threshold = best_parts_stats["75%"]
best_threshold_description = "75th percentile"
display(best_threshold)

# %%
scenarios_results["Real data"] = (best_partitions, best_parts_stats, best_threshold)

# %% [markdown] tags=[]
# ## Runs on simulated null scenario 1

# %% [markdown] tags=[]
# The input data for all cases are TWAS results: a matrix with genes x traits, and p-values/z-scores in cells. We call this matrix M.
# For this Null #1 scenario, we shuffle genes in M, and then this shuffled matrix is projected into the LV space, where clustering is performed.
# Here we read these results.

# %% tags=[]
NULL_DIR = conf.RESULTS["CLUSTERING_NULL_DIR"] / "shuffle_genes"

# %%
CONSENSUS_CLUSTERING_DIR = Path(NULL_DIR, "consensus_clustering").resolve()

display(CONSENSUS_CLUSTERING_DIR)

# %% tags=[]
input_file = Path(CONSENSUS_CLUSTERING_DIR, "best_partitions_by_k.pkl").resolve()
display(input_file)

# %% tags=[]
best_partitions = pd.read_pickle(input_file)

# %% tags=[]
best_partitions.shape

# %% tags=[]
best_partitions.head()

# %%
best_partitions["selected"].value_counts()

# %%
# get threshold
best_parts_stats = best_partitions["ari_median"].describe()
display(best_parts_stats)

# %%
best_threshold = best_parts_stats["75%"]
best_threshold_description = "75th percentile"
display(best_threshold)

# %%
scenarios_results["Null #1"] = (best_partitions, best_parts_stats, best_threshold)

# %% [markdown] tags=[]
# ## Runs on simulated null scenario 2

# %% [markdown] tags=[]
# For this Null #2 scenario, we projected the original M matrix into the latent space, and then shuffle LVs in the projected matrix.
# Here we read these results.

# %% tags=[]
NULL_DIR = conf.RESULTS["CLUSTERING_NULL_DIR"] / "shuffle_lvs"

# %%
CONSENSUS_CLUSTERING_DIR = Path(NULL_DIR, "consensus_clustering").resolve()

display(CONSENSUS_CLUSTERING_DIR)

# %% tags=[]
input_file = Path(CONSENSUS_CLUSTERING_DIR, "best_partitions_by_k.pkl").resolve()
display(input_file)

# %% tags=[]
best_partitions = pd.read_pickle(input_file)

# %% tags=[]
best_partitions.shape

# %% tags=[]
best_partitions.head()

# %%
best_partitions["selected"].value_counts()

# %%
# get threshold
best_parts_stats = best_partitions["ari_median"].describe()
display(best_parts_stats)

# %%
best_threshold = best_parts_stats["75%"]
best_threshold_description = "75th percentile"
display(best_threshold)

# %%
scenarios_results["Null #2"] = (best_partitions, best_parts_stats, best_threshold)

# %% [markdown] tags=[]
# # Plot of selected best partitions

# %%
real_threshold = scenarios_results["Real data"][2]
display(real_threshold)

# %%
all_data = []
for k, v in scenarios_results.items():
    all_data.append(v[0].assign(scenario=k).reset_index())

# %% tags=[]
plot_data = pd.concat(all_data, ignore_index=True)
display(plot_data.shape)
display(plot_data.head(5))
assert plot_data.shape[0] == 3 * 59

# %%
_tmp = best_partitions.columns[
    ~best_partitions.columns.isin(["k", "method", "partition", "selected"])
]
assert _tmp.shape[0] == 1

_measure_col = _tmp[0]
display(_measure_col)

# %%
_selected_measure = "ARI"
_selected_stat = "Median"

# %% tags=[]
with sns.plotting_context("talk", font_scale=0.75), sns.axes_style(
    "whitegrid", {"grid.linestyle": "--"}
), sns.color_palette("muted"):
    current_palette = iter(sns.color_palette())

    fig, ax = plt.subplots(figsize=(18, 6))
    ax = sns.pointplot(
        # data=plot_data, x="k", y=_measure_col, color=next(current_palette), hue="scenario",
        data=plot_data.rename(columns={"scenario": "Scenario"}),
        x="k",
        y=_measure_col,
        hue="Scenario",
    )
    ax.axhline(
        real_threshold,
        ls="--",
        # color=next(current_palette),
        label=best_threshold_description,
    )
    ax.set_ylabel(f"Agreement with ensemble\n({_selected_stat} {_selected_measure})")
    ax.set_xlabel("Number of clusters ($k$)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    # plt.legend()
    plt.grid(True)
    #     plt.tight_layout()

    output_filepath = OUTPUT_FIGURES_DIR / "selected_best_partitions_by_k.svg"
    display(output_filepath)

    plt.savefig(
        output_filepath,
        #         dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )

# %% tags=[]
