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
# It analyzes how consensus partitions generated before agree with the ensemble, and selects the best ones for downstream analyses.

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

# %% [markdown] tags=[]
# # Load consensus clustering results

# %%
NULL_DIR = conf.RESULTS["CLUSTERING_NULL_DIR"] / "shuffle_genes"

# %% tags=[]
CONSENSUS_CLUSTERING_DIR = Path(
    NULL_DIR, "consensus_clustering"
).resolve()

display(CONSENSUS_CLUSTERING_DIR)

# %% tags=[]
input_file = Path(CONSENSUS_CLUSTERING_DIR, "consensus_clustering_runs.pkl").resolve()
display(input_file)

# %% tags=[]
consensus_clustering_results = pd.read_pickle(input_file)

# %% tags=[]
consensus_clustering_results.shape

# %% tags=[]
consensus_clustering_results.head()

# %% [markdown] tags=[]
# # Explore clustering indexes

# %% tags=[]
_col = "ari_mean"

_best_parts_by_ari = (
    consensus_clustering_results.groupby("k")
    .apply(lambda x: x.sort_values(_col, ascending=False).head(1))
    .sort_values(_col, ascending=False)[["method", "k", _col]]
    .rename(columns={_col: "index_value"})
)

# %% tags=[]
_col = "ami_mean"

_best_parts_by_ami = (
    consensus_clustering_results.groupby("k")
    .apply(lambda x: x.sort_values(_col, ascending=False).head(1))
    .sort_values(_col, ascending=False)[["method", "k", _col]]
    .rename(columns={_col: "index_value"})
)

# %% tags=[]
_col = "nmi_mean"

_best_parts_by_nmi = (
    consensus_clustering_results.groupby("k")
    .apply(lambda x: x.sort_values(_col, ascending=False).head(1))
    .sort_values(_col, ascending=False)[["method", "k", _col]]
    .rename(columns={_col: "index_value"})
)

# %% tags=[]
_indexes_colors = sns.color_palette("colorblind", 3)
display(_indexes_colors)

# %% tags=[]
with sns.plotting_context("talk", font_scale=0.75), sns.axes_style(
    "whitegrid", {"grid.linestyle": "--"}
):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax = sns.pointplot(
        data=_best_parts_by_ari,
        x="k",
        y="index_value",
        color=_indexes_colors[0],
        ci=None,
    )
    ax = sns.pointplot(
        data=_best_parts_by_ami,
        x="k",
        y="index_value",
        color=_indexes_colors[1],
        ci=None,
    )
    ax = sns.pointplot(
        data=_best_parts_by_nmi,
        x="k",
        y="index_value",
        color=_indexes_colors[2],
        ci=None,
    )

    ax.set_ylabel(f"Agreement with ensemble")
    ax.set_xlabel("Number of clusters ($k$)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.legend(labels=["ARI", "AMI", "NMI"])
    plt.grid(True)
    plt.tight_layout()

# %% [markdown] tags=[]
# AMI and NMI show the same trend for higher `k`. That's surprising. I would have expected that AMI has the same pattern as ARI, since both are adjusted-for-chance, and should not show higher values for higher `k` as it is expected for a not adjusted-for-chance index as NMI.
#
# **CONCLUSION:** I will pick ARI for the follow up analysis.

# %% [markdown] tags=[]
# # Explore best partition per k

# %% tags=[]
_selected_measure = "ARI"
_mean_column, _median_column = "ari_mean", "ari_median"

# %% tags=[]
best_parts_by_mean = (
    consensus_clustering_results.groupby("k")
    .apply(lambda x: x.sort_values(_mean_column, ascending=False).head(1))
    .sort_values(_mean_column, ascending=False)[["method", "k", _mean_column]]
)
display(best_parts_by_mean.head(10))

# %% tags=[]
best_parts_by_median = (
    consensus_clustering_results.groupby("k")
    .apply(lambda x: x.sort_values(_median_column, ascending=False).head(1))
    .sort_values(_median_column, ascending=False)[["method", "k", _median_column]]
)
display(best_parts_by_median.head(10))

# %% tags=[]
with sns.plotting_context("talk", font_scale=0.75), sns.axes_style(
    "whitegrid", {"grid.linestyle": "--"}
):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax = sns.pointplot(
        data=best_parts_by_mean,
        x="k",
        y=_mean_column,
        ci=None,
        color=_indexes_colors[0],
        label="Mean",
    )
    ax = sns.pointplot(
        data=best_parts_by_median,
        x="k",
        y=_median_column,
        ci=None,
        color=_indexes_colors[1],
        label="Median",
        ax=ax,
    )
    ax.set_ylabel(f"Agreement with ensemble ({_selected_measure})")
    ax.set_xlabel("Number of clusters ($k$)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.legend(labels=["Mean", "Median"])
    plt.grid(True)
    plt.tight_layout()

# %% [markdown] tags=[]
# Both central tendency measures (the mean and the median) have the same behevior: higher agreement on lower/medium k values, and lower agreement on higher k values.

# %% [markdown] tags=[]
# # Which consensus method performs better?

# %% [markdown] tags=[]
# For this comparison, I take the partitions with an agreement higher than the 75th percentile. From this set, I count how many times each method won.

# %% [markdown] tags=[]
# ## Using best by mean

# %% tags=[]
_stats_data = best_parts_by_mean[_mean_column].describe()
display(_stats_data)

# %% tags=[]
best_parts_by_mean[best_parts_by_mean[_mean_column] > _stats_data["75%"]][
    "method"
].value_counts()

# %% [markdown] tags=[]
# SCC picked the "best partition" 14 times, whereas EAC (hierarhical clustering) did it only once.

# %% [markdown] tags=[]
# ## Using best by median

# %% tags=[]
_stats_data = best_parts_by_median[_median_column].describe()
display(_stats_data)

# %% tags=[]
best_parts_by_median[best_parts_by_median[_median_column] > _stats_data["75%"]][
    "method"
].value_counts()

# %% [markdown] tags=[]
# If we use the "best partitions by median", EAC (HC) picked the best one 5 times, whereas SCC did it 10 times.

# %% [markdown] tags=[]
# **CONCLUSION:** we select SCC as the method for follow up analysis.

# %% [markdown] tags=[]
# # Select best partition per k

# %% tags=[]
_selected_stat = "Median"
_measure_col = _median_column

# %% tags=[]
best_parts = (
    consensus_clustering_results[
        consensus_clustering_results["method"].str.startswith("scc_")
    ]
    .groupby("k")
    .apply(lambda x: x.sort_values(_measure_col, ascending=False).head(1))
    .sort_values(_measure_col, ascending=False)[
        ["method", "k", "partition", _measure_col]
    ]
)

# %% tags=[]
best_parts = best_parts.set_index("k")

# %% tags=[]
best_parts.shape

# %% tags=[]
# show partitions with top values
best_parts.head(10)

# %% tags=[]
best_parts.sort_values("k")

# %% [markdown] tags=[]
# ## Select partitions with highest agreement

# %% [markdown] tags=[]
# We do not expect all partitions with different `k` to be good ones. Thus, here I select the partitions with an ensemble agreement that pass a relative high threshold (75th percentile).

# %% tags=[]
best_parts_stats = best_parts[_measure_col].describe()
display(best_parts_stats)

# %% tags=[]
best_threshold = best_parts_stats["75%"]
best_threshold_description = "75th percentile"
display(best_threshold)

best_parts = best_parts.assign(
    selected=best_parts[_measure_col].apply(lambda x: x >= best_threshold)
)

# %% tags=[]
best_parts.shape

# %% tags=[]
best_parts.head()

# %% [markdown] tags=[]
# ## Save best partitions per k

# %% tags=[]
output_file = Path(CONSENSUS_CLUSTERING_DIR, "best_partitions_by_k.pkl").resolve()
display(output_file)

# %% tags=[]
best_parts.to_pickle(output_file)

# %% [markdown] tags=[]
# # Plot of selected best partitions

# %% tags=[]
plot_data = best_parts.reset_index()
display(plot_data.head(5))

# %% tags=[]
with sns.plotting_context("talk", font_scale=0.75), sns.axes_style(
    "whitegrid", {"grid.linestyle": "--"}
), sns.color_palette("muted"):
    current_palette = iter(sns.color_palette())

    fig, ax = plt.subplots(figsize=(12, 6))
    ax = sns.pointplot(
        data=plot_data, x="k", y=_measure_col, color=next(current_palette)
    )
    ax.axhline(
        best_threshold,
        ls="--",
        color=next(current_palette),
        label=best_threshold_description,
    )
    ax.set_ylabel(f"Agreement with ensemble\n({_selected_stat} {_selected_measure})")
    ax.set_xlabel("Number of clusters ($k$)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

# %% [markdown] tags=[]
# The horizontal line in the plot is the median of the average agreement value; partitions above that line are marked as selected for downstream analysis

# %% tags=[]
# this list shows the selected final partitions, and which methods achieved the highest agreement
plot_data[plot_data["selected"]].sort_values("k")

# %% [markdown] tags=[]
# From the two evidence accumulation approaches (EAC) we are using, the spectral clustering based one does it better for almost all `k` values, whereas the hierarchical clustering based approach seems to do a little bit better for lower `k`.

# %% tags=[]
