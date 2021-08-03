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
# It generates a figure with the agreement of the final consensus partitions with the ensemble

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

# %% [markdown] tags=[]
# # Threshold

# %% tags=[]
best_parts_stats = best_partitions["ari_median"].describe()
display(best_parts_stats)

# %% tags=[]
best_threshold = best_parts_stats["75%"]
best_threshold_description = "75th percentile"
display(best_threshold)

# %% [markdown] tags=[]
# # Plot of selected best partitions

# %% tags=[]
plot_data = best_partitions.reset_index()
display(plot_data.head(5))

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
