# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
#     formats: ipynb,py//R:percent
#     text_representation:
#       extension: .R
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: R
#     language: R
#     name: ir
# ---

# %% [markdown]
# # Description

# %% [markdown]
# It uses the `clustree` package to generate clustering tree visualizations.

# %% [markdown]
# # Modules loading

# %%
library(clustree)
library(tidyverse)

# %% [markdown]
# # Settings

# %%
CLUSTERING_DIR <- Sys.getenv("PHENOPLIER_RESULTS_CLUSTERING_DIR")

# %%
CLUSTERING_DIR

# %%
CONSENSUS_CLUSTERING_DIR = file.path(CLUSTERING_DIR, "consensus_clustering")

# %%
CONSENSUS_CLUSTERING_DIR

# %%
MANUSCRIPT_FIGURES_DIR <- Sys.getenv("PHENOPLIER_MANUSCRIPT_FIGURES_DIR")

# %%
MANUSCRIPT_FIGURES_DIR

# %% [markdown]
# # Load data

# %%
data <- read_tsv(file.path(CONSENSUS_CLUSTERING_DIR, "clustering_tree_data.tsv"))

# %%
dim(data)

# %%
head(data)

# %% [markdown]
# # Plot clustering tree

# %% [markdown]
# ## Plain

# %%
options(repr.plot.width = 20, repr.plot.height = 15)
clustree(data, prefix = "k")

ggsave(
    file.path(MANUSCRIPT_FIGURES_DIR, "clustering", "clustering_tree.pdf"),
    height=15,
    width=20,
    scale=1,
)

# %%
