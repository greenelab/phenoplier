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
    "/tmp/clustering_tree.pdf",
    height=15,
    width=20,
    scale=1,
)

# %% [markdown]
# ## With labels

# %%
# label_position <- function(labels) {
#     if (length(unique(labels)) == 1) {
#         position <- as.character(unique(labels))
#     } else {
#         position <- NA
#     }
#     return(position)
# }

# options(repr.plot.width = 25, repr.plot.height = 15)
# clustree(data, prefix = "k", node_label="labels", node_label_aggr = "label_position")

# %% [markdown]
# # Plot overlay

# %% [markdown]
# ## With PCA

# %%
# options(repr.plot.width = 15, repr.plot.height = 11)
# clustree_overlay(data, prefix = "k", x_value = "PCA1", y_value = "PCA2")

# %% [markdown]
# ## With UMAP

# %%
# options(repr.plot.width = 15, repr.plot.height = 11)
# clustree_overlay(data, prefix = "k", x_value = "UMAP1", y_value = "UMAP2")

# %%
