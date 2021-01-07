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
# TODO

# %%
library(clustree)
library(tidyverse)

# %%
# FIXME: hardcoded
orig_data <- read_tsv(
    '/media/miltondp/Elements1/projects/phenoplier/results/clustering/consensus_clustering/clustering_tree_data.tsv',
)

# %%
dim(orig_data)

# %%
head(orig_data)

# %%
# data = select(orig_data, c(labels, PCA1, PCA2, UMAP1, UMAP2, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20))
data = orig_data

# %%
dim(data)

# %%
head(data)

# %%
options(repr.plot.width = 20, repr.plot.height = 15)
clustree(data, prefix = "k")

# %%
# data = select(
#     orig_data,
#     c(
#         labels, PCA1, PCA2, UMAP1, UMAP2,
#         k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13, k14, k15, k16, k17, k18, k19, k20, k21, k22, k23, k24, k25, k26, k27,
#         k28, k29, k30, k31, k32, k33, k34, k35, k36, k37, k38, k39, k40
#      )
# )

# %%
# options(repr.plot.width = 20, repr.plot.height = 15)
# clustree(data, prefix = "k")

# %%
label_position <- function(labels) {
    if (length(unique(labels)) == 1) {
        position <- as.character(unique(labels))
    } else {
        position <- NA
    }
    return(position)
}

options(repr.plot.width = 25, repr.plot.height = 15)
clustree(data, prefix = "k", node_label="labels", node_label_aggr = "label_position")

# %%
options(repr.plot.width = 15, repr.plot.height = 11)
clustree_overlay(data, prefix = "k", x_value = "PCA1", y_value = "PCA2")

# %%
options(repr.plot.width = 15, repr.plot.height = 11)
clustree_overlay(data, prefix = "k", x_value = "UMAP1", y_value = "UMAP2")

# %%
