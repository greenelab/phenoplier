# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
#     formats: ipynb,py//R:percent
#     text_representation:
#       extension: .R
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: R
#     language: R
#     name: ir
# ---

# %% [markdown] tags=[]
# # Description

# %% [markdown] tags=[]
# It uses the `clustree` package to generate clustering tree visualizations.

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
library(clustree)
library(tidyverse)

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
CLUSTERING_DIR <- Sys.getenv("PHENOPLIER_RESULTS_CLUSTERING_DIR")

# %% tags=[]
CLUSTERING_DIR

# %% tags=[]
CONSENSUS_CLUSTERING_DIR <- file.path(CLUSTERING_DIR, "consensus_clustering")

# %% tags=[]
CONSENSUS_CLUSTERING_DIR

# %% tags=[]
MANUSCRIPT_FIGURES_DIR <- Sys.getenv("PHENOPLIER_MANUSCRIPT_FIGURES_DIR")

# %% tags=[]
if (MANUSCRIPT_FIGURES_DIR == "") {
  MANUSCRIPT_FIGURES_DIR <- "/tmp"
}

# %% tags=[]
MANUSCRIPT_FIGURES_DIR

# %% tags=[]
OUTPUT_FIG_DIR <- file.path(MANUSCRIPT_FIGURES_DIR, "clustering")
dir.create(OUTPUT_FIG_DIR, recursive = TRUE)

# %% tags=[]
OUTPUT_FIG_DIR

# %% [markdown] tags=[]
# # Load data

# %% tags=[]
data <- read_tsv(file.path(CONSENSUS_CLUSTERING_DIR, "clustering_tree_data.tsv"))

# %% tags=[]
dim(data)

# %% tags=[]
head(data)

# %% [markdown] tags=[]
# # Plot clustering tree

# %% [markdown] tags=[]
# ## Plain

# %% tags=[]
options(repr.plot.width = 20, repr.plot.height = 15)
clustree(data, prefix = "k")

ggsave(
  file.path(OUTPUT_FIG_DIR, "clustering_tree.svg"),
  height = 15,
  width = 20,
  scale = 1,
)

# %% tags=[]
