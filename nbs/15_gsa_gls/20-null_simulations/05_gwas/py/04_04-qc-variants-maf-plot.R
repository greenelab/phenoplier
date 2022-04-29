# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
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
# TODO

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
library(tidyverse)

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
A1000G_GENOTYPES_DIR <- Sys.getenv("PHENOPLIER_A1000G_GENOTYPES_DIR")

# %% tags=[]
A1000G_GENOTYPES_DIR

# %% tags=[]
SUBSETS_DIR <- file.path(A1000G_GENOTYPES_DIR, "subsets")

# %% tags=[]
SUBSETS_DIR

# %% [markdown] tags=[]
# # Load data

# %% tags=[]
maf_freq <- as.data.frame(read_table(file.path(SUBSETS_DIR, "all_phase3.2.maf.frq"), col_types = cols_only(MAF = "n")))

# %% tags=[]
dim(maf_freq)

# %% tags=[]
head(maf_freq)

# %% [markdown] tags=[]
# # Histogram

# %% tags=[]
hist(maf_freq[,1],main = "MAF distribution", xlab = "MAF")

# %% tags=[]
maf_freq %>%
  filter(MAF > 0.01) %>%
  dim_desc()

# %% tags=[]
maf_freq %>%
  filter(MAF > 0.05) %>%
  dim_desc()

# %% tags=[]
