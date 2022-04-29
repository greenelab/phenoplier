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
hwe <- as.data.frame(read_table(file.path(SUBSETS_DIR, "all_phase3.3.hwe"), col_types = cols_only(P = "n")))

# %% tags=[]
dim(hwe)

# %% tags=[]
head(hwe)

# %% [markdown] tags=[]
# # Histogram on all SNPs

# %% tags=[]
hist(hwe[, 1], main = "Histogram HWE")

# %% [markdown] tags=[]
# # Histogram on strongly HWE deviating SNPs only

# %% tags=[]
hwe %>%
  filter(P < 1e-1) %>%
  dim_desc()

# %% tags=[]
hwe %>%
  filter(P < 1e-2) %>%
  dim_desc()

# %% tags=[]
hwe %>%
  filter(P < 1e-4) %>%
  dim_desc()

# %% tags=[]
hwe %>%
  filter(P < 1e-6) %>%
  dim_desc()

# %% tags=[]
hwe %>%
  filter(P < 1e-8) %>%
  dim_desc()

# %% tags=[]
hwe %>%
  filter(P < 1e-10) %>%
  dim_desc()

# %% tags=[]
hwe %>%
  filter(P < 1e-15) %>%
  dim_desc()

# %% tags=[]
hwe_zoom <- hwe %>% filter(P < 5e-6)

# %% tags=[]
hist(hwe_zoom[, 1], main = "Histogram HWE: strongly deviating SNPs only")

# %% tags=[]
