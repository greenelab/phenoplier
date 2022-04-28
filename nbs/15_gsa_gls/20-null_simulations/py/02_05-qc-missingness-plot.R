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
# Read the missing info computed previously (by individual and variant) and plots some histograms.

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
indmiss <- as.data.frame(read_table(file.path(SUBSETS_DIR, "all_phase3.missingness.imiss"), col_types = cols_only(F_MISS = "n")))

# %% tags=[]
dim(indmiss)

# %% tags=[]
head(indmiss)

# %% tags=[]
snpmiss <- as.data.frame(read_table(file.path(SUBSETS_DIR, "all_phase3.missingness.lmiss"), col_types = cols_only(F_MISS = "n")))

# %% tags=[]
dim(snpmiss)

# %% tags=[]
head(snpmiss)

# %% [markdown] tags=[]
# # Individuals

# %% tags=[]
hist(indmiss[, 1], main = "Histogram individual missingness")

# %% tags=[]
indmiss %>% summarise(mean = mean(F_MISS), sd = sd(F_MISS), max = max(F_MISS), min = min(F_MISS))

# %% tags=[]
indmiss %>%
  filter(F_MISS > 0) %>%
  dim_desc()

# %% tags=[]
indmiss %>%
  filter(F_MISS > 0.01) %>%
  dim_desc()

# %% [markdown] tags=[]
# # Variants

# %% tags=[]
hist(snpmiss[, 1], main = "Histogram SNP missingness")

# %% tags=[]
snpmiss %>% summarise(mean = mean(F_MISS), sd = sd(F_MISS), max = max(F_MISS), min = min(F_MISS))

# %% tags=[]
snpmiss %>%
  filter(F_MISS > 0) %>%
  dim_desc()

# %% tags=[]
snpmiss %>%
  filter(F_MISS > 0.01) %>%
  dim_desc()

# %% tags=[]
snpmiss %>%
  filter(F_MISS > 0.02) %>%
  dim_desc()

# %% [markdown] tags=[]
# Only remove variants with missingness > 0.01

# %% tags=[]
