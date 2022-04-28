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

# %% [markdown]
# # Description

# %% [markdown]
# Read the missing info computed previously (by individual and variant) and plots some histograms.

# %% [markdown]
# # Modules

# %%
library(tidyverse)

# %% [markdown]
# # Paths

# %%
A1000G_GENOTYPES_DIR <- Sys.getenv("PHENOPLIER_A1000G_GENOTYPES_DIR")

# %%
A1000G_GENOTYPES_DIR

# %%
SUBSETS_DIR <- file.path(A1000G_GENOTYPES_DIR, "subsets")

# %%
SUBSETS_DIR

# %% [markdown]
# # Load data

# %%
indmiss <- as.data.frame(read_table(file.path(SUBSETS_DIR, "missingness.imiss"), col_types = cols_only(F_MISS = "n")))

# %%
dim(indmiss)

# %%
head(indmiss)

# %%
snpmiss <- as.data.frame(read_table(file.path(SUBSETS_DIR, "missingness.lmiss"), col_types = cols_only(F_MISS = "n")))

# %%
dim(snpmiss)

# %%
head(snpmiss)

# %% [markdown]
# # Individuals

# %%
hist(indmiss[, 1], main = "Histogram individual missingness")

# %%
indmiss %>% summarise(mean = mean(F_MISS), sd = sd(F_MISS), max = max(F_MISS), min = min(F_MISS))

# %%
indmiss %>%
  filter(F_MISS > 0) %>%
  dim_desc()

# %%
indmiss %>%
  filter(F_MISS > 0.01) %>%
  dim_desc()

# %% [markdown]
# # Variants

# %%
hist(snpmiss[, 1], main = "Histogram SNP missingness")

# %%
snpmiss %>% summarise(mean = mean(F_MISS), sd = sd(F_MISS), max = max(F_MISS), min = min(F_MISS))

# %%
snpmiss %>%
  filter(F_MISS > 0) %>%
  dim_desc()

# %%
snpmiss %>%
  filter(F_MISS > 0.01) %>%
  dim_desc()

# %% [markdown]
# Only remove variants with missingness > 0.01

# %%
