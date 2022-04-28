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
# TODO

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
hwe <- as.data.frame(read_table(file.path(SUBSETS_DIR, "all_phase3.3.hwe"), col_types = cols_only(P = 'n')))

# %%
dim(hwe)

# %%
head(hwe)

# %% [markdown]
# # Histogram on all SNPs

# %%
hist(hwe[,1],main="Histogram HWE")

# %% [markdown]
# # Histogram on strongly HWE deviating SNPs only

# %%
hwe %>% filter(P < 0.01)  %>% dim_desc

# %%
hwe %>% filter(P < 1e-2)  %>% dim_desc

# %%
hwe %>% filter(P < 1e-4)  %>% dim_desc

# %%
hwe %>% filter(P < 1e-6)  %>% dim_desc

# %%
hwe %>% filter(P < 1e-10)  %>% dim_desc

# %%
hwe %>% filter(P < 1e-15)  %>% dim_desc

# %%
hwe_zoom <- hwe %>% filter(P < 1e-10)

# %%
hist(hwe_zoom[,1],main="Histogram HWE: strongly deviating SNPs only")

# %%
