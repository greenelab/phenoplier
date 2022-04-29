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
relatedness = read.table(file.path(SUBSETS_DIR, "all_phase3.6.pihat_min0.2.genome"), header=T)

# %%
dim(relatedness)

# %%
head(relatedness)

# %%
relatedness %>%  filter(PI_HAT > 0.20) %>% dim_desc

# %%
relatedness %>%  filter(RT != "OT") %>% dim_desc

# %% [markdown]
# All are inferred as OT (other).

# %%
relatedness %>% summarise(mean = mean(PI_HAT), sd = sd(PI_HAT), max = max(PI_HAT), min = min(PI_HAT))

# %%
indmiss <- as.data.frame(read_table(file.path(SUBSETS_DIR, "all_phase3.missingness.imiss")))

# %%
dim(indmiss)

# %%
head(indmiss)

# %%
