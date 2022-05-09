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
# It reads outputs from sex discrepancy previosly computed, and plots the inbreeding coefficient (https://www.cog-genomics.org/plink/1.9/formats#sexcheck) for samples to know whether self-reported sex and imputed sex from X chromosome match.

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

# %%
gender <- read.table(file.path(SUBSETS_DIR, "all_phase3.1.split_x.sexcheck.sexcheck"), header=T,as.is=T)

# %%
dim(gender)

# %%
head(gender)

# %% [markdown] tags=[]
# # Histogram

# %% tags=[]
hist(gender[,6],main="Gender", xlab="F")

# %% tags=[]
gender %>% summarise(mean = mean(F), sd = sd(F), max = max(F), min = min(F))

# %% tags=[]
gender %>%
  filter(F < 0.20) %>%
  dim_desc()

# %% tags=[]
gender %>%
  filter(F < 0.21) %>%
  dim_desc()

# %% tags=[]
gender %>%
  filter(F < 0.22) %>%
  dim_desc()

# %% tags=[]
gender %>%
  filter(F < 0.23) %>%
  dim_desc()

# %% tags=[]
gender %>%
  filter(F < 0.24) %>%
  dim_desc()

# %% tags=[]
gender %>%
  filter(F < 0.25) %>%
  dim_desc()

# %% tags=[]
gender %>%
  filter(F > 0.80) %>%
  dim_desc()

# %% tags=[]
gender %>%
  filter(STATUS == "PROBLEM") %>%
  dim_desc()

# %% tags=[]
