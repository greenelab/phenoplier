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
# It takes TWAS results on a random phenotype and verifies that the QQ-plots look fine (without inflation).

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
library(tidyverse)

# %% tags=[]
library(qqman)

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
GLS_NULL_SIMS_DIR <- Sys.getenv("PHENOPLIER_RESULTS_GLS_NULL_SIMS")

# %% tags=[]
GLS_NULL_SIMS_DIR

# %% tags=[]
SPREDIXCAN_DIR <- file.path(GLS_NULL_SIMS_DIR, "twas", "spredixcan")

# %% tags=[]
SPREDIXCAN_DIR

# %% tags=[]
SMULTIXCAN_DIR <- file.path(GLS_NULL_SIMS_DIR, "twas", "smultixcan")

# %% tags=[]
SMULTIXCAN_DIR

# %% [markdown] tags=[]
# # Random pheno 0

# %% [markdown] tags=[]
# ## S-PrediXcan

# %% [markdown] tags=[]
# ### Load data

# %% tags=[]
twas <- as.data.frame(read_csv(file.path(SPREDIXCAN_DIR, "random.pheno0-gtex_v8-mashr-Whole_Blood.csv")))

# %% tags=[]
dim(twas)

# %% tags=[]
head(twas)

# %% [markdown] tags=[]
# ### QQ-plot

# %% tags=[]
options(repr.plot.width = 10, repr.plot.height = 10)

qq(twas$pvalue, main = "Q-Q plot of TWAS p-values : log")

# %% [markdown] tags=[]
# ## S-MultiXcan

# %% [markdown] tags=[]
# ### Load data

# %% tags=[]
twas <- as.data.frame(read_table(file.path(SMULTIXCAN_DIR, "random.pheno0-gtex_v8-mashr-smultixcan.txt")))

# %% tags=[]
dim(twas)

# %% tags=[]
head(twas)

# %% [markdown] tags=[]
# ### QQ-plot

# %% tags=[]
options(repr.plot.width = 10, repr.plot.height = 10)

qq(twas$pvalue, main = "Q-Q plot of TWAS p-values : log")

# %% [markdown] tags=[]
# # Random pheno 28

# %% [markdown] tags=[]
# Random phenotype 28 has the largest inflation factor in the original GWAS summary stats (although within the acceptable limits).

# %% [markdown] tags=[]
# ## S-PrediXcan

# %% [markdown] tags=[]
# ### Load data

# %% tags=[]
twas <- as.data.frame(read_csv(file.path(SPREDIXCAN_DIR, "random.pheno28-gtex_v8-mashr-Whole_Blood.csv")))

# %% tags=[]
dim(twas)

# %% tags=[]
head(twas)

# %% [markdown] tags=[]
# ### QQ-plot

# %% tags=[]
options(repr.plot.width = 10, repr.plot.height = 10)

qq(twas$pvalue, main = "Q-Q plot of TWAS p-values : log")

# %% [markdown] tags=[]
# ## S-MultiXcan

# %% [markdown] tags=[]
# ### Load data

# %% tags=[]
twas <- as.data.frame(read_table(file.path(SMULTIXCAN_DIR, "random.pheno28-gtex_v8-mashr-smultixcan.txt")))

# %% tags=[]
dim(twas)

# %% tags=[]
head(twas)

# %% [markdown] tags=[]
# ### QQ-plot

# %% tags=[]
options(repr.plot.width = 10, repr.plot.height = 10)

qq(twas$pvalue, main = "Q-Q plot of TWAS p-values : log")

# %% tags=[]
