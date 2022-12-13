# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
#     text_representation:
#       extension: .R
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: R
#     language: R
#     name: ir
# ---

# %% [markdown] tags=[]
# # Description

# %% [markdown] tags=[]
# It takes TWAS results and verifies that the QQ-plots look fine (without inflation).

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
library(tidyverse)

# %% tags=[]
library(qqman)

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
BASE_DIR <- Sys.getenv("PHENOPLIER_PROJECTS_ASTHMA_COPD_RESULTS_DIR")

# %% tags=[]
BASE_DIR

# %% tags=[]
SPREDIXCAN_DIR <- file.path(BASE_DIR, "twas", "spredixcan")

# %% tags=[]
SPREDIXCAN_DIR

# %% tags=[]
SMULTIXCAN_DIR <- file.path(BASE_DIR, "twas", "smultixcan")

# %% tags=[]
SMULTIXCAN_DIR

# %% [markdown] tags=[]
# # Asthma only

# %% tags=[]
gwas_title <- "Asthma only (imputed)"

# %% [markdown] tags=[]
# ## S-PrediXcan

# %% [markdown] tags=[]
# ### Load data

# %% tags=[]
twas <- as.data.frame(read_csv(file.path(SPREDIXCAN_DIR, "GWAS_Asthma_only_GLM_SNPs_info0.7-gtex_v8-mashr-Whole_Blood.csv")))

# %% tags=[]
dim(twas)

# %% tags=[]
head(twas)

# %% [markdown] tags=[]
# ### QQ-plot

# %% tags=[]
options(repr.plot.width = 10, repr.plot.height = 10)

qq(twas$pvalue, main = paste0("Q-Q plot of S-PrediXcan for ", gwas_title))

# %% [markdown] tags=[]
# ## S-MultiXcan

# %% [markdown] tags=[]
# ### Load data

# %% tags=[]
twas <- as.data.frame(read_table(file.path(SMULTIXCAN_DIR, "GWAS_Asthma_only_GLM_SNPs_info0.7-gtex_v8-mashr-smultixcan.txt")))

# %% tags=[]
dim(twas)

# %% tags=[]
head(twas)

# %% [markdown] tags=[]
# ### QQ-plot

# %% tags=[]
options(repr.plot.width = 10, repr.plot.height = 10)

qq(twas$pvalue, main = paste0("Q-Q plot of S-MultiXcan for ", gwas_title))

# %% [markdown] tags=[]
# # COPD only

# %% tags=[]
gwas_title <- "COPD only (imputed)"

# %% [markdown] tags=[]
# ## S-PrediXcan

# %% [markdown] tags=[]
# ### Load data

# %% tags=[]
twas <- as.data.frame(read_csv(file.path(SPREDIXCAN_DIR, "GWAS_COPD_only_GLM_SNPs_info0.7-gtex_v8-mashr-Whole_Blood.csv")))

# %% tags=[]
dim(twas)

# %% tags=[]
head(twas)

# %% [markdown] tags=[]
# ### QQ-plot

# %% tags=[]
options(repr.plot.width = 10, repr.plot.height = 10)

qq(twas$pvalue, main = paste0("Q-Q plot of S-PrediXcan for ", gwas_title))

# %% [markdown] tags=[]
# ## S-MultiXcan

# %% [markdown] tags=[]
# ### Load data

# %% tags=[]
twas <- as.data.frame(read_table(file.path(SMULTIXCAN_DIR, "GWAS_COPD_only_GLM_SNPs_info0.7-gtex_v8-mashr-smultixcan.txt")))

# %% tags=[]
dim(twas)

# %% tags=[]
head(twas)

# %% [markdown] tags=[]
# ### QQ-plot

# %% tags=[]
options(repr.plot.width = 10, repr.plot.height = 10)

qq(twas$pvalue, main = paste0("Q-Q plot of S-MultiXcan for ", gwas_title))

# %% [markdown] tags=[]
# # Asthma-COPD Overlap Syndrome (ACOS)

# %% tags=[]
gwas_title <- "ACOS (imputed)"

# %% [markdown] tags=[]
# ## S-PrediXcan

# %% [markdown] tags=[]
# ### Load data

# %% tags=[]
twas <- as.data.frame(read_csv(file.path(SPREDIXCAN_DIR, "GWAS_ACO_GLM_SNPs_info0.7-gtex_v8-mashr-Whole_Blood.csv")))

# %% tags=[]
dim(twas)

# %% tags=[]
head(twas)

# %% [markdown] tags=[]
# ### QQ-plot

# %% tags=[]
options(repr.plot.width = 10, repr.plot.height = 10)

qq(twas$pvalue, main = paste0("Q-Q plot of S-PrediXcan for ", gwas_title))

# %% [markdown] tags=[]
# ## S-MultiXcan

# %% [markdown] tags=[]
# ### Load data

# %% tags=[]
twas <- as.data.frame(read_table(file.path(SMULTIXCAN_DIR, "GWAS_ACO_GLM_SNPs_info0.7-gtex_v8-mashr-smultixcan.txt")))

# %% tags=[]
dim(twas)

# %% tags=[]
head(twas)

# %% [markdown] tags=[]
# ### QQ-plot

# %% tags=[]
options(repr.plot.width = 10, repr.plot.height = 10)

qq(twas$pvalue, main = paste0("Q-Q plot of S-MultiXcan for ", gwas_title))

# %% tags=[]
