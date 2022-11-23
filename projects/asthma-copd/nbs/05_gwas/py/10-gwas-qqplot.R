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
# It takes a GWAS on a random phenotype and verifies that the Manhattan and QQ-plots look fine (without inflation).

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
library(tidyverse)

# %% tags=[]
library(qqman)

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
DATA_DIR <- Sys.getenv("PHENOPLIER_PROJECTS_ASTHMA_COPD_DATA_DIR")

# %% tags=[]
DATA_DIR

# %% tags=[]
INPUT_GWAS_DIR <- file.path(DATA_DIR, "gwas")

# %% tags=[]
INPUT_GWAS_DIR

# %% [markdown] tags=[]
# # Asthma only

# %% tags=[]
gwas_title <- "Asthma only"

# %% [markdown] tags=[]
# ## Load data

# %% tags=[]
gwas <- as.data.frame(read_table(file.path(INPUT_GWAS_DIR, "GWAS_Asthma_only_GLM_SNPs_info0.7.txt")))

# %% tags=[]
dim(gwas)

# %% tags=[]
head(gwas)

# %% tags=[]
gwas <- gwas %>% filter(P >= 0 & P <= 1)

# %% tags=[]
dim(gwas)

# %% [markdown] tags=[]
# ## Stats

# %% tags=[]
summary(gwas)

# %% [markdown] tags=[]
# ## Manhattan plot

# %% tags=[]
options(repr.plot.width = 20, repr.plot.height = 10)

manhattan(
  gwas,
  chr = "#CHROM",
  bp = "POS",
  p = "P",
  snp = "ID",
  main = gwas_title,
  suggestiveline = F,
  genomewideline = -log10(5e-08),
  cex = 0.6,
  cex.axis = 0.9,
)

# %% [markdown] tags=[]
# ## QQ-plot

# %% tags=[]
options(repr.plot.width = 10, repr.plot.height = 10)

qq(
  gwas$P,
  main = paste0("Q-Q plot - ", gwas_title)
)

# %% [markdown] tags=[]
# # COPD only

# %% tags=[]
gwas_title <- "COPD only"

# %% [markdown] tags=[]
# ## Load data

# %% tags=[]
gwas <- as.data.frame(read_table(file.path(INPUT_GWAS_DIR, "GWAS_COPD_only_GLM_SNPs_info0.7.txt")))

# %% tags=[]
dim(gwas)

# %% tags=[]
head(gwas)

# %% tags=[]
gwas <- gwas %>% filter(P >= 0 & P <= 1)

# %% tags=[]
dim(gwas)

# %% [markdown] tags=[]
# ## Stats

# %% tags=[]
summary(gwas)

# %% [markdown] tags=[]
# ## Manhattan plot

# %% tags=[]
options(repr.plot.width = 20, repr.plot.height = 10)

manhattan(
  gwas,
  chr = "#CHROM",
  bp = "POS",
  p = "P",
  snp = "ID",
  main = gwas_title,
  suggestiveline = F,
  genomewideline = -log10(5e-08),
  cex = 0.6,
  cex.axis = 0.9,
)

# %% [markdown] tags=[]
# ## QQ-plot

# %% tags=[]
options(repr.plot.width = 10, repr.plot.height = 10)

qq(
  gwas$P,
  main = paste0("Q-Q plot - ", gwas_title)
)

# %% [markdown] tags=[]
# # Asthma-COPD Overlap Syndrome (ACOS)

# %% tags=[]
gwas_title <- "ACOS"

# %% [markdown] tags=[]
# ## Load data

# %% tags=[]
gwas <- as.data.frame(read_table(file.path(INPUT_GWAS_DIR, "GWAS_ACO_GLM_SNPs_info0.7.txt")))

# %% tags=[]
dim(gwas)

# %% tags=[]
head(gwas)

# %% tags=[]
gwas <- gwas %>% filter(P >= 0 & P <= 1)

# %% tags=[]
dim(gwas)

# %% [markdown] tags=[]
# ## Stats

# %% tags=[]
summary(gwas)

# %% [markdown] tags=[]
# ## Manhattan plot

# %% tags=[]
options(repr.plot.width = 20, repr.plot.height = 10)

manhattan(
  gwas,
  chr = "#CHROM",
  bp = "POS",
  p = "P",
  snp = "ID",
  main = gwas_title,
  suggestiveline = F,
  genomewideline = -log10(5e-08),
  cex = 0.6,
  cex.axis = 0.9,
)

# %% [markdown] tags=[]
# ## QQ-plot

# %% tags=[]
options(repr.plot.width = 10, repr.plot.height = 10)

qq(
  gwas$P,
  main = paste0("Q-Q plot - ", gwas_title)
)

# %% tags=[]
