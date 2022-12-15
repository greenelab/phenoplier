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
# It takes a GWAS that was imputed and postprocessed (using the PrediXcan scripts here https://github.com/hakyimlab/summary-gwas-imputation) and verifies that the Manhattan and QQ-plots look fine (without inflation).

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
library(tidyverse)

# %% tags=[]
library(qqman)

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
DATA_DIR <- Sys.getenv("PHENOPLIER_PROJECTS_ASTHMA_COPD_RESULTS_DIR")

# %% tags=[]
DATA_DIR

# %% tags=[]
INPUT_GWAS_DIR <- file.path(DATA_DIR, "final_imputed_gwas")

# %% tags=[]
INPUT_GWAS_DIR

# %% [markdown] tags=[]
# # Asthma only

# %% tags=[]
gwas_title <- "Asthma only (imputed)"

# %% [markdown] tags=[]
# ## Load data

# %% tags=[]
gwas <- as.data.frame(read_table(file.path(INPUT_GWAS_DIR, "GWAS_Asthma_only_GLM_SNPs_info0.7-harmonized-imputed.txt.gz")))

# %% tags=[]
dim(gwas)

# %% tags=[]
head(gwas)

# %% [markdown] tags=[]
# ### Extract chromosome

# %% tags=[]
unique(gwas$chromosome)

# %% tags=[]
gwas$chrom <- gsub("chr([0-9]+)", "\\1", gwas$chromosome)
gwas <- transform(gwas, chrom = as.numeric(chrom))

# %% tags=[]
unique(gwas$chrom)

# %% [markdown] tags=[]
# ## Stats

# %% tags=[]
summary(gwas)

# %% [markdown] tags=[]
# ## Remove NA pvalues

# %% tags=[]
dim(gwas)

# %% tags=[]
gwas <- gwas %>% filter(pvalue >= 0 & pvalue <= 1)

# %% tags=[]
dim(gwas)

# %% [markdown] tags=[]
# ## Manhattan plot

# %% tags=[]
options(repr.plot.width = 20, repr.plot.height = 10)

manhattan(
  gwas,
  chr = "chrom",
  bp = "position",
  p = "pvalue",
  snp = "variant_id",
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
  gwas$pvalue,
  main = paste0("Q-Q plot - ", gwas_title)
)

# %% [markdown] tags=[]
# # COPD only

# %% tags=[]
gwas_title <- "COPD only (imputed)"

# %% [markdown] tags=[]
# ## Load data

# %% tags=[]
gwas <- as.data.frame(read_table(file.path(INPUT_GWAS_DIR, "GWAS_COPD_only_GLM_SNPs_info0.7-harmonized-imputed.txt.gz")))

# %% tags=[]
dim(gwas)

# %% tags=[]
head(gwas)

# %% [markdown] tags=[]
# ### Extract chromosome

# %% tags=[]
unique(gwas$chromosome)

# %% tags=[]
gwas$chrom <- gsub("chr([0-9]+)", "\\1", gwas$chromosome)
gwas <- transform(gwas, chrom = as.numeric(chrom))

# %% tags=[]
unique(gwas$chrom)

# %% [markdown] tags=[]
# ## Stats

# %% tags=[]
summary(gwas)

# %% [markdown] tags=[]
# ## Remove NA pvalues

# %% tags=[]
dim(gwas)

# %% tags=[]
gwas <- gwas %>% filter(pvalue >= 0 & pvalue <= 1)

# %% tags=[]
dim(gwas)

# %% [markdown] tags=[]
# ## Manhattan plot

# %% tags=[]
options(repr.plot.width = 20, repr.plot.height = 10)

manhattan(
  gwas,
  chr = "chrom",
  bp = "position",
  p = "pvalue",
  snp = "variant_id",
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
  gwas$pvalue,
  main = paste0("Q-Q plot - ", gwas_title)
)

# %% [markdown] tags=[]
# # Asthma-COPD Overlap Syndrome (ACOS)

# %% tags=[]
gwas_title <- "ACOS (imputed)"

# %% [markdown] tags=[]
# ## Load data

# %% tags=[]
gwas <- as.data.frame(read_table(file.path(INPUT_GWAS_DIR, "GWAS_ACO_GLM_SNPs_info0.7-harmonized-imputed.txt.gz")))

# %% tags=[]
dim(gwas)

# %% tags=[]
head(gwas)

# %% [markdown] tags=[]
# ### Extract chromosome

# %% tags=[]
unique(gwas$chromosome)

# %% tags=[]
gwas$chrom <- gsub("chr([0-9]+)", "\\1", gwas$chromosome)
gwas <- transform(gwas, chrom = as.numeric(chrom))

# %% tags=[]
unique(gwas$chrom)

# %% [markdown] tags=[]
# ## Stats

# %% tags=[]
summary(gwas)

# %% [markdown] tags=[]
# ## Remove NA pvalues

# %% tags=[]
dim(gwas)

# %% tags=[]
gwas <- gwas %>% filter(pvalue >= 0 & pvalue <= 1)

# %% tags=[]
dim(gwas)

# %% [markdown] tags=[]
# ## Manhattan plot

# %% tags=[]
options(repr.plot.width = 20, repr.plot.height = 10)

manhattan(
  gwas,
  chr = "chrom",
  bp = "position",
  p = "pvalue",
  snp = "variant_id",
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
  gwas$pvalue,
  main = paste0("Q-Q plot - ", gwas_title)
)

# %% tags=[]
