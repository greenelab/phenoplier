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
# It takes a GWAS that was imputed and postprocessed (using the PrediXcan scripts here https://github.com/hakyimlab/summary-gwas-imputation) on a random phenotype and verifies that the Manhattan and QQ-plots look fine (without inflation).

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
GWAS_DIR <- file.path(GLS_NULL_SIMS_DIR, "final_imputed_gwas")

# %% tags=[]
GWAS_DIR

# %% [markdown] tags=[]
# # Random pheno 0

# %% [markdown] tags=[]
# ## Load data

# %% tags=[]
gwas <- as.data.frame(read_table(file.path(GWAS_DIR, "random.pheno0.glm-imputed.txt.gz")))

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
# ## Manhattan plot

# %% tags=[]
options(repr.plot.width = 20, repr.plot.height = 10)

manhattan(
  gwas,
  chr = "chrom",
  bp = "position",
  p = "pvalue",
  snp = "variant_id",
  main = "Manhattan plot",
  suggestiveline = F,
  genomewideline = -log10(5e-08),
  cex = 0.6,
  cex.axis = 0.9,
  ylim = c(0, 10),
)

# %% [markdown] tags=[]
# ## QQ-plot

# %% tags=[]
options(repr.plot.width = 10, repr.plot.height = 10)

qq(gwas$pvalue, main = "Q-Q plot of GWAS p-values")

# %% [markdown] tags=[]
# # Random pheno 373

# %% [markdown] tags=[]
# ## Load data

# %% tags=[]
gwas <- as.data.frame(read_table(file.path(GWAS_DIR, "random.pheno373.glm-imputed.txt.gz")))

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
# ## Manhattan plot

# %% tags=[]
options(repr.plot.width = 20, repr.plot.height = 10)

manhattan(
  gwas,
  chr = "chrom",
  bp = "position",
  p = "pvalue",
  snp = "variant_id",
  main = "Manhattan plot",
  suggestiveline = F,
  genomewideline = -log10(5e-08),
  cex = 0.6,
  cex.axis = 0.9,
  ylim = c(0, 10),
)

# %% [markdown] tags=[]
# ## QQ-plot

# %% tags=[]
options(repr.plot.width = 10, repr.plot.height = 10)

qq(gwas$pvalue, main = "Q-Q plot of GWAS p-values")

# %% [markdown] tags=[]
# # Random pheno 622

# %% [markdown] tags=[]
# ## Load data

# %% tags=[]
gwas <- as.data.frame(read_table(file.path(GWAS_DIR, "random.pheno622.glm-imputed.txt.gz")))

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
# ## Manhattan plot

# %% tags=[]
options(repr.plot.width = 20, repr.plot.height = 10)

manhattan(
  gwas,
  chr = "chrom",
  bp = "position",
  p = "pvalue",
  snp = "variant_id",
  main = "Manhattan plot",
  suggestiveline = F,
  genomewideline = -log10(5e-08),
  cex = 0.6,
  cex.axis = 0.9,
  ylim = c(0, 10),
)

# %% [markdown] tags=[]
# ## QQ-plot

# %% tags=[]
options(repr.plot.width = 10, repr.plot.height = 10)

qq(gwas$pvalue, main = "Q-Q plot of GWAS p-values")

# %% [markdown] tags=[]
# # Random pheno 825

# %% [markdown] tags=[]
# ## Load data

# %% tags=[]
gwas <- as.data.frame(read_table(file.path(GWAS_DIR, "random.pheno825.glm-imputed.txt.gz")))

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
# ## Manhattan plot

# %% tags=[]
options(repr.plot.width = 20, repr.plot.height = 10)

manhattan(
  gwas,
  chr = "chrom",
  bp = "position",
  p = "pvalue",
  snp = "variant_id",
  main = "Manhattan plot",
  suggestiveline = F,
  genomewideline = -log10(5e-08),
  cex = 0.6,
  cex.axis = 0.9,
  ylim = c(0, 10),
)

# %% [markdown] tags=[]
# ## QQ-plot

# %% tags=[]
options(repr.plot.width = 10, repr.plot.height = 10)

qq(gwas$pvalue, main = "Q-Q plot of GWAS p-values")

# %% [markdown] tags=[]
# # Random pheno 135

# %% [markdown] tags=[]
# ## Load data

# %% tags=[]
gwas <- as.data.frame(read_table(file.path(GWAS_DIR, "random.pheno135.glm-imputed.txt.gz")))

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
# ## Manhattan plot

# %% tags=[]
options(repr.plot.width = 20, repr.plot.height = 10)

manhattan(
  gwas,
  chr = "chrom",
  bp = "position",
  p = "pvalue",
  snp = "variant_id",
  main = "Manhattan plot",
  suggestiveline = F,
  genomewideline = -log10(5e-08),
  cex = 0.6,
  cex.axis = 0.9,
  ylim = c(0, 10),
)

# %% [markdown] tags=[]
# ## QQ-plot

# %% tags=[]
options(repr.plot.width = 10, repr.plot.height = 10)

qq(gwas$pvalue, main = "Q-Q plot of GWAS p-values")

# %% [markdown] tags=[]
# # Random pheno 412

# %% [markdown] tags=[]
# ## Load data

# %% tags=[]
gwas <- as.data.frame(read_table(file.path(GWAS_DIR, "random.pheno412.glm-imputed.txt.gz")))

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
# ## Manhattan plot

# %% tags=[]
options(repr.plot.width = 20, repr.plot.height = 10)

manhattan(
  gwas,
  chr = "chrom",
  bp = "position",
  p = "pvalue",
  snp = "variant_id",
  main = "Manhattan plot",
  suggestiveline = F,
  genomewideline = -log10(5e-08),
  cex = 0.6,
  cex.axis = 0.9,
  ylim = c(0, 10),
)

# %% [markdown] tags=[]
# ## QQ-plot

# %% tags=[]
options(repr.plot.width = 10, repr.plot.height = 10)

qq(gwas$pvalue, main = "Q-Q plot of GWAS p-values")

# %% tags=[]
