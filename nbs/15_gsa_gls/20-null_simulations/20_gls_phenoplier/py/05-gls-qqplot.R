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
# It takes GLSPhenoplier results on a random phenotype and verifies that the QQ-plots look fine (without inflation).

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
PHENOPLIER_GLS_DIR <- file.path(GLS_NULL_SIMS_DIR, "phenoplier", "gls")

# %% tags=[]
PHENOPLIER_GLS_DIR

# %% [markdown] tags=[]
# # Random pheno 1

# %% [markdown] tags=[]
# Random phenotype 1 has inflation factor of 1

# %% [markdown] tags=[]
# ## Load data

# %% tags=[]
glsph <- as.data.frame(read_tsv(file.path(PHENOPLIER_GLS_DIR, "random.pheno1-gls_phenoplier.tsv.gz")))

# %% tags=[]
dim(glsph)

# %% tags=[]
head(glsph)

# %% tags=[]
glsph %>%
  filter(pvalue <= 0.05) %>%
  dim_desc()

# %% [markdown] tags=[]
# ## QQ-plot

# %% tags=[]
options(repr.plot.width = 10, repr.plot.height = 10)

qq(glsph$pvalue, main = "Q-Q plot of GLS PhenoPLIER p-values")

# %% [markdown] tags=[]
# # Random pheno 28

# %% [markdown] tags=[]
# Random phenotype 28 has the largest inflation factor in the original GWAS summary stats (although within the acceptable limits).

# %% [markdown] tags=[]
# ## Load data

# %% tags=[]
glsph <- as.data.frame(read_tsv(file.path(PHENOPLIER_GLS_DIR, "random.pheno28-gls_phenoplier.tsv.gz")))

# %% tags=[]
dim(glsph)

# %% tags=[]
head(glsph)

# %% [markdown] tags=[]
# ## QQ-plot

# %% tags=[]
options(repr.plot.width = 10, repr.plot.height = 10)

qq(glsph$pvalue, main = "Q-Q plot of GLS PhenoPLIER p-values")

# %% tags=[]
