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
# It takes a GWAS on a random phenotype and verifies that the Manhattan and QQ-plots look fine (without inflation).

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
library(tidyverse)

# %%
library(qqman)

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
GLS_NULL_SIMS_DIR <- Sys.getenv("PHENOPLIER_RESULTS_GLS_NULL_SIMS")

# %% tags=[]
GLS_NULL_SIMS_DIR

# %% tags=[]
GWAS_DIR <- file.path(GLS_NULL_SIMS_DIR, "gwas")

# %% tags=[]
GWAS_DIR

# %% [markdown] tags=[]
# # Random pheno 0

# %% [markdown] tags=[]
# ## Load data

# %% tags=[]
gwas <- as.data.frame(read_table(file.path(GWAS_DIR, "random.pheno0.glm.linear")))

# %% tags=[]
dim(gwas)

# %% tags=[]
head(gwas)

# %%
gwas <- gwas %>% filter(P >= 0 & P <= 1)

# %%
dim(gwas)

# %% [markdown]
# ## Manhattan plot

# %%
options(repr.plot.width=20, repr.plot.height=10)

manhattan(
    gwas,
    chr="#CHROM",
    bp="POS",
    p="P",
    snp="ID",
    main = "Manhattan plot: logistic",
    suggestiveline = F,
    genomewideline = -log10(5e-08),
    cex = 0.6,
    cex.axis = 0.9,
    ylim = c(0, 10),
)

# %% [markdown]
# ## QQ-plot

# %%
options(repr.plot.width=10, repr.plot.height=10)

qq(gwas$P, main = "Q-Q plot of GWAS p-values : log")

# %% [markdown] tags=[]
# # Random pheno 28

# %% [markdown] tags=[]
# Random phenotype 28 has the largest inflation factor (although within the acceptable limits).

# %% [markdown] tags=[]
# ## Load data

# %% tags=[]
gwas <- as.data.frame(read_table(file.path(GWAS_DIR, "random.pheno28.glm.linear")))

# %% tags=[]
dim(gwas)

# %% tags=[]
head(gwas)

# %%
gwas <- gwas %>% filter(P >= 0 & P <= 1)

# %%
dim(gwas)

# %% [markdown]
# ## Manhattan plot

# %%
options(repr.plot.width=20, repr.plot.height=10)

manhattan(
    gwas,
    chr="#CHROM",
    bp="POS",
    p="P",
    snp="ID",
    main = "Manhattan plot: logistic",
    suggestiveline = F,
    genomewideline = -log10(5e-08),
    cex = 0.6,
    cex.axis = 0.9,
    ylim = c(0, 10),
)

# %% [markdown]
# ## QQ-plot

# %%
options(repr.plot.width=10, repr.plot.height=10)

qq(gwas$P, main = "Q-Q plot of GWAS p-values : log")

# %%
