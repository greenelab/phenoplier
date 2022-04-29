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
# Read the missing info computed previously (by individual and variant) and plots some histograms.

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
library(tidyverse)
library(ggplot2)

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

# %% tags=[]
data <- as.data.frame(read_table(file.path(SUBSETS_DIR, "all_phase3.7.pca_covar.eigenvec"), col_names=FALSE))

# %% tags=[]
dim(data)

# %%
data <- rename(data, FID = X1, IID = X2, PC1 = X3, PC2 = X4)

# %% tags=[]
head(data)

# %% [markdown] tags=[]
# # Plot

# %%
options(repr.plot.width=12, repr.plot.height=12)

data %>%
    ggplot(aes(x=PC1, y=PC2)) +
    geom_point(size=2.5) +
    theme(text = element_text(size = 25), legend.key.size = unit(1.5, 'cm'))

# %%
