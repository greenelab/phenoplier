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
# It plots the PCA computed previously on samples. It checks that samples from the European ancestry group are homogeneous, and writes a file to exclude later those that are not.

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
data <- as.data.frame(read_table(file.path(SUBSETS_DIR, "all_phase3.6.pca_covar.eigenvec"), col_names=FALSE))

# %% tags=[]
dim(data)

# %%
data <- rename(data, FID = X1, IID = X2)

# %% tags=[]
head(data)

# %%
race <- as.data.frame(read_table(file.path(A1000G_GENOTYPES_DIR, "all_phase3.psam")))

# %%
dim(race)

# %%
race <- rename(race, IID = "#IID")

# %%
head(race)

# %%
unique(race$SuperPop)

# %% [markdown] tags=[]
# # Plot

# %%
datafile <- merge(data, race, by=c("IID"))

# %%
dim(datafile)

# %%
datafile <- rename(datafile, PC1 = X3, PC2 = X4)

# %%
head(datafile)

# %%
options(repr.plot.width=12, repr.plot.height=12)

datafile %>%
    ggplot(aes(x=PC1, y=PC2, shape=SuperPop, color=SuperPop)) +
    geom_point(size=2.5) +
    theme(text = element_text(size = 25), legend.key.size = unit(1.5, 'cm'))

# %% [markdown] tags=[]
# # Select

# %% [markdown]
# Here I manually select threshold for the first two principal components to exclude those samples that are further away from the European ancestry clusters.

# %%
options(repr.plot.width=12, repr.plot.height=12)

datafile %>%
    ggplot(aes(x=PC1, y=PC2, shape=SuperPop, color=SuperPop)) +
    geom_point(size=2.5) +
    geom_vline(aes(xintercept = -0.0147)) +
    geom_hline(aes(yintercept = -0.027)) +
    theme(text = element_text(size = 25), legend.key.size = unit(1.5, 'cm'))

# %%
output_file <- datafile %>% filter(PC1 < -0.0147 & PC2 < -0.027) %>% select(FID, IID) %>% rename(`#FID`=FID)

# %%
dim(output_file)

# %%
head(output_file)

# %%
write_delim(output_file, file.path(SUBSETS_DIR, "all_phase3.6.pca.eur"), delim = " ", quote="none")

# %%
