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

# %% [markdown]
# # Description

# %% [markdown]
# It analyzes the heterozygosity of samples.

# %% [markdown]
# # Modules

# %%
library(tidyverse)

# %% [markdown]
# # Paths

# %%
A1000G_GENOTYPES_DIR <- Sys.getenv("PHENOPLIER_A1000G_GENOTYPES_DIR")

# %%
A1000G_GENOTYPES_DIR

# %%
SUBSETS_DIR <- file.path(A1000G_GENOTYPES_DIR, "subsets")

# %%
SUBSETS_DIR

# %% [markdown]
# # Load data

# %%
het <- read.table(file.path(SUBSETS_DIR, "all_phase3.4.indepSNP.R_check.het"), head = TRUE)

# %%
dim(het)

# %%
head(het)

# %%
het$HET_RATE <- (het$"N.NM." - het$"O.HOM.") / het$"N.NM."

# %%
head(het)

# %%
het %>% summarise(mean = mean(HET_RATE), sd = sd(HET_RATE), max = max(HET_RATE), min = min(HET_RATE))

# %%
het_fail <- subset(het, (het$HET_RATE < mean(het$HET_RATE) - 2 * sd(het$HET_RATE)) | (het$HET_RATE > mean(het$HET_RATE) + 2 * sd(het$HET_RATE)))
# %%
het_fail$HET_DST <- (het_fail$HET_RATE - mean(het$HET_RATE)) / sd(het$HET_RATE)
# %%
dim(het_fail)

# %% [markdown]
# There are no samples that deviate too much.

# %%
head(het_fail)

# %%
write.table(het_fail, file.path(SUBSETS_DIR, "all_phase3.4.fail-het-qc.txt"), row.names = FALSE)

# %%
