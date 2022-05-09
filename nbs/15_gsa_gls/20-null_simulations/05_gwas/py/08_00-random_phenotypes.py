# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Description

# %% [markdown]
# It reads the final set of samples in genotype data and generates random phenotypes for them.

# %% [markdown]
# # Modules

# %%
import pandas as pd
import numpy as np

import conf

# %% [markdown]
# # Settings

# %%
N_PHENOTYPES = 100

# %% [markdown]
# # Paths

# %%
SUBSETS_DIR = conf.A1000G["GENOTYPES_DIR"] / "subsets"

# %%
SUBSET_FILE_PREFIX = "all_phase3.8"

# %% [markdown]
# # Load data

# %%
input_filepath = SUBSETS_DIR / f"{SUBSET_FILE_PREFIX}.fam"
display(input_filepath)

data = pd.read_csv(input_filepath, sep="\s+", header=None)

# %%
data.shape

# %%
data.head()

# %%
data = data.iloc[:, 0:2]

# %%
data.head()

# %%
data = data.rename(columns={0: "FID", 1: "IID"})

# %%
data.head()

# %%
n_samples = data.shape[0]
display(n_samples)

# %% [markdown]
# # Generate random phenotypes

# %%
rs = np.random.RandomState(0)
random_phenos = {}
for i in range(N_PHENOTYPES):
    random_phenos[f"pheno{i}"] = rs.normal(size=n_samples)

# %%
random_data = data.assign(**random_phenos)

# %%
random_data.shape

# %%
random_data.head()

# %%
output_filename = SUBSETS_DIR / f"{SUBSET_FILE_PREFIX}.random_pheno.txt"
display(output_filename)

random_data.to_csv(
    output_filename, sep=" ", index=False, header=True, float_format="%.5f"
)

# %%
