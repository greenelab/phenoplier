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
# It reads all GWASs in PhenomeXcan and counts how many variants are included in each one.

# %% [markdown]
# # Modules

# %%
from pathlib import Path

import pandas as pd

import conf

# %% [markdown]
# # Settings

# %%
GWAS_PARSING_BASE_DIR = conf.PHENOMEXCAN["BASE_DIR"] / "gwas_parsing"
display(GWAS_PARSING_BASE_DIR)
GWAS_PARSING_BASE_DIR.mkdir(exist_ok=True, parents=True)

# %%
GWAS_PARSING_N_LINES_DIR = GWAS_PARSING_BASE_DIR / "gwas_parsing_n_lines"
display(GWAS_PARSING_N_LINES_DIR)
GWAS_PARSING_N_LINES_DIR.mkdir(exist_ok=True, parents=True)

# %%
GWAS_PARSING_INPUT_DIR = GWAS_PARSING_BASE_DIR / "full"
display(GWAS_PARSING_INPUT_DIR)
assert GWAS_PARSING_INPUT_DIR.exists()

# %% [markdown]
# # Read PhenomeXcan GWAS' number of variants

# %% magic_args="-s \"$GWAS_PARSING_INPUT_DIR\" \"$GWAS_PARSING_N_LINES_DIR\"" language="bash"
# parallel -j3 zcat {} | wc -l > ${2}/{/.} ::: ${1}/*.txt.gz

# %%
files = list(GWAS_PARSING_N_LINES_DIR.glob("*.txt"))

# %%
len(files)

# %%
# read number of lines per GWAS
gwas_n_vars = {}

for f in files:
    with open(f) as fh:
        gwas_n_vars[f.name.split(".txt")[0]] = int(fh.readlines()[0].strip())

# %%
df = pd.DataFrame.from_dict(gwas_n_vars, orient="index").squeeze()

# %%
df.shape

# %%
df.head()

# %% [markdown]
# # Save

# %%
output_file = GWAS_PARSING_BASE_DIR / "gwas_n_variants.pkl"
display(output_file)

# %%
df.to_pickle(output_file)

# %%
