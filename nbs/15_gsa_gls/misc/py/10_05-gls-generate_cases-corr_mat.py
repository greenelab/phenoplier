# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
#     formats: ipynb,py//py:percent
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

# %% [markdown] tags=[]
# # Description

# %% [markdown] tags=[]
# **TODO:** update

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
import sys

import numpy as np
from scipy.spatial.distance import squareform
import pandas as pd
from scipy import stats

# import matplotlib.pyplot as plt
# import seaborn as sns

import conf
import utils
from entity import Gene

# %% [markdown] tags=[]
# # Settings

# %% tags=["parameters"]
# a cohort name (it could be something like UK_BIOBANK, etc)
COHORT_NAME = "1000G_EUR"

# reference panel such as 1000G or GTEX_V8
REFERENCE_PANEL = "1000G"

# predictions models such as MASHR or ELASTIC_NET
EQTL_MODEL = "MASHR"

# %% tags=[]
OUTPUT_DIR_BASE = (
    conf.RESULTS["GLS"]
    / "gene_corrs"
    / "cohorts"
    / COHORT_NAME.lower()
    / REFERENCE_PANEL.lower()
    / EQTL_MODEL.lower()
    / "all_genes"
)
display(OUTPUT_DIR_BASE)
assert OUTPUT_DIR_BASE.exists()

# %%
OUTPUT_DIR = utils.get_git_repository_path() / "tests" / "data" / "gls"
display(OUTPUT_DIR)
assert OUTPUT_DIR.exists()

# %% [markdown] tags=[]
# # Load data

# %% [markdown] tags=[]
# ## Load full correlation matrix

# %% tags=[]
output_file = OUTPUT_DIR_BASE / "gene_corrs-symbols.pkl"

display(output_file)

# %%
full_corr_matrix_gene_symbols = pd.read_pickle(output_file)

# %% tags=[]
full_corr_matrix_gene_symbols.shape

# %% tags=[]
full_corr_matrix_gene_symbols.head()

# %% [markdown] tags=[]
# # Make matrix compatible with GLS

# %%
_eigvals = np.linalg.eigvals(full_corr_matrix_gene_symbols)
display(_eigvals[_eigvals < 0].shape[0])
display(_eigvals[_eigvals < 0])

# %%
try:
    np.linalg.cholesky(full_corr_matrix_gene_symbols)
    print("No need to fix")
except Exception as e:
    print(f"Failed with:\n {str(e)}")

# %% [markdown] tags=[]
# # Save

# %%
orig_corr_mat = full_corr_matrix_gene_symbols

# %%
orig_corr_mat.to_pickle(OUTPUT_DIR / "corr_mat.pkl.xz")

# %% [markdown]
# **IMPORTANT:** if the matrix did not need adjustment, then it's necessary to also update the folder `tests/data/gls/corr_mat_folder/` by copying the files generated using notebook `nbs/15_gsa_gls/18-create_corr_mat_per_lv.ipynb`.

# %%
