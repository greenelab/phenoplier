# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
#     formats: ipynb,py//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] tags=[]
# # Description

# %% [markdown] tags=[]
# This notebook takes all results per cohort (PhenomeXcan and eMERGE), removes repeated runs, and adjust their pvalues.

# %% [markdown] tags=[]
# # Environment variables

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
import conf

# %% tags=[]
N_JOBS = conf.GENERAL["N_JOBS"]
display(N_JOBS)

# %% tags=[]
# %env MKL_NUM_THREADS=$N_JOBS
# %env OPEN_BLAS_NUM_THREADS=$N_JOBS
# %env NUMEXPR_NUM_THREADS=$N_JOBS
# %env OMP_NUM_THREADS=$N_JOBS

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
from pathlib import Path

import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from tqdm import tqdm

from gls import GLSPhenoplier

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
OUTPUT_DIR = conf.RESULTS["GLS"]
display(OUTPUT_DIR)

assert OUTPUT_DIR.exists()

# %% [markdown] tags=[]
# # Get results files

# %% tags=[]
INPUT_FILES_PER_COHORT = {
    "phenomexcan": list(OUTPUT_DIR.glob("gls_phenotypes*phenomexcan*.pkl")),
    "emerge": list(OUTPUT_DIR.glob("gls_phenotypes*emerge*.pkl")),
}

# %% tags=[]
display(INPUT_FILES_PER_COHORT)

# %% [markdown] tags=[]
# # Combine by cohort

# %% tags=[]
for cohort, result_files in INPUT_FILES_PER_COHORT.items():
    display(cohort)

    dfs = []
    for res in result_files:
        dfs.append(pd.read_pickle(res))

    dfs = pd.concat(dfs, ignore_index=True)

    # remove duplicate runs
    dfs = dfs.drop_duplicates(subset=["phenotype", "lv"])

    # adjust pvalues
    adj_pval = multipletests(dfs["pvalue"], alpha=0.05, method="fdr_bh")
    dfs = dfs.assign(fdr=adj_pval[1])

    # drop unneeded columns
    dfs = dfs.drop(columns=["lv_set", "lv_with_pathway", "summary"])

    output_file = OUTPUT_DIR / f"gls_phenotypes-combined-{cohort}.pkl"
    display(output_file)

    dfs.to_pickle(output_file)

# %% tags=[]
