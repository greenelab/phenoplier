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
# TODO

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import HTML
from statsmodels.stats.multitest import multipletests

from entity import Trait
from data.cache import read_data
import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
FGSEA_INPUT_FILEPATH = Path(
    conf.RESULTS["CRISPR_ANALYSES"]["BASE_DIR"], "fgsea-hi_conf-all_lvs.tsv"
).resolve()

# %% [markdown] tags=[]
# # Data loading

# %% [markdown] tags=[]
# ## LVs enrichment on DEG from CRISPR screen

# %% tags=[]
deg_enrich = pd.read_csv(
    FGSEA_INPUT_FILEPATH,
    sep="\t",
)

# %% tags=[]
deg_enrich.shape

# %% tags=[]
deg_enrich.head()

# %% tags=[]
deg_enrich = deg_enrich.dropna()

# %% tags=[]
# for each lv/pathway pair we ran fgsea 10 times; here take the maximum pvalue (least significant) among those runs
deg_enrich_max_idx = deg_enrich.groupby(["lv", "pathway"])["pval"].idxmax()

# %% tags=[]
deg_enrich = deg_enrich.loc[deg_enrich_max_idx].reset_index(drop=True)
display(deg_enrich.shape)
display(deg_enrich.head())

# %% [markdown] tags=[]
# ## MultiPLIER summary

# %% tags=[]
# multiplier_model_summary = read_data(conf.MULTIPLIER["MODEL_SUMMARY_FILE"])

# %% tags=[]
# multiplier_model_summary.shape

# %% tags=[]
# multiplier_model_summary.head()

# %% [markdown] tags=[]
# # Adjust p-values

# %% tags=[]
adj_pvals = multipletests(deg_enrich["pval"], alpha=0.05, method="fdr_bh")

# %% tags=[]
adj_pvals

# %% tags=[]
np.sum(adj_pvals[0])

# %% [markdown]
# There are no significant LVs after correcting for multiple-testing

# %% tags=[]
deg_enrich = deg_enrich.assign(fdr=adj_pvals[1])

# %% tags=[]
deg_enrich.head()

# %% [markdown] tags=[]
# # Analysis

# %% tags=[]
df = deg_enrich[(deg_enrich["pval"] < 0.01)].sort_values("pval", ascending=True)

# %% tags=[]
df.shape

# %% tags=[]
df.head()

# %% [markdown] tags=[]
# # Save

# %% tags=[]
# override the original file with adjusted p-values
deg_enrich.to_csv(
    FGSEA_INPUT_FILEPATH,
    sep="\t",
    index=False,
)

# %% tags=[]
