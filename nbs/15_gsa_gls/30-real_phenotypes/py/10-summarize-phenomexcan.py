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
# It summarizes the GLS (Generalized Least Squares) results on PhenomeXcan, adjusting pvalues using FDR, and saving the final results to a pickle file for later use.

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import conf
from phenoplier_plots import qqplot

# %% [markdown] tags=[]
# # Settings

# %%
N_PHENOTYPES = 4091
N_LVS = 987

# %% tags=[]
INPUT_DIR = conf.RESULTS["GLS"]
display(INPUT_DIR)
assert INPUT_DIR.exists()

INPUT_PATTERN = "phenomexcan_*/**/*.tsv.gz"
display(INPUT_PATTERN)

# %%
PVALUE_COLUMN = "pvalue"

# %%
OUTPUT_DIR = conf.RESULTS["GLS"]
display(OUTPUT_DIR)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# # Load data

# %% [markdown]
# ## Phenotype info

# %%
pheno_info = pd.read_csv(conf.PHENOMEXCAN["UNIFIED_PHENO_INFO_FILE"], sep="\t")

# %%
pheno_info.shape

# %%
pheno_info.head()

# %%
assert pheno_info["short_code"].is_unique

# %%
pheno_code_to_desc_map = pheno_info.set_index("short_code")[
    "unique_description"
].to_dict()

# %% [markdown]
# ## GLS results

# %% [markdown]
# ### Get files list

# %%
INPUT_FILES = list(INPUT_DIR.glob(INPUT_PATTERN))
display(INPUT_FILES[:5])

# %%
_tmp = len(INPUT_FILES)
display(_tmp)
assert _tmp == N_PHENOTYPES

# %% [markdown]
# ### Read results

# %%
INPUT_FILES[0].name.split("gls_phenoplier-")[1].split(".tsv.gz")[0]

# %% tags=[]
dfs = [
    pd.read_csv(f, sep="\t", usecols=["lv", "pvalue_onesided"]).assign(
        phenotype=f.name.split("gls_phenoplier-")[1].split(".tsv.gz")[0]
    )
    for f in INPUT_FILES
]

# %% tags=[]
display(len(dfs))
assert len(dfs) == N_PHENOTYPES

# %% tags=[]
dfs = pd.concat(dfs, axis=0, ignore_index=True).rename(
    columns={"pvalue_onesided": "pvalue"}
)

# %%
display(dfs.shape)
assert dfs.shape[0] == N_PHENOTYPES * N_LVS

# %%
# add phenotype description
dfs = dfs.assign(
    phenotype_desc=dfs["phenotype"].apply(lambda x: pheno_code_to_desc_map[x])
)
dfs = dfs[["phenotype", "phenotype_desc", "lv", "pvalue"]]

# %%
dfs.head()

# %%
_tmp = dfs.groupby("phenotype")["lv"].nunique().unique()
assert _tmp.shape[0] == 1
assert _tmp[0] == N_LVS

# %% [markdown]
# ### FDR adjust

# %%
adj_pval = multipletests(dfs[PVALUE_COLUMN], alpha=0.05, method="fdr_bh")
dfs = dfs.assign(fdr=adj_pval[1])

# %%
dfs.shape

# %%
dfs.head()

# %% [markdown]
# # QQ-plot

# %%
with sns.plotting_context("paper", font_scale=1.8), mpl.rc_context(
    {"lines.markersize": 3}
):
    fig, ax = qqplot(dfs["pvalue"])
    ax.set_title(f"PhenomeXcan - {N_PHENOTYPES} traits")

# %% [markdown]
# # Top hits

# %%
with pd.option_context("display.max_columns", None, "display.max_colwidth", None):
    _tmp = dfs.sort_values("fdr").drop(columns="phenotype")
    _tmp = _tmp[_tmp["fdr"] < 0.05]
    display(_tmp.head(50))

# %% [markdown]
# ## LV246

# %%
with pd.option_context("display.max_columns", None, "display.max_colwidth", None):
    _tmp = dfs[dfs["lv"] == "LV246"].sort_values("fdr").drop(columns="phenotype")
    _tmp = _tmp[_tmp["fdr"] < 0.05]
    display(_tmp.head(50))

# %% [markdown]
# # Optimize data types

# %%
dfs.head()

# %%
dfs.memory_usage()

# %%
dfs["phenotype"] = dfs["phenotype"].astype("category")
dfs["phenotype_desc"] = dfs["phenotype_desc"].astype("category")
dfs["lv"] = dfs["lv"].astype("category")

# %%
dfs.head()

# %%
dfs.memory_usage()

# %% [markdown]
# # Save

# %%
output_file = OUTPUT_DIR / "gls-summary-phenomexcan.pkl.gz"
display(output_file)

# %%
dfs.to_pickle(output_file)

# %%
