# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
#     formats: ipynb,py//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] tags=[]
# # Description

# %% [markdown] tags=[]
# It summarizes the GLS (Generalized Least Squares) results, adjusting pvalues using FDR, and saving the final results to a pickle file for later use.

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

# %% tags=["parameters"]
N_PHENOTYPES = 3
N_LVS = 987

# %% tags=["injected-parameters"]
# Parameters
PHENOPLIER_NOTEBOOK_FILEPATH = (
    "projects/asthma-copd/nbs/30_gls_phenoplier/05-summarize.ipynb"
)


# %% tags=[]
INPUT_DIR = conf.PROJECTS["ASTHMA_COPD"]["RESULTS_DIR"] / "gls_phenoplier" / "gls"
display(INPUT_DIR)
assert INPUT_DIR.exists()

INPUT_PATTERN = "*.tsv.gz"
display(INPUT_PATTERN)

# %% tags=[]
PVALUE_COLUMN = "pvalue"

# %% tags=[]
OUTPUT_DIR = conf.PROJECTS["ASTHMA_COPD"]["RESULTS_DIR"] / "gls_phenoplier"
display(OUTPUT_DIR)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# %% [markdown] tags=[]
# # Load data

# %% [markdown] tags=[]
# ## Phenotype info

# %% tags=[]
pheno_info = pd.read_csv(conf.PROJECTS["ASTHMA_COPD"]["TRAITS_INFO_FILE"])

# %% tags=[]
pheno_info.shape

# %% tags=[]
pheno_info.head()

# %% tags=[]
assert pheno_info["id"].is_unique

# %% tags=[]
# pheno_code_to_desc_map = pheno_info.set_index("short_code")[
#     "unique_description"
# ].to_dict()

# %% [markdown] tags=[]
# ## GLS results

# %% [markdown] tags=[]
# ### Get files list

# %% tags=[]
INPUT_FILES = list(INPUT_DIR.glob(INPUT_PATTERN))
display(INPUT_FILES[:5])

# %% tags=[]
_tmp = len(INPUT_FILES)
display(_tmp)
assert _tmp == N_PHENOTYPES

# %% [markdown] tags=[]
# ### Read results

# %% tags=[]
INPUT_FILES[0].name.split("gls_phenoplier.")[0]

# %% tags=[]
dfs = [
    pd.read_csv(
        f, sep="\t", usecols=["lv", "beta", "beta_se", "pvalue_onesided"]
    ).assign(phenotype=f.name.split("gls_phenoplier.")[0])
    for f in INPUT_FILES
]

# %% tags=[]
display(len(dfs))
assert len(dfs) == N_PHENOTYPES

# %% tags=[]
dfs = pd.concat(dfs, axis=0, ignore_index=True).rename(
    columns={"pvalue_onesided": PVALUE_COLUMN}
)

# %% tags=[]
display(dfs.shape)
assert dfs.shape[0] == N_PHENOTYPES * N_LVS

# %% tags=[]
# # add phenotype description
# dfs = dfs.assign(
#     phenotype_desc=dfs["phenotype"].apply(lambda x: pheno_code_to_desc_map[x])
# )
# dfs = dfs[["phenotype", "phenotype_desc", "lv", "pvalue"]]

# %% tags=[]
dfs.head()

# %% tags=[]
_tmp = dfs.groupby("phenotype")["lv"].nunique().unique()
assert _tmp.shape[0] == 1
assert _tmp[0] == N_LVS

# %% [markdown] tags=[]
# ### FDR adjust

# %% tags=[]
adj_pval = multipletests(dfs[PVALUE_COLUMN], alpha=0.05, method="fdr_bh")
dfs = dfs.assign(fdr=adj_pval[1])

# %% tags=[]
dfs.shape

# %% tags=[]
dfs.head()

# %% [markdown] tags=[]
# # QQ-plot

# %% tags=[]
with sns.plotting_context("paper", font_scale=1.8), mpl.rc_context(
    {"lines.markersize": 3}
):
    fig, ax = qqplot(dfs["pvalue"])
    ax.set_title(f"PhenomeXcan - {N_PHENOTYPES} traits")

# %% [markdown] tags=[]
# # Top hits

# %% tags=[]
with pd.option_context("display.max_columns", None, "display.max_colwidth", None):
    _tmp = dfs.sort_values("fdr")  # .drop(columns="phenotype")
    _tmp = _tmp[_tmp["fdr"] < 0.05]
    display(_tmp.head(50))

# %% [markdown] tags=[]
# # Optimize data types

# %% tags=[]
dfs.head()

# %% tags=[]
dfs.dtypes

# %% tags=[]
dfs.memory_usage()

# %% tags=[]
dfs["phenotype"] = dfs["phenotype"].astype("category")
dfs["lv"] = dfs["lv"].astype("category")

# %% tags=[]
dfs.head()

# %% tags=[]
dfs.memory_usage()

# %% [markdown] tags=[]
# # Save

# %% [markdown] tags=[]
# ## Pickle

# %% tags=[]
output_file = OUTPUT_DIR / "gls-summary.pkl.gz"
display(output_file)

# %% tags=[]
dfs.to_pickle(output_file)

# %% [markdown] tags=[]
# ## Text

# %% tags=[]
output_file = OUTPUT_DIR / "gls-summary.tsv.gz"
display(output_file)

# %% tags=[]
dfs.to_csv(output_file, sep="\t", index=False)

# %% tags=[]
