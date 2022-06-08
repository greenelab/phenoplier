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
# TODO
#
# - rename this file to be 27-
#
# This file is actually the same as `05-gls-qqplot.ipynb` but in python; pick one of them.

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.graphics.gofplots import qqplot_2samples
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
# INPUT_DIR = conf.RESULTS["GLS_NULL_SIMS"] / "phenoplier" / "gls"
INPUT_DIR = (
    conf.RESULTS["GLS_NULL_SIMS"] / "phenoplier" / "gls-gtex-mashr-mean_gene_expr"
)
display(INPUT_DIR)


# %% [markdown] tags=[]
# # Functions

# %%
def show_prop(data, frac=0.05):
    _pvalue_lt_frac = data[data["pvalue"] < frac]
    #     display(_pvalue_lt_frac.head())
    return _pvalue_lt_frac.shape[0] / data.shape[0]


# %%
def get_prop(pvalues, frac=0.05):
    _pvalue_lt_frac = pvalues[pvalues < frac]
    #     display(_pvalue_lt_frac.head())
    return _pvalue_lt_frac.shape[0] / pvalues.shape[0]


# %%
assert get_prop(np.array([0.20, 0.50]), 0.05) == 0.0
assert get_prop(np.array([0.20, 0.50, 0.75, 0.10, 0.04]), 0.05) == 0.2

# %%
assert get_prop(pd.Series(np.array([0.20, 0.50])), 0.05) == 0.0
assert get_prop(pd.Series(np.array([0.20, 0.50, 0.75, 0.10, 0.04])), 0.05) == 0.2

# %% [markdown]
# # Get files list

# %%
INPUT_FILES = list(INPUT_DIR.glob("*.tsv.gz"))
display(INPUT_FILES[:5])

# %% [markdown]
# # Load data

# %% tags=[]
dfs = [
    pd.read_csv(f, sep="\t").assign(phenotype=f.name.split("-")[0]) for f in INPUT_FILES
]

# %% tags=[]
len(dfs)

# %% tags=[]
dfs = pd.concat(dfs, axis=0, ignore_index=True)

# %%
dfs.shape

# %%
dfs.head()

# %% [markdown]
# # Summary

# %%
summary_list = []
for lv, lv_data in dfs.groupby("lv"):
    assert lv_data.shape[0] == 100

    summary_list.append(
        {
            "lv": lv,
            "1": get_prop(lv_data["pvalue"], 0.01),
            "5": get_prop(lv_data["pvalue"], 0.05),
            "10": get_prop(lv_data["pvalue"], 0.10),
        }
    )

summary_df = pd.DataFrame(summary_list)

# %%
summary_df.shape

# %%
summary_df.head()

# %%
summary_df.describe()

# %%
summary_df[summary_df["5"] < 0.08]

# %%
summary_df[summary_df["5"] == 0.30]

# %%
dfs[dfs["lv"] == "LV704"].sort_values("pvalue").head(40)

# %%
summary_df[summary_df["lv"] != "LV704"].quantile(
    [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.50, 0.75, 0.80, 0.85, 0.90, 0.95]
)

# %% [markdown]
# # Some QQ-plots

# %%
results = dfs[dfs["phenotype"] == "random.pheno6"]  # .sample(n=100)

# %%
results.shape

# %%
results.head()

# %% [markdown]
# ## Proportion pvalue < 0.05

# %%
show_prop(results, 0.01)

# %%
show_prop(results, 0.05)

# %%
show_prop(results, 0.10)

# %%
show_prop(results, 0.15)

# %%
show_prop(results, 0.20)

# %% [markdown]
# ## Plot

# %% tags=[]
data = results["pvalue"].to_numpy()
n = data.shape[0]
uniform_data = np.array([i / (n + 1) for i in range(1, n + 1)])

# %% tags=[]
display(data[:5])
display(uniform_data[:5])

# %% tags=[]
observed_data = -np.log10(data)
expected_data = -np.log10(uniform_data)

with sns.plotting_context("paper", font_scale=1.8), mpl.rc_context(
    {"lines.markersize": 3}
):
    fig, ax = plt.subplots(figsize=(8, 8))

    fig = qqplot_2samples(expected_data, observed_data, line="45", ax=ax)

    ax.set_xlim(expected_data.min() - 0.05, expected_data.max() + 0.05)

    ax.set_xlabel("$-\log_{10}$(expected pvalue)")
    ax.set_ylabel("$-\log_{10}$(observed pvalue)")
    ax.set_title("QQ-Plot - Null with MASHR models")

# %%
