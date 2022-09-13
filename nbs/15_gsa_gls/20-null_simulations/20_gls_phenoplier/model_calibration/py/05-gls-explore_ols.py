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
# The idea of this notebook is to explore the results of a simple OLS model (Ordinary Least Squares) to associate an LV (gene weights) with a trait (gene pvalues).
# Since predicted gene expression is correlated, especially among adjacent genes, a simple OLS model is expected to fail by having high type I errors in some LVs.

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.gofplots import qqplot_2samples
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import conf
from data.recount2 import LVAnalysis
from phenoplier_plots import qqplot

# %% [markdown] tags=[]
# # Settings

# %%
N_PHENOTYPES = 1000
N_LVS = 987

# %% tags=[]
INPUT_DIR = (
    conf.RESULTS["GLS_NULL_SIMS"]
    / "phenoplier"
    / "1000g_eur"
    / "covars"
    / "_corrs_all"
    / "gls-debug_use_ols"
)
display(INPUT_DIR)
assert INPUT_DIR.exists()

# %%
PVALUE_COLUMN = "pvalue_onesided"

# %% [markdown]
# # Load MultiXcan genes present in results

# %%
_tmp = pd.read_csv(
    conf.RESULTS["GLS_NULL_SIMS"]
    / "twas"
    / "smultixcan"
    / "random.pheno0-gtex_v8-mashr-smultixcan.txt",
    sep="\t",
)

# %%
_tmp.shape

# %%
_tmp.head()

# %%
multixcan_genes = set(_tmp["gene_name"])
display(len(multixcan_genes))
display(list(multixcan_genes)[:10])

# %% [markdown]
# # Load MultiPLIER Z matrix

# %%
multiplier_z = pd.read_pickle(conf.MULTIPLIER["MODEL_Z_MATRIX_FILE"])

# %%
multiplier_z.shape

# %%
# keep genes only present in MultiXcan
multiplier_z = multiplier_z.loc[
    sorted(multixcan_genes.intersection(multiplier_z.index))
]

# %%
multiplier_z.shape

# %%
multiplier_z.head()


# %% [markdown] tags=[]
# # Functions

# %%
def show_prop(data, frac=0.05):
    pvalues = data[PVALUE_COLUMN]
    return get_prop(pvalues, frac=frac)


# %%
def get_prop(pvalues, frac=0.05):
    _pvalue_lt_frac = pvalues[pvalues < frac]
    return _pvalue_lt_frac.shape[0] / pvalues.shape[0]


# %%
assert get_prop(np.array([0.20, 0.50]), 0.05) == 0.0
assert get_prop(np.array([0.20, 0.50, 0.75, 0.10, 0.04]), 0.05) == 0.2

# %%
assert get_prop(pd.Series(np.array([0.20, 0.50])), 0.05) == 0.0
assert get_prop(pd.Series(np.array([0.20, 0.50, 0.75, 0.10, 0.04])), 0.05) == 0.2


# %%
def get_lv_genes(lv_code):
    lv_genes = multiplier_z[lv_code].sort_values(ascending=False)
    lv_obj = LVAnalysis(lv_code)
    return lv_obj.lv_genes.set_index("gene_name").loc[lv_genes.index]


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
display(len(dfs))
assert len(dfs) == N_PHENOTYPES

# %% tags=[]
dfs = pd.concat(dfs, axis=0, ignore_index=True)

# %%
display(dfs.shape)
assert dfs.shape[0] == N_PHENOTYPES * N_LVS

# %%
dfs.head()

# %%
_tmp = dfs.groupby("phenotype")["lv"].nunique().unique()
assert _tmp.shape[0] == 1
assert _tmp[0] == N_LVS

# %% [markdown]
# # Mean type I error

# %%
_mt1e = show_prop(dfs)
display(_mt1e)

# %% [markdown]
# # QQ-plot

# %%
with sns.plotting_context("paper", font_scale=1.8), mpl.rc_context(
    {"lines.markersize": 3}
):
    fig, ax = qqplot(dfs[PVALUE_COLUMN])
    ax.set_title(
        f"OLS (baseline)\nMean type I error: {_mt1e:.3f}\n{N_LVS} LVs - {N_PHENOTYPES} random phenotypes"
    )

# %% [markdown]
# # Summary of mean type I error per LV

# %%
summary_list = []
for lv, lv_data in dfs.groupby("lv"):
    assert lv_data.shape[0] == N_PHENOTYPES

    summary_list.append(
        {
            "lv": lv,
            "1": get_prop(lv_data[PVALUE_COLUMN], 0.01),
            "5": get_prop(lv_data[PVALUE_COLUMN], 0.05),
            "10": get_prop(lv_data[PVALUE_COLUMN], 0.10),
        }
    )

summary_df = pd.DataFrame(summary_list)
assert summary_df.shape[0] == N_LVS

# %%
summary_df.shape

# %%
summary_df.head()

# %%
summary_df.describe()

# %% [markdown]
# ## LVs with expected type I error

# %%
lvs_expected_error = summary_df[summary_df["5"].between(0.049, 0.051)]
display(lvs_expected_error.shape)
display(lvs_expected_error.sort_values("5").head(20))
display(lvs_expected_error.sort_values("5").tail(20))

# %% [markdown]
# ## LVs with high type I error

# %%
lvs_high_error = summary_df[summary_df["5"] > 0.06]
display(lvs_high_error.shape)
# display(lvs_high_error.sort_values("5").head(20))
display(lvs_high_error.sort_values("5").tail(20))

# %% [markdown]
# Many LVs have a mean type I error greater than expected.
#
# LV234, LV847 and LV45 are among the ones with the largest mean type I error (up to 0.131).
#
# Let's take a look at these.

# %% [markdown]
# # LVs with high mean type I error

# %% [markdown]
# ## LV234

# %%
lv_code = "LV234"

# %%
results = dfs[dfs["lv"] == lv_code]

# %%
results.shape

# %%
results.head()

# %% [markdown]
# ### Mean type I errors at different thresholds

# %%
show_prop(results, 0.01)

# %%
_mt1e = show_prop(results, 0.05)
display(_mt1e)

# %%
show_prop(results, 0.10)

# %%
show_prop(results, 0.15)

# %%
show_prop(results, 0.20)

# %% [markdown]
# ### QQplot

# %%
with sns.plotting_context("paper", font_scale=1.8), mpl.rc_context(
    {"lines.markersize": 3}
):
    fig, ax = qqplot(results[PVALUE_COLUMN])
    ax.set_title(
        f"OLS model - {lv_code}\nMean type I error: {_mt1e:.3f}\n{results.shape[0]} random phenotypes"
    )

# %% [markdown]
# ### Top genes in LV

# %%
lv_genes = get_lv_genes(lv_code)
display(lv_genes.head(25))

# %%
# see bands of top genes
lv_genes.head(25)["gene_band"].value_counts().head(10)

# %% [markdown]
# **Note**: top genes from the same band.

# %%
# take a look at all genes with non-zero weight
lv_genes_nonzero_weight = lv_genes[lv_genes[lv_code] > 0]

# %%
# how many genes have non-zero weight?
lv_genes_nonzero_weight.shape

# %%
# how are bands distributed among zero weighted genes?
lv_genes_nonzero_weight["gene_band"].value_counts().head(10)

# %%
# distribution of amount of bands in LV
g = sns.displot(lv_genes_nonzero_weight["gene_band"].value_counts(), height=7, aspect=2)

# %%
# distribution of weights in LV
g = sns.displot(lv_genes[lv_genes[lv_code] > 0][lv_code], height=7, aspect=2)

# %% [markdown]
# ## LV847

# %%
lv_code = "LV847"

# %%
results = dfs[dfs["lv"] == lv_code]

# %%
results.shape

# %%
results.head()

# %% [markdown]
# ### Mean type I errors at different thresholds

# %%
show_prop(results, 0.01)

# %%
_mt1e = show_prop(results, 0.05)
display(_mt1e)

# %%
show_prop(results, 0.10)

# %%
show_prop(results, 0.15)

# %%
show_prop(results, 0.20)

# %% [markdown]
# ### QQplot

# %%
with sns.plotting_context("paper", font_scale=1.8), mpl.rc_context(
    {"lines.markersize": 3}
):
    fig, ax = qqplot(results[PVALUE_COLUMN])
    ax.set_title(
        f"OLS model - {lv_code}\nMean type I error: {_mt1e:.3f}\n{results.shape[0]} random phenotypes"
    )

# %% [markdown]
# ### Top genes in LV

# %%
lv_genes = get_lv_genes(lv_code)
display(lv_genes.head(25))

# %%
# see bands of top genes
lv_genes.head(25)["gene_band"].value_counts().head(10)

# %% [markdown]
# **Note**: same as LV234

# %%
# take a look at all genes with non-zero weight
lv_genes_nonzero_weight = lv_genes[lv_genes[lv_code] > 0]

# %%
# how many genes have non-zero weight?
lv_genes_nonzero_weight.shape

# %%
# how are bands distributed among zero weighted genes?
lv_genes_nonzero_weight["gene_band"].value_counts().head(10)

# %%
# distribution of amount of bands in LV
g = sns.displot(lv_genes_nonzero_weight["gene_band"].value_counts(), height=7, aspect=2)

# %%
# distribution of weights in LV
g = sns.displot(lv_genes[lv_genes[lv_code] > 0][lv_code], height=7, aspect=2)

# %% [markdown]
# ## LV45

# %%
lv_code = "LV45"

# %%
results = dfs[dfs["lv"] == lv_code]

# %%
results.shape

# %%
results.head()

# %% [markdown]
# ### Mean type I errors at different thresholds

# %%
show_prop(results, 0.01)

# %%
_mt1e = show_prop(results, 0.05)
display(_mt1e)

# %%
show_prop(results, 0.10)

# %%
show_prop(results, 0.15)

# %%
show_prop(results, 0.20)

# %% [markdown]
# ### QQplot

# %%
with sns.plotting_context("paper", font_scale=1.8), mpl.rc_context(
    {"lines.markersize": 3}
):
    fig, ax = qqplot(results[PVALUE_COLUMN])
    ax.set_title(
        f"OLS model - {lv_code}\nMean type I error: {_mt1e:.3f}\n{results.shape[0]} random phenotypes"
    )

# %% [markdown]
# ### Top genes in LV

# %%
lv_genes = get_lv_genes(lv_code)
display(lv_genes.head(25))

# %%
# see bands of top genes
lv_genes.head(25)["gene_band"].value_counts().head(10)

# %% [markdown]
# **Note**: same as LV234, very likely causing high type I errors.

# %%
# take a look at all genes with non-zero weight
lv_genes_nonzero_weight = lv_genes[lv_genes[lv_code] > 0]

# %%
# how many genes have non-zero weight?
lv_genes_nonzero_weight.shape

# %%
# how are bands distributed among nonzero weighted genes?
lv_genes_nonzero_weight["gene_band"].value_counts().head(10)

# %%
# distribution of amount of bands in LV
g = sns.displot(lv_genes_nonzero_weight["gene_band"].value_counts(), height=7, aspect=2)

# %%
# distribution of weights in LV
g = sns.displot(lv_genes[lv_genes[lv_code] > 0][lv_code], height=7, aspect=2)

# %% [markdown]
# ## LV800

# %%
lv_code = "LV800"

# %%
results = dfs[dfs["lv"] == lv_code]

# %%
results.shape

# %%
results.head()

# %% [markdown]
# ### Mean type I errors at different thresholds

# %%
show_prop(results, 0.01)

# %%
_mt1e = show_prop(results, 0.05)
display(_mt1e)

# %%
show_prop(results, 0.10)

# %%
show_prop(results, 0.15)

# %%
show_prop(results, 0.20)

# %% [markdown]
# ### QQplot

# %%
with sns.plotting_context("paper", font_scale=1.8), mpl.rc_context(
    {"lines.markersize": 3}
):
    fig, ax = qqplot(results[PVALUE_COLUMN])
    ax.set_title(
        f"OLS model - {lv_code}\nMean type I error: {_mt1e:.3f}\n{results.shape[0]} random phenotypes"
    )

# %% [markdown]
# ### Top genes in LV

# %%
lv_genes = get_lv_genes(lv_code)
display(lv_genes.head(25))

# %% [markdown]
# **Note**: similar to the first LVs in that top genes are mostly from the same band, but here weights are way smaller.

# %%
# see bands of top genes
lv_genes.head(25)["gene_band"].value_counts().head(10)

# %% [markdown]
# **Note**: same as LV234.

# %%
# take a look at all genes with non-zero weight
lv_genes_nonzero_weight = lv_genes[lv_genes[lv_code] > 0]

# %%
# how many genes have non-zero weight?
lv_genes_nonzero_weight.shape

# %%
# how are bands distributed among zero weighted genes?
lv_genes_nonzero_weight["gene_band"].value_counts().head(10)

# %%
# distribution of amount of bands in LV
g = sns.displot(lv_genes_nonzero_weight["gene_band"].value_counts(), height=7, aspect=2)

# %%
# distribution of weights in LV
g = sns.displot(lv_genes[lv_genes[lv_code] > 0][lv_code], height=7, aspect=2)

# %% [markdown]
# ## LV914

# %%
lv_code = "LV914"

# %%
results = dfs[dfs["lv"] == lv_code]

# %%
results.shape

# %%
results.head()

# %% [markdown]
# ### Mean type I errors at different thresholds

# %%
show_prop(results, 0.01)

# %%
_mt1e = show_prop(results, 0.05)
display(_mt1e)

# %%
show_prop(results, 0.10)

# %%
show_prop(results, 0.15)

# %%
show_prop(results, 0.20)

# %% [markdown]
# ### QQplot

# %%
with sns.plotting_context("paper", font_scale=1.8), mpl.rc_context(
    {"lines.markersize": 3}
):
    fig, ax = qqplot(results[PVALUE_COLUMN])
    ax.set_title(
        f"OLS model - {lv_code}\nMean type I error: {_mt1e:.3f}\n{results.shape[0]} random phenotypes"
    )

# %% [markdown]
# ### Top genes in LV

# %%
lv_genes = get_lv_genes(lv_code)
display(lv_genes.head(25))

# %% [markdown]
# **Note**: Ok, first LV that does not have genes from the same band at the top, but results are still not calibrated.

# %%
# see bands of top genes
lv_genes.head(25)["gene_band"].value_counts().head(10)

# %%
# take a look at all genes with non-zero weight
lv_genes_nonzero_weight = lv_genes[lv_genes[lv_code] > 0]

# %%
# how many genes have non-zero weight?
lv_genes_nonzero_weight.shape

# %%
# how are bands distributed among zero weighted genes?
lv_genes_nonzero_weight["gene_band"].value_counts().head(10)

# %%
# distribution of amount of bands in LV
g = sns.displot(lv_genes_nonzero_weight["gene_band"].value_counts(), height=7, aspect=2)

# %%
# distribution of weights in LV
g = sns.displot(lv_genes[lv_genes[lv_code] > 0][lv_code], height=7, aspect=2)

# %% [markdown]
# ## LV189

# %%
lv_code = "LV189"

# %%
results = dfs[dfs["lv"] == lv_code]

# %%
results.shape

# %%
results.head()

# %% [markdown]
# ### Mean type I errors at different thresholds

# %%
show_prop(results, 0.01)

# %%
_mt1e = show_prop(results, 0.05)
display(_mt1e)

# %%
show_prop(results, 0.10)

# %%
show_prop(results, 0.15)

# %%
show_prop(results, 0.20)

# %% [markdown]
# ### QQplot

# %%
with sns.plotting_context("paper", font_scale=1.8), mpl.rc_context(
    {"lines.markersize": 3}
):
    fig, ax = qqplot(results[PVALUE_COLUMN])
    ax.set_title(
        f"OLS model - {lv_code}\nMean type I error: {_mt1e:.3f}\n{results.shape[0]} random phenotypes"
    )

# %% [markdown]
# ### Top genes in LV

# %%
lv_genes = get_lv_genes(lv_code)
display(lv_genes.head(25))

# %%
# see bands of top genes
lv_genes.head(25)["gene_band"].value_counts().head(10)

# %%
# take a look at all genes with non-zero weight
lv_genes_nonzero_weight = lv_genes[lv_genes[lv_code] > 0]

# %%
# how many genes have non-zero weight?
lv_genes_nonzero_weight.shape

# %%
# how are bands distributed among zero weighted genes?
lv_genes_nonzero_weight["gene_band"].value_counts().head(10)

# %%
# distribution of amount of bands in LV
g = sns.displot(lv_genes_nonzero_weight["gene_band"].value_counts(), height=7, aspect=2)

# %%
# distribution of weights in LV
g = sns.displot(lv_genes[lv_genes[lv_code] > 0][lv_code], height=7, aspect=2)

# %% [markdown]
# # LVs with expected mean type I error

# %%
display(lvs_expected_error.sort_values("5").head(20))

# %% [markdown]
# Here I'm manually selecting from this list, since I want those that are well calibrated across different p-value thresholds.

# %% [markdown]
# ## LV924

# %%
lv_code = "LV924"

# %%
results = dfs[dfs["lv"] == lv_code]

# %%
results.shape

# %%
results.head()

# %% [markdown]
# ### Mean type I errors at different thresholds

# %%
show_prop(results, 0.01)

# %%
_mt1e = show_prop(results, 0.05)
display(_mt1e)

# %%
show_prop(results, 0.10)

# %%
show_prop(results, 0.15)

# %%
show_prop(results, 0.20)

# %% [markdown]
# ### QQplot

# %%
with sns.plotting_context("paper", font_scale=1.8), mpl.rc_context(
    {"lines.markersize": 3}
):
    fig, ax = qqplot(results[PVALUE_COLUMN])
    ax.set_title(
        f"OLS model - {lv_code}\nMean type I error: {_mt1e:.3f}\n{results.shape[0]} random phenotypes"
    )

# %% [markdown]
# Hm, well calibrated in general, but with one small pvalue.

# %% [markdown]
# ### Top genes in LV

# %%
lv_genes = get_lv_genes(lv_code)
display(lv_genes.head(25))

# %%
# see bands of top genes
lv_genes.head(25)["gene_band"].value_counts().head(10)

# %%
# take a look at all genes with non-zero weight
lv_genes_nonzero_weight = lv_genes[lv_genes[lv_code] > 0]

# %%
# how many genes have non-zero weight?
lv_genes_nonzero_weight.shape

# %%
# how are bands distributed among zero weighted genes?
lv_genes_nonzero_weight["gene_band"].value_counts().head(10)

# %% [markdown]
# Similar to LV769.

# %%
# distribution of amount of bands in LV
g = sns.displot(lv_genes_nonzero_weight["gene_band"].value_counts(), height=7, aspect=2)

# %%
# distribution of weights in LV
g = sns.displot(lv_genes[lv_genes[lv_code] > 0][lv_code], height=7, aspect=2)

# %% [markdown]
# ## LV675

# %%
lv_code = "LV675"

# %%
results = dfs[dfs["lv"] == lv_code]

# %%
results.shape

# %%
results.head()

# %% [markdown]
# ### Mean type I errors at different thresholds

# %%
show_prop(results, 0.01)

# %%
_mt1e = show_prop(results, 0.05)
display(_mt1e)

# %%
show_prop(results, 0.10)

# %%
show_prop(results, 0.15)

# %%
show_prop(results, 0.20)

# %% [markdown]
# ### QQplot

# %%
with sns.plotting_context("paper", font_scale=1.8), mpl.rc_context(
    {"lines.markersize": 3}
):
    fig, ax = qqplot(results[PVALUE_COLUMN])
    ax.set_title(
        f"OLS model - {lv_code}\nMean type I error: {_mt1e:.3f}\n{results.shape[0]} random phenotypes"
    )

# %% [markdown]
# Same as before, well calibrated, with one very small pvalue.

# %% [markdown]
# ### Top genes in LV

# %%
lv_genes = get_lv_genes(lv_code)
display(lv_genes.head(25))

# %%
# see bands of top genes
lv_genes.head(25)["gene_band"].value_counts().head(10)

# %%
# take a look at all genes with non-zero weight
lv_genes_nonzero_weight = lv_genes[lv_genes[lv_code] > 0]

# %%
# how many genes have non-zero weight?
lv_genes_nonzero_weight.shape

# %% [markdown]
# Similar to LV769 and LV412

# %%
# how are bands distributed among zero weighted genes?
lv_genes_nonzero_weight["gene_band"].value_counts().head(10)

# %%
# distribution of amount of bands in LV
g = sns.displot(lv_genes_nonzero_weight["gene_band"].value_counts(), height=7, aspect=2)

# %% [markdown]
# The distribution seems similar to LV412, but here we have smaller bands.

# %%
# distribution of weights in LV
g = sns.displot(lv_genes[lv_genes[lv_code] > 0][lv_code], height=7, aspect=2)

# %% [markdown]
# ## LV691

# %%
lv_code = "LV691"

# %%
results = dfs[dfs["lv"] == lv_code]

# %%
results.shape

# %%
results.head()

# %% [markdown]
# ### Mean type I errors at different thresholds

# %%
show_prop(results, 0.01)

# %%
_mt1e = show_prop(results, 0.05)
display(_mt1e)

# %%
show_prop(results, 0.10)

# %%
show_prop(results, 0.15)

# %%
show_prop(results, 0.20)

# %% [markdown]
# ### QQplot

# %%
with sns.plotting_context("paper", font_scale=1.8), mpl.rc_context(
    {"lines.markersize": 3}
):
    fig, ax = qqplot(results[PVALUE_COLUMN])
    ax.set_title(
        f"OLS model - {lv_code}\nMean type I error: {_mt1e:.3f}\n{results.shape[0]} random phenotypes"
    )

# %% [markdown]
# ### Top genes in LV

# %%
lv_genes = get_lv_genes(lv_code)
display(lv_genes.head(25))

# %%
# see bands of top genes
lv_genes.head(25)["gene_band"].value_counts().head(10)

# %%
# take a look at all genes with non-zero weight
lv_genes_nonzero_weight = lv_genes[lv_genes[lv_code] > 0]

# %%
# how many genes have non-zero weight?
lv_genes_nonzero_weight.shape

# %%
# how are bands distributed among zero weighted genes?
lv_genes_nonzero_weight["gene_band"].value_counts().head(10)

# %%
# distribution of amount of bands in LV
g = sns.displot(lv_genes_nonzero_weight["gene_band"].value_counts(), height=7, aspect=2)

# %%
# distribution of weights in LV
g = sns.displot(lv_genes[lv_genes[lv_code] > 0][lv_code], height=7, aspect=2)

# %% [markdown]
# # Conclusions

# %% [markdown]
# Looks like not-well-calibrated LVs are mainly due to too many top genes from the same band.
#
# However, some LVs, like LV914, are not driven by top genes from the same band.

# %%
