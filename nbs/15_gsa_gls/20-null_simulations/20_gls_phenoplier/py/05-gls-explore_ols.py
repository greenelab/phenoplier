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
# The idea of this notebook is to explore a simple OLS model (Ordinary Least Squares) to associate an LV (gene weights) with a trait (gene z-scores). Since predicted gene expression is correlated, especially among adjacent genes, a simple OLS model is expected to fail by having high type I errors.

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

# %% [markdown] tags=[]
# # Settings

# %%
N_PHENOTYPES = 1000
N_LVS = 987

# %% tags=[]
INPUT_DIR = conf.RESULTS["GLS_NULL_SIMS"] / "phenoplier"
display(INPUT_DIR)

# %% [markdown]
# # Load MultiXcan genes present in results

# %%
_tmp = pd.read_csv(conf.RESULTS["GLS_NULL_SIMS"] / "twas" / "smultixcan" / "random.pheno0-gtex_v8-mashr-smultixcan.txt", sep="\t")

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
multiplier_z = multiplier_z.loc[sorted(multixcan_genes.intersection(multiplier_z.index))]

# %%
multiplier_z.shape

# %%
multiplier_z.head()


# %% [markdown] tags=[]
# # Functions

# %%
def show_prop(data, frac=0.05):
    _pvalue_lt_frac = data[data["pvalue"] < frac]
    return _pvalue_lt_frac.shape[0] / data.shape[0]


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
def qqplot_unif(results):
    data = results["pvalue"].to_numpy()
    n = data.shape[0]
    uniform_data = np.array([i / (n + 1) for i in range(1, n + 1)])
    
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
def get_lv_genes(lv_code):
    lv_genes = multiplier_z[lv_code].sort_values(ascending=False)
    lv_obj = LVAnalysis(lv_code)
    return lv_obj.lv_genes.set_index("gene_name").loc[lv_genes.index]


# %% [markdown]
# # Set input directory

# %%
input_directory = INPUT_DIR / "gls-debug_use_ols"
display(input_directory)

# %% [markdown]
# # Get files list

# %%
INPUT_FILES = list(input_directory.glob("*.tsv.gz"))
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
show_prop(dfs)

# %% [markdown]
# It should be around 0.05. Let's check what happened at individual LVs.

# %% [markdown]
# # Summary of mean type I error per LV

# %%
summary_list = []
for lv, lv_data in dfs.groupby("lv"):
    assert lv_data.shape[0] == N_PHENOTYPES

    summary_list.append(
        {
            "lv": lv,
            "1": get_prop(lv_data["pvalue"], 0.01),
            "5": get_prop(lv_data["pvalue"], 0.05),
            "10": get_prop(lv_data["pvalue"], 0.10),
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
# LV45 has the largest mean type I error (0.158). Let's take a look at these LVs with poor mean type I errors.

# %% [markdown]
# # LVs with high mean type I error

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
show_prop(results, 0.05)

# %%
show_prop(results, 0.10)

# %%
show_prop(results, 0.15)

# %%
show_prop(results, 0.20)

# %% [markdown]
# ### QQplot

# %%
qqplot_unif(results)

# %% [markdown]
# ### Why mean type I errors are so high?

# %%
lv_genes = get_lv_genes(lv_code)
display(lv_genes.head(25))

# %%
lv_genes.head(25)["gene_band"].value_counts().head(10)

# %% [markdown]
# Top genes in this LV are all from the same band, very likely causing high type I errors.

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
show_prop(results, 0.05)

# %%
show_prop(results, 0.10)

# %%
show_prop(results, 0.15)

# %%
show_prop(results, 0.20)

# %% [markdown]
# ### QQplot

# %%
qqplot_unif(results)

# %% [markdown]
# ### Why mean type I errors are so high?

# %%
lv_genes = get_lv_genes(lv_code)
display(lv_genes.head(25))

# %%
lv_genes.head(25)["gene_band"].value_counts().head(10)

# %% [markdown]
# Same as LV45

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
show_prop(results, 0.05)

# %%
show_prop(results, 0.10)

# %%
show_prop(results, 0.15)

# %%
show_prop(results, 0.20)

# %% [markdown]
# ### QQplot

# %%
qqplot_unif(results)

# %% [markdown]
# ### Why mean type I errors are so high?

# %%
lv_genes = get_lv_genes(lv_code)
display(lv_genes.head(25))

# %%
lv_genes.head(25)["gene_band"].value_counts().head(10)

# %% [markdown]
# Same as LV45

# %% [markdown]
# ## LV110

# %%
lv_code = "LV110"

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
show_prop(results, 0.05)

# %%
show_prop(results, 0.10)

# %%
show_prop(results, 0.15)

# %%
show_prop(results, 0.20)

# %% [markdown]
# ### QQplot

# %%
qqplot_unif(results)

# %% [markdown]
# ### Why mean type I errors are so high?

# %%
lv_genes = get_lv_genes(lv_code)
display(lv_genes.head(25))

# %%
lv_genes.head(25)["gene_band"].value_counts().head(10)

# %% [markdown]
# Same as LV45

# %% [markdown]
# ## LV769

# %%
lv_code = "LV769"

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
show_prop(results, 0.05)

# %%
show_prop(results, 0.10)

# %%
show_prop(results, 0.15)

# %%
show_prop(results, 0.20)

# %% [markdown]
# ### QQplot

# %%
qqplot_unif(results)

# %% [markdown]
# ### Why mean type I errors are so high?

# %%
lv_genes = get_lv_genes(lv_code)
display(lv_genes.head(25))

# %% [markdown]
# Ok, first LV that does not have genes from the same band at the top. What's going on here?

# %%
lv_genes.head(25)["gene_band"].value_counts().head(10)

# %% [markdown]
# There are same genes from the same band at the top, but not that many as the other LVs.
# **However**, top genes from chr 6 are here. Those might be highly correlated.
#
# What happens if I take a look at more top genes?

# %%
lv_genes.head(50)["gene_band"].value_counts().head(5)

# %%
lv_genes.head(100)["gene_band"].value_counts().head(15)

# %%
lv_genes.head(150)["gene_band"].value_counts().head(20)

# %%
lv_genes.head(200)["gene_band"].value_counts().head(30)

# %% [markdown]
# ## Distribution of gene weights

# %%
lv_genes[lv_code].describe()

# %%
lv_genes[lv_code].quantile(np.arange(0.80, 1.0, 0.01))

# %%
sns.displot(data=lv_genes, x=lv_code, kind="ecdf")

# %%
g = sns.displot(data=lv_genes, x=lv_code, kind="ecdf")
g.ax.set_xlim((-0.05, 0.50))

# %% [markdown]
# How does band number patterns and distribution of weights in this LV compare with LVs with expected mean type I error?

# %%
lv769_genes = lv_genes

# %% [markdown]
# # LVs with expected mean type I error

# %%
display(lvs_expected_error.sort_values("5").head(20))

# %% [markdown]
# ## LV412

# %%
lv_code = "LV412"

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
show_prop(results, 0.05)

# %%
show_prop(results, 0.10)

# %%
show_prop(results, 0.15)

# %%
show_prop(results, 0.20)

# %% [markdown]
# ### QQplot

# %%
qqplot_unif(results)

# %% [markdown]
# ### Why mean type I errors are expected?

# %%
lv_genes = get_lv_genes(lv_code)
display(lv_genes.head(25))

# %%
lv_genes.head(25)["gene_band"].value_counts().head(10)

# %% [markdown]
# Not different than LV759 at the top 25 genes in terms of numbers of genes from the same band.
# **However**, we don't have genes from the same band in chr 6 here.
#
# Let's keep looking:

# %%
lv_genes.head(50)["gene_band"].value_counts().head(5)

# %% [markdown]
# No difference with LV769

# %%
lv_genes.head(100)["gene_band"].value_counts().head(15)

# %% [markdown]
# No big difference with LV769

# %%
lv_genes.head(150)["gene_band"].value_counts().head(20)

# %% [markdown]
# No big difference with LV769.

# %%
lv_genes.head(200)["gene_band"].value_counts().head(30)

# %% [markdown]
# Here there are even more regions repeated, although we are looking at the top 200 genes now.

# %% [markdown]
# ## Distribution of gene weights

# %%
lv_genes[lv_code].describe()

# %%
lv_genes[lv_code].quantile(np.arange(0.80, 1.0, 0.01))

# %%
sns.displot(data=lv_genes, x=lv_code, kind="ecdf")

# %%
g = sns.displot(data=lv_genes, x=lv_code, kind="ecdf")
g.ax.set_xlim((-0.05, 0.50))

# %%
g = sns.displot(x=lv_genes[lv_code], kind="kde")
g = sns.displot(x=lv769_genes["LV769"], kind="kde", ax=g.ax)
# g.ax.set_xlim((-0.05, 0.50))
# g.ax.set_ylim((-0.05, 0.50))

# %%
g = sns.displot(x=lv_genes[lv_code], y=lv769_genes["LV769"], kind="kde")
g.ax.set_xlim((-0.05, 0.50))
g.ax.set_ylim((-0.05, 0.50))

# %% [markdown]
# DE ESTE GRAFICO parece que ambos LVs abarcan genes totalmente diferentes (los que tienen weight de cero en uno tienen weights altos en el otro).

# %%
POR CADA LV, CONTAR CUANTOS genes tienen weight cero

LUEGO, armar un displot de los genes que tienen valores mayores a cero

una hipotesis es que quizá LV769 (el que da mal) es sparce, y el otro (LV412) es mas dense (tiene valores positivos para mas genes)
