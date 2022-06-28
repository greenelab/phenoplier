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
from scipy import stats
from statsmodels.graphics.gofplots import qqplot_2samples
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import conf

# %% [markdown] tags=[]
# # Settings

# %%
N_PHENOTYPES = 100
N_LVS = 987

# %% tags=[]
# INPUT_DIR = conf.RESULTS["GLS_NULL_SIMS"] / "phenoplier" / "gls"
INPUT_DIR = (
    conf.RESULTS["GLS_NULL_SIMS"] / "phenoplier"  # / "gls-gtex-mashr-mean_gene_expr"
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
# # Standard model (no correlation matrix/identity matrix)

# %%
input_directory = INPUT_DIR / "gls-gtex-mashr-identity"
display(input_directory)

# %% [markdown]
# ## Get files list

# %%
INPUT_FILES = list(input_directory.glob("*.tsv.gz"))
display(INPUT_FILES[:5])

# %% [markdown]
# ## Load data

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
dfs.shape

# %%
dfs.head()

# %%
_tmp = dfs.groupby("phenotype")["lv"].nunique().unique()
assert _tmp.shape[0] == 1
assert _tmp[0] == N_LVS

# %%
show_prop(dfs)

# %% [markdown]
# ## Summary

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

# %%
summary_df[summary_df["5"] > 0.08].sort_values("5")

# %% [markdown]
# ## LV16

# %%
# results = dfs[dfs["phenotype"] == "random.pheno100"]  # .sample(n=100)
results = dfs[dfs["lv"] == "LV16"]  # .sample(n=100)

# %%
results.shape

# %%
results.head()

# %% [markdown]
# ### Proportion pvalue < 0.05

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

# %% [markdown] tags=[]
# ## LV110

# %%
results = dfs[dfs["lv"] == "LV110"]

# %%
results.shape

# %%
results.head()

# %% [markdown]
# ### Proportion pvalue < 0.05

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

# %% [markdown] tags=[]
# ## LV121

# %%
results = dfs[dfs["lv"] == "LV121"]

# %%
results.shape

# %%
results.head()

# %% [markdown]
# ### Proportion pvalue < 0.05

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
# # SSM corr values only

# %%
input_directory = INPUT_DIR / "gls-gtex-mashr-ssm_corrs"
display(input_directory)

# %% [markdown]
# ## Get files list

# %%
INPUT_FILES = list(
    f
    for f in input_directory.glob("*.tsv.gz")
    if int(f.name.split("random.pheno")[1].split("-")[0]) < 100
)
display(INPUT_FILES[:5])

# %% [markdown]
# ## Load data

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
dfs.shape

# %%
dfs.head()

# %%
_tmp = dfs.groupby("phenotype")["lv"].nunique().unique()
assert _tmp.shape[0] == 1
assert _tmp[0] == N_LVS

# %%
show_prop(dfs)

# %% [markdown]
# ## Summary

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

# %%
summary_df[summary_df["5"] > 0.08].sort_values("5")

# %% [markdown]
# ## LV16

# %%
summary_df[summary_df["lv"] == "LV16"]

# %%
# results = dfs[dfs["phenotype"] == "random.pheno100"]  # .sample(n=100)
results = dfs[dfs["lv"] == "LV389"]  # .sample(n=100)

# %%
results.shape

# %%
results.head()

# %% [markdown]
# ### Proportion pvalue < 0.05

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

# %% [markdown]
# # MAGMA-derived correaltions between two genes

# %% [markdown]
# # Mean correlation between principal components

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
# # See what's going on with LV704?

# %%
multiplier_z = pd.read_pickle(conf.MULTIPLIER["MODEL_Z_MATRIX_FILE"])

# %%
multiplier_z.shape

# %%
multiplier_z.head()

# %%
multiplier_z["LV704"].sort_values(ascending=False).head(20)

# %%
# LV1 is an example of a good "LV" with good type I error rates
multiplier_z["LV1"].sort_values(ascending=False).head(20)

# %% tags=[]
lv_x = "LV1"
lv_y = "LV704"

observed_data = multiplier_z[lv_y].to_numpy()
expected_data = multiplier_z[lv_x].to_numpy()

with sns.plotting_context("paper", font_scale=1.8), mpl.rc_context(
    {"lines.markersize": 3}
):
    fig, ax = plt.subplots(figsize=(8, 8))

    fig = qqplot_2samples(expected_data, observed_data, line="45", ax=ax)

    ax.set_xlim(expected_data.min() - 0.05, expected_data.max() + 0.05)

    ax.set_xlabel(lv_x)
    ax.set_ylabel(lv_y)
    ax.set_title("QQ-Plot")

# %% [markdown]
# For LV704, most of the gene values are zero, and just a few have higher values.
#
# The same happens with LV100 (below)

# %% tags=[]
# two good LVs
lv_x = "LV1"
lv_y = "LV100"

observed_data = multiplier_z[lv_y].to_numpy()
expected_data = multiplier_z[lv_x].to_numpy()

with sns.plotting_context("paper", font_scale=1.8), mpl.rc_context(
    {"lines.markersize": 3}
):
    fig, ax = plt.subplots(figsize=(8, 8))

    fig = qqplot_2samples(expected_data, observed_data, line="45", ax=ax)

    ax.set_xlim(expected_data.min() - 0.05, expected_data.max() + 0.05)

    ax.set_xlabel(lv_x)
    ax.set_ylabel(lv_y)
    ax.set_title("QQ-Plot")

# %% [markdown]
# But LV100 has good type I error rates, compared with LV704. So what's the difference? **Maybe** the top genes in LV704 are more correlated than in LV100

# %% [markdown]
# ## Let's see the band of genes in these two LVs

# %%
from data.recount2 import LVAnalysis

# %% [markdown]
# ### LV704

# %%
lv704 = LVAnalysis("LV704")

# %%
lv704.lv_genes.head()

# %%
_tmp = (
    lv704.lv_genes.head(50)
    .groupby("gene_band")
    .count()
    .sort_values("gene_name", ascending=False)
)
display(_tmp[_tmp["gene_name"] > 1])

# %% [markdown]
# ### LV1

# %%
lv1 = LVAnalysis("LV1")

# %%
lv1.lv_genes.head()

# %%
_tmp = (
    lv1.lv_genes.head(50)
    .groupby("gene_band")
    .count()
    .sort_values("gene_name", ascending=False)
)
display(_tmp[_tmp["gene_name"] > 1])

# %% [markdown]
# Hm, this is not what I expected.
#
# Let's see the QQ plot of LV704.

# %% [markdown]
# # QQ-plot of LV704

# %%
# results = dfs[dfs["phenotype"] == "random.pheno100"]  # .sample(n=100)
results = dfs[dfs["lv"] == "LV704"]  # .sample(n=100)

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
results.sort_values("pvalue").head(50)


# %% [markdown]
# # Load gene correlations

# %%
def get_upper_triag(similarity_matrix: pd.DataFrame, k: int = 1):
    """
    It returns the upper triangular matrix of a dataframe representing a
    similarity matrix between n elements.
    Args:
        similarity_matrix: a squared dataframe with a pairwise similarity
          matrix. That means the matrix is equal to its transposed version.
        k: argument given to numpy.triu function. It indicates the that the
          elements of the k-th diagonal to be zeroed.
    Returns:
        A dataframe with non-selected elements as NaNs.
    """
    mask = np.triu(np.ones(similarity_matrix.shape), k=k).astype(bool)
    return similarity_matrix.where(mask)


# %%
OUT_GENE_CORRS_DIR = Path(conf.DATA_DIR, "tmp", "gene_corrs").resolve()
display(OUT_GENE_CORRS_DIR)
OUT_GENE_CORRS_DIR.mkdir(exist_ok=True, parents=True)

# %%
gene_corrs_file = (
    conf.PHENOMEXCAN["LD_BLOCKS"]["GENE_CORRS_DIR"]
    / "gtex_v8/mashr/multiplier_genes-pred_expression_corr_avg-mean-gene_symbols.pkl"
)

# %%
gene_corrs = pd.read_pickle(gene_corrs_file)

# %%
gene_corrs.shape

# %%
gene_corrs.head()

# %%
get_upper_triag(gene_corrs).unstack().dropna().describe().apply(str)

# %% [markdown]
# ## Create positive only

# %%
gene_corrs_positive_only = OUT_GENE_CORRS_DIR / "gene_corrs-positive_only.pkl"

# %%
gene_corrs.min().min()

# %%
_tmp = gene_corrs.copy()
_tmp[_tmp < 0] = 0.0

# %%
get_upper_triag(_tmp).unstack().dropna().describe().apply(str)

# %%
_tmp.to_pickle(gene_corrs_positive_only)

# %% [markdown]
# ## Create r**2 matrix

# %%
gene_corrs_r2 = OUT_GENE_CORRS_DIR / "gene_corrs-r2.pkl"

# %%
gene_corrs.min().min()

# %%
_tmp = np.power(gene_corrs.copy(), 2)

# %%
_tmp.min().min()

# %%
get_upper_triag(_tmp).unstack().dropna().describe().apply(str)

# %%
_tmp.to_pickle(gene_corrs_r2)

# %% [markdown]
# ## Create default gene correlations matrix

# %%
default_gene_corrs_file = OUT_GENE_CORRS_DIR / "gene_corrs-default_identity_matrix.pkl"

# %%
gene_corrs_default = pd.DataFrame(
    data=np.identity(gene_corrs.shape[0]),
    index=gene_corrs.index.copy(),
    columns=gene_corrs.columns.copy(),
)

# %%
gene_corrs_default.shape

# %%
gene_corrs_default.head()

# %%
gene_corrs_default.to_pickle(default_gene_corrs_file)

# %% [markdown]
# # Load random.pheno255

# %%
# let's take a look at the MultiXcan results in random.pheno255
random_pheno_255_mx = pd.read_csv(
    conf.RESULTS["GLS_NULL_SIMS"]
    / "twas"
    / "smultixcan"
    / "random.pheno255-gtex_v8-mashr-smultixcan.txt",
    sep="\t",
)

# %%
random_pheno_255_mx.shape

# %%
random_pheno_255_mx.head()

# %%
data = random_pheno_255_mx[["gene_name", "pvalue"]].set_index("gene_name").squeeze()

# %%
data.shape

# %%
data = data.loc[~data.index.duplicated(keep="first")]

# %%
data.shape

# %%
data = data.dropna()

# %%
data.shape

# %%
common_genes = data.index.intersection(multiplier_z.index).intersection(
    gene_corrs.index
)

# %%
len(common_genes)

# %%
y = data.loc[common_genes]

# %%
y.shape

# %%
y

# %% [markdown]
# QQ plot of MultiXcan results with genes only in MultiPLIER models...

# %% tags=[]
data = y.to_numpy()
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

# %% [markdown]
# Looks good

# %%
# convert p-values to z-scores
y = pd.Series(data=np.abs(stats.norm.ppf(y / 2)), index=y.index.copy())

# %%
y.shape

# %%
y.dropna().shape

# %%
y.head()

# %%
sns.displot(x=multiplier_z.loc[common_genes, "LV704"], y=y.loc[common_genes])

# %%
stats.pearsonr(multiplier_z.loc[common_genes, "LV704"], y)

# %%
sns.displot(x=multiplier_z.loc[common_genes, "LV1"], y=y.loc[common_genes])

# %%
stats.pearsonr(multiplier_z.loc[common_genes, "LV1"], y)

# %%
multiplier_z.loc[common_genes, "LV704"]

# %%
from gls import GLSPhenoplier

# %%
model = GLSPhenoplier(
    gene_corrs_file_path=gene_corrs_file,
)

model.fit_named("LV704", y)
res = model.results

print(res.summary())

# %%
model = GLSPhenoplier(
    gene_corrs_file_path=default_gene_corrs_file,
)

model.fit_named("LV704", y)
res = model.results

print(res.summary())

# %% [markdown]
# **Conclusion**: If I compare the GLS model with a standard one (sigma is just the identity matrix), things do not seem to improve, actually it gets worse.

# %% [markdown]
# Try to

# %% [markdown]
# # Try with positive-only gene corr matrix

# %%
model = GLSPhenoplier(
    gene_corrs_file_path=gene_corrs_positive_only,
)

model.fit_named("LV704", y)
res = model.results

print(res.summary())

# %% [markdown]
# # Try with r2 gene corr matrix

# %%
model = GLSPhenoplier(
    gene_corrs_file_path=gene_corrs_r2,
)

model.fit_named("LV704", y)
res = model.results

print(res.summary())

# %%

# %%
from entity import Gene

# %%
gene_objs = [Gene(name=g) for g in y.index]

# %%
gene_objs[0]

# %%
