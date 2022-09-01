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
# Here I do the same as the previous notebook, but with the GLS model proposed in PhenoPLIER.

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
INPUT_DIR = (
    conf.RESULTS["GLS_NULL_SIMS"]
    / "phenoplier"
    / "1000g_eur"
    / "covars"
    / "_corrs_5mb"
    / "gls-gtex_v8_mashr-sub_corr"
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
def qqplot_unif(results, check_n_lvs=True, other_results=None):
    data = results[PVALUE_COLUMN].to_numpy()
    n = data.shape[0]
    observed_data = -np.log10(data)

    observed_lvs = results["lv"].unique()
    n_observed_lvs = len(observed_lvs)
    if check_n_lvs:
        assert n_observed_lvs == 1
        observed_lv = observed_lvs[0]
    else:
        observed_lv = f"All LVs ({n_observed_lvs})"

    other_lv = ""
    if other_results is not None:
        other_data = other_results[PVALUE_COLUMN].to_numpy()
        expected_data = -np.log10(other_data)

        other_lv = other_results["lv"].unique()
        assert len(other_lv) == 1
        other_lv = other_lv[0]
    else:
        uniform_data = np.array([i / (n + 1) for i in range(1, n + 1)])
        expected_data = -np.log10(uniform_data)

    with sns.plotting_context("paper", font_scale=1.8), mpl.rc_context(
        {"lines.markersize": 3}
    ):
        fig, ax = plt.subplots(figsize=(8, 8))

        fig = qqplot_2samples(expected_data, observed_data, line="45", ax=ax)

        ax.set_xlim(expected_data.min() - 0.05, expected_data.max() + 0.05)

        ax.set_xlabel("$-\log_{10}$" + f"(expected pvalue) - {other_lv}")
        ax.set_ylabel("$-\log_{10}$" + f"(observed pvalue) - {observed_lv}")

        # ax.set_title("OLS (baseline)\nMean type I error: 5.5%")
        ax.set_title("OLS (baseline)")


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
show_prop(dfs)

# %% [markdown]
# # QQ-plot

# %%
qqplot_unif(dfs, check_n_lvs=False)

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
# # Query specific LVs

# %%
summary_df[summary_df["lv"].isin(("LV246", "LV603"))]

# %% [markdown]
# ## LV246

# %%
lv_code = "LV246"

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
# # Comparison with OLS model

# %% [markdown]
# ## LVs with high mean type I error in OLS model

# %% [markdown]
# Here I compare the OLS's high mean type I error LVs with results from the GLS model.

# %% [markdown]
# ### LV234

# %%
lv_code = "LV234"

# %%
results = dfs[dfs["lv"] == lv_code]

# %%
results.shape

# %%
results.head()

# %% [markdown]
# #### Mean type I errors at different thresholds

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
# #### QQplot

# %%
qqplot_unif(results)

# %% [markdown]
# #### Top genes in LV

# %%
lv_genes = get_lv_genes(lv_code)
display(lv_genes.head(25))

# %%
# see bands of top genes
lv_genes.head(68)["gene_band"].value_counts().head(10)

# %% [markdown]
# ### LV847

# %%
lv_code = "LV847"

# %%
results = dfs[dfs["lv"] == lv_code]

# %%
results.shape

# %%
results.head()

# %% [markdown]
# #### Mean type I errors at different thresholds

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
# #### QQplot

# %%
qqplot_unif(results)

# %% [markdown]
# ### LV45

# %%
lv_code = "LV45"

# %%
results = dfs[dfs["lv"] == lv_code]

# %%
results.shape

# %%
results.head()

# %% [markdown]
# #### Mean type I errors at different thresholds

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
# #### QQplot

# %%
qqplot_unif(results)

# %% [markdown]
# ### LV800

# %%
lv_code = "LV800"

# %%
results = dfs[dfs["lv"] == lv_code]

# %%
results.shape

# %%
results.head()

# %% [markdown]
# #### Mean type I errors at different thresholds

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
# #### QQplot

# %%
qqplot_unif(results)

# %% [markdown]
# ### LV914

# %% [markdown]
# This one is not corrected here, I analyze it below.

# %% [markdown]
# ### LV189

# %%
lv_code = "LV189"

# %%
results = dfs[dfs["lv"] == lv_code]

# %%
results.shape

# %%
results.head()

# %% [markdown]
# #### Mean type I errors at different thresholds

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
# #### QQplot

# %%
qqplot_unif(results)

# %% [markdown]
# ## LVs with expected mean type I error

# %% [markdown]
# Here I make sure the well calibrated LVs in the OLS model are still well calibrated here.

# %%
display(lvs_expected_error.sort_values("5").head(20))

# %% [markdown]
# Here I'm manually selecting from this list, since I want those that are well calibrated across different p-value thresholds.

# %% [markdown]
# ### LV924

# %%
lv_code = "LV924"

# %%
results = dfs[dfs["lv"] == lv_code]

# %%
results.shape

# %%
results.head()

# %% [markdown]
# #### Mean type I errors at different thresholds

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
# #### QQplot

# %%
qqplot_unif(results)

# %% [markdown]
# Hm, well calibrated in general, but with one small pvalue.

# %% [markdown]
# ### LV675

# %%
lv_code = "LV675"

# %%
results = dfs[dfs["lv"] == lv_code]

# %%
results.shape

# %%
results.head()

# %% [markdown]
# #### Mean type I errors at different thresholds

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
# #### QQplot

# %%
qqplot_unif(results)

# %% [markdown]
# ### LV691

# %%
lv_code = "LV691"

# %%
results = dfs[dfs["lv"] == lv_code]

# %%
results.shape

# %%
results.head()

# %% [markdown]
# #### Mean type I errors at different thresholds

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
# #### QQplot

# %%
qqplot_unif(results)

# %% [markdown]
# # GLS model - LVs with high mean type I error

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
# ### Top genes in LV

# %%
lv_genes = get_lv_genes(lv_code)
display(lv_genes.head(25))

# %%
# see bands of top genes
lv_genes.head(68)["gene_band"].value_counts().head(10)

# %% [markdown]
# ## LV816

# %%
lv_code = "LV816"

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
# ### Top genes in LV

# %%
lv_genes = get_lv_genes(lv_code)
display(lv_genes.head(25))

# %%
# see bands of top genes
lv_genes.head(68)["gene_band"].value_counts().head(10)

# %% [markdown]
# ## LV588

# %%
lv_code = "LV588"

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
# ### Top genes in LV

# %%
lv_genes = get_lv_genes(lv_code)
display(lv_genes.head(25))

# %%
# see bands of top genes
lv_genes.head(68)["gene_band"].value_counts().head(10)

# %% [markdown]
# # Conclusions

# %% [markdown]
# The GLS model is definitely an improvement over the OLS model:
#
# * It corrects LVs with top genes from the same band, like LV234, LV847, LV45.
# * There is a small improvement on mean type I error (0.0584 vs 0.0557).
#
# However, it does not fix other problematic LVs, like LV914.

# %%
