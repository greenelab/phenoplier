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
# This notebook analyses the "high polygenicity" problem in our TWAS results from PhenomeXcan and explores ways to fix it. That is, some traits such as "height", are highly polygenic (many genes are correlated with it), and this is a problem for our downstream analysis where these traits (anthropometric ones in particular) are everywhere.
#
# Some approaches to try to fix this issue are explored, such as standardizing by GWAS sample size.

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
from pathlib import Path

from IPython.display import display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import conf
from entity import Gene, Trait
from data.cache import read_data
from multiplier import MultiplierProjection

# %% [markdown] tags=[]
# # Settings

# %% [markdown] tags=[]
# # Load PhenomeXcan data (S-MultiXcan)

# %% tags=[]
smultixcan_results_filename = conf.PHENOMEXCAN[
    "SMULTIXCAN_EFO_PARTIAL_MASHR_ZSCORES_FILE"
]

display(smultixcan_results_filename)

# %% tags=[]
results_filename_stem = smultixcan_results_filename.stem
display(results_filename_stem)

# %% tags=[]
smultixcan_results = pd.read_pickle(smultixcan_results_filename)

# %% tags=[]
smultixcan_results.shape

# %% tags=[]
smultixcan_results.head()

# %% tags=[]
pd.Series(smultixcan_results.values.flatten()).describe().apply(str)

# %% [markdown] tags=[]
# ## Convert gene IDs to Gene names

# %% tags=[]
smultixcan_results = smultixcan_results.rename(index=Gene.GENE_ID_TO_NAME_MAP)

# %% tags=[]
smultixcan_results.shape

# %% tags=[]
smultixcan_results.head()

# %% [markdown] tags=[]
# ## Remove duplicated gene entries

# %% tags=[]
smultixcan_results.index[smultixcan_results.index.duplicated(keep="first")]

# %% tags=[]
smultixcan_results = smultixcan_results.loc[
    ~smultixcan_results.index.duplicated(keep="first")
]

# %% tags=[]
smultixcan_results.shape

# %% [markdown] tags=[]
# ## Some checks

# %% tags=[]
# the data should have no NaN values
assert smultixcan_results.shape == smultixcan_results.dropna(how="any").shape

# %% [markdown] tags=[]
# # Project original S-MultiXcan into MultiPLIER latent space

# %% tags=[]
mproj = MultiplierProjection()

# %% tags=[]
smultixcan_into_multiplier = mproj.transform(smultixcan_results)

# %% tags=[]
smultixcan_into_multiplier.shape

# %% tags=[]
smultixcan_into_multiplier.head()

# %% [markdown] tags=[]
# ## Quick analysis

# %% [markdown] tags=[]
# List top traits associated to some LVs when the original data is projected.

# %% tags=[]
smultixcan_into_multiplier.loc["LV603"].sort_values(ascending=False).head(20)

# %% tags=[]
smultixcan_into_multiplier.loc["LV136"].sort_values(ascending=False).head(20)

# %% tags=[]
smultixcan_into_multiplier.loc["LV844"].sort_values(ascending=False).head(20)

# %% [markdown] tags=[]
# # Select traits

# %% [markdown] tags=[]
# ## Get sample size for all traits

# %% [markdown] tags=[]
# Since some PhenomeXcan traits were previously combined into EFO terms, here I need to compute the sample size for meta-analyzed traits (cases were several PhenomeXcan's traits mapped to the same EFO term).

# %% tags=[]
df_dicts = []

for trait_name in smultixcan_results.columns:
    if Trait.is_efo_label(trait_name):
        # if it is an EFO label, then sum sample sizes of mapped PhenomeXcan
        # traits.
        traits_list = Trait.get_traits_from_efo(trait_name)

        n_list = [t.n for t in traits_list]
        n_cases_list = [t.n_cases for t in traits_list]
        n_controls_list = [t.n_controls for t in traits_list]

        trait_info = {
            "trait": trait_name,
            "n": sum(n_list),
            "n_cases": sum(n_cases_list),
            "n_controls": sum(n_controls_list),
        }
    else:
        t = Trait.get_trait(full_code=trait_name)
        trait_info = {
            "trait": trait_name,
            "n": t.n,
            "n_cases": t.n_cases,
            "n_controls": t.n_controls,
        }

    df_dicts.append(trait_info)

# %% tags=[]
_tmp = len(df_dicts)
display(_tmp)
assert _tmp == smultixcan_results.shape[1]

# %% tags=[]
traits_sample_size_df = pd.DataFrame(df_dicts).set_index("trait")

# %% tags=[]
assert traits_sample_size_df.index.is_unique

# %% tags=[]
traits_sample_size_df.shape

# %% tags=[]
traits_sample_size_df.head()

# %% tags=[]
# some testing
_tmp = traits_sample_size_df.loc["100001_raw-Food_weight"]
assert _tmp.n == 51453
assert pd.isnull(_tmp.n_cases)
assert pd.isnull(_tmp.n_controls)

_tmp = traits_sample_size_df.loc["estrogen-receptor negative breast cancer"]
assert _tmp.n == 120000
assert _tmp.n_cases == 31000
assert _tmp.n_controls == 89000

_tmp = traits_sample_size_df.loc["eosinophil count"]
assert _tmp.n == 173480
assert pd.isnull(_tmp.n_cases)
assert pd.isnull(_tmp.n_controls)

_tmp = traits_sample_size_df.loc["injury"]
assert _tmp.n == 3250693
assert _tmp.n_cases == 4939
assert _tmp.n_controls == 3245754

_tmp = traits_sample_size_df.loc["asthma"]
assert _tmp.n == 814122
assert _tmp.n_cases == 55344
assert _tmp.n_controls == 758778

# %% [markdown] tags=[]
# ## Select representative traits

# %% [markdown] tags=[]
# Here I select some representative traits for downstream analyses in this notebook. The idea is to pick some traits with different sample sizes, and then see what happens before and after a standardization approach.

# %% tags=[]
traits_sample_size_df.sort_values("n").dropna()

# %% tags=[]
traits_sample_size_df.sort_values("n_cases").dropna()

# %% tags=[]
traits_list = [
    "body height",  # continuous, large n
    "asthma",  # binary, large n and n_cases (meta-analyzed)
    "22174-Recent_medication_for_bronchiectasis",  # lowest n
    "abnormal foot morphology",  # lowest n_cases
]


# %% [markdown] tags=[]
# # Functions

# %% tags=[]
def get_n(trait_name, simple=True):
    """
    Given a trait name, it returns either the `sqrt(n)`, that is the total sample size (when simple=True),
    or `(t.n_cases / t.n_controls) * np.sqrt(t.n)` (when simple=False).
    """
    t = traits_sample_size_df.loc[trait_name]
    if simple:
        return np.sqrt(t.n)

    if not pd.isnull(t.n_cases) and not pd.isnull(t.n_controls):
        return (t.n_cases / t.n_controls) * np.sqrt(t.n)
    else:
        return np.sqrt(t.n)


# %% tags=[]
def show_hist(trait_names, data):
    """
    Shows a density plot (KDE) for a trait given a data version.
    """
    _df = (
        data[trait_names]
        .stack()
        .reset_index()
        .rename(columns={"level_1": "trait", 0: "z-score"})
    )
    return sns.displot(_df, x="z-score", hue="trait", rug=False, kind="kde")


# %% [markdown] tags=[]
# # Approach \#1: Standardize by GWAS sample size

# %% [markdown] tags=[]
# For each trait (column), divide by $sqrt(n)$, where $n$ is the sample size in the GWAS for the trait.

# %% [markdown] tags=[]
# ## Standardize

# %% tags=[]
_tmp = smultixcan_results.apply(lambda x: x / get_n(x.name, simple=True))

# %% tags=[]
_tmp.shape

# %% tags=[]
assert _tmp.shape == smultixcan_results.shape

# %% tags=[]
_tmp.head()

# %% tags=[]
# some testing
_trait = "100001_raw-Food_weight"
_gene = "DPM1"
assert _tmp.loc[_gene, _trait] == smultixcan_results.loc[_gene, _trait] / np.sqrt(51453)

_trait = "asthma"
_gene = "C1orf112"
assert _tmp.loc[_gene, _trait] == smultixcan_results.loc[_gene, _trait] / np.sqrt(
    814122
)

# %% tags=[]
smultixcan_results_std = _tmp

# %% [markdown] tags=[]
# ## Analyze distributions

# %% tags=[]
fg = show_hist(traits_list, data=smultixcan_results)
ax = fg.axes[0, 0]
ax.set_xlim(-1, 10)
ax.set_title("Original distribution")

# %% [markdown] tags=[]
# In the original distribution, `body height` has a very long tail, and it has larger values on average than the rest.
# `asthma` is behind `body height`, and the other two traits (with very low sample size) are, as expected, closer to zero.

# %% tags=[]
fg = show_hist(traits_list, data=smultixcan_results_std)
ax = fg.axes[0, 0]
ax.set_xlim(-0.001, 0.025)
ax.set_title("Standardized by $sqrt(n)$")

# %% [markdown] tags=[]
# This standardization approach make `asthma` closer to one of the low-sample-sized traits, height is still "highly polygenic", and the weirdest thing is that very low sample size trait `recent medication...` has now very large values compared with the others.
#
# This won't work.

# %% [markdown] tags=[]
# ## Projection

# %% [markdown] tags=[]
# Here I use the standardized data and project it into the latent space, and I see what happens with well-known LVs: LV603 (related to neutrophils and white blood cells) and LV136 (cardiovascular traits and keratometry measurements).

# %% tags=[]
mproj = MultiplierProjection()

# %% tags=[]
smultixcan_into_multiplier = mproj.transform(smultixcan_results_std)

# %% tags=[]
smultixcan_into_multiplier.shape

# %% tags=[]
smultixcan_into_multiplier.head()

# %% tags=[]
_tmp = smultixcan_into_multiplier.loc["LV603"].sort_values(ascending=False).head(20)
display(_tmp)

# %% tags=[]
traits_sample_size_df.loc[_tmp.index[0]]

# %% tags=[]
_tmp = smultixcan_into_multiplier.loc["LV136"].sort_values(ascending=False).head(20)
display(_tmp)

# %% tags=[]
traits_sample_size_df.loc[_tmp.index[0]]

# %% [markdown] tags=[]
# **Conclusion:** as anticipated, this standardization approach breaks expected relationships, and makes low sample size traits to be at the top, presenting random associations.

# %% [markdown] tags=[]
# # Approach \#2: Standardize by GWAS sample size considering n_cases and n_controls

# %% [markdown] tags=[]
# Here I take into account the ratio `n_cases / n_controls` (in binary traits) in the standardization, by doing: `(n_cases / n_controls) * sqrt(n)`

# %% [markdown] tags=[]
# ## Standardize

# %% tags=[]
_tmp = smultixcan_results.apply(lambda x: x / get_n(x.name, simple=False))

# %% tags=[]
_tmp.shape

# %% tags=[]
assert _tmp.shape == smultixcan_results.shape

# %% tags=[]
_tmp.head()

# %% tags=[]
# some testing
_trait = "100001_raw-Food_weight"
_gene = "DPM1"
assert _tmp.loc[_gene, _trait] == smultixcan_results.loc[_gene, _trait] / np.sqrt(51453)

_trait = "estrogen-receptor negative breast cancer"
_gene = "CFH"
assert _tmp.loc[_gene, _trait] == smultixcan_results.loc[_gene, _trait] / (
    (31000 / 89000) * np.sqrt(120000)
)

_trait = "asthma"
_gene = "C1orf112"
assert _tmp.loc[_gene, _trait] == smultixcan_results.loc[_gene, _trait] / (
    (55344 / 758778) * np.sqrt(814122)
)

# %% tags=[]
smultixcan_results_std = _tmp

# %% [markdown] tags=[]
# ## Analyze distributions

# %% tags=[]
fg = show_hist(traits_list, data=smultixcan_results)
ax = fg.axes[0, 0]
ax.set_xlim(-1, 10)
ax.set_title("Original distribution")

# %% tags=[]
fg = show_hist(traits_list, data=smultixcan_results_std)
ax = fg.axes[0, 0]
ax.set_xlim(-0.01, 0.10)
ax.set_ylim(0, 9)
ax.set_title("Standardized by $(n\_cases / n\_controls) * sqrt(n)$")

# %% [markdown] tags=[]
# Here `asthma` is now more to the right compared with `height`, but the low sample size traits have now too large values.

# %% [markdown] tags=[]
# ## Projection

# %% tags=[]
mproj = MultiplierProjection()

# %% tags=[]
smultixcan_into_multiplier = mproj.transform(smultixcan_results_std)

# %% tags=[]
smultixcan_into_multiplier.shape

# %% tags=[]
smultixcan_into_multiplier.head()

# %% tags=[]
_tmp = smultixcan_into_multiplier.loc["LV603"].sort_values(ascending=False).head(20)
display(_tmp)

# %% tags=[]
traits_sample_size_df.loc[_tmp.index[0]]

# %% tags=[]
_tmp = smultixcan_into_multiplier.loc["LV136"].sort_values(ascending=False).head(20)
display(_tmp)

# %% tags=[]
traits_sample_size_df.loc[_tmp.index[0]]

# %% [markdown] tags=[]
# **Conclusion:** this approach doesn't work either.

# %% [markdown] tags=[]
# # Approach \#3: Standardize by sum of z-scores

# %% [markdown] tags=[]
# Here I divide all z-scores for a trait by `sum(z)` (sum of z-scores). The idea is that highly polygenic traits would be more penalized, which is what we want.

# %% [markdown] tags=[]
# ## Standardize

# %% tags=[]
_tmp = smultixcan_results.apply(lambda x: x / x.sum())

# %% tags=[]
_tmp.shape

# %% tags=[]
assert _tmp.shape == smultixcan_results.shape

# %% tags=[]
_tmp.head()

# %% tags=[]
# some testing
_trait = "body height"
_gene = "SCYL3"
assert (
    _tmp.loc[_gene, _trait]
    == smultixcan_results.loc[_gene, _trait] / smultixcan_results[_trait].sum()
)

# %% tags=[]
smultixcan_results_std = _tmp

# %% [markdown] tags=[]
# ## Analyze distributions

# %% tags=[]
fg = show_hist(traits_list, data=smultixcan_results)
ax = fg.axes[0, 0]
ax.set_xlim(-1, 10)
ax.set_title("Original distribution")

# %% tags=[]
fg = show_hist(traits_list, data=smultixcan_results_std)
ax = fg.axes[0, 0]
ax.set_xlim(-0.00005, 0.0003)
ax.set_title("Standardized by $sum(zscores)$")

# %% [markdown] tags=[]
# Now the distribution makes more sense to me: `asthma` is slightly more towards the right than `height`, and the low sample size traits are at the leftmost.

# %% [markdown] tags=[]
# ## Projection

# %% tags=[]
mproj = MultiplierProjection()

# %% tags=[]
smultixcan_into_multiplier = mproj.transform(smultixcan_results_std)

# %% tags=[]
smultixcan_into_multiplier.shape

# %% tags=[]
smultixcan_into_multiplier.head()

# %% tags=[]
_tmp = smultixcan_into_multiplier.loc["LV603"].sort_values(ascending=False).head(20)
display(_tmp)

# %% tags=[]
traits_sample_size_df.loc[_tmp.index[0]]

# %% tags=[]
_tmp = smultixcan_into_multiplier.loc["LV136"].sort_values(ascending=False).head(20)
display(_tmp)

# %% tags=[]
traits_sample_size_df.loc[_tmp.index[0]]

# %% tags=[]
_tmp = smultixcan_into_multiplier.loc["LV844"].sort_values(ascending=False).head(20)
display(_tmp)

# %% tags=[]
traits_sample_size_df.loc[_tmp.index[0]]

# %% [markdown] tags=[]
# **Conclusion:** trait associations with known LVs are preserved with this approach.

# %% [markdown] tags=[]
# # Conclusion

# %% [markdown] tags=[]
# We select the last approach (Approach \#3) for downstream analyses.

# %% tags=[]
