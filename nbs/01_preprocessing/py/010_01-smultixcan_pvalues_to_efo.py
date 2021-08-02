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
# It uses the PhenomeXcan traits to EFO mapping files to group traits that end up having the same EFO label. This only combines the S-MultiXcan results (p-values) by taking the minimum p-value across all traits with the same EFO label.

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
from IPython.display import display
import numpy as np
import pandas as pd

import conf
from data.cache import read_data
from entity import Trait

# %% [markdown] tags=[]
# # Load S-MultiXcan

# %% [markdown] tags=[]
# ## z-scores (EFO-mapped)

# %% tags=[]
smultixcan_zscores_combined = read_data(
    conf.PHENOMEXCAN["SMULTIXCAN_EFO_PARTIAL_MASHR_ZSCORES_FILE"]
)

# %% tags=[]
smultixcan_zscores_combined.shape

# %% tags=[]
smultixcan_zscores_combined.head()

# %% [markdown] tags=[]
# ## p-values (original)

# %% tags=[]
smultixcan_pvalues = read_data(conf.PHENOMEXCAN["SMULTIXCAN_MASHR_PVALUES_FILE"])

# %% tags=[]
smultixcan_pvalues.shape

# %% tags=[]
smultixcan_pvalues.head()

# %% [markdown] tags=[]
# # Get PhenomeXcan traits

# %% tags=[]
phenomexcan_fullcode_to_traits = {
    (trait_obj := Trait.get_trait(full_code=trait_name)).full_code: trait_obj
    for trait_name in smultixcan_pvalues.columns
}

# %% tags=[]
len(phenomexcan_fullcode_to_traits)

# %% tags=[]
assert len(phenomexcan_fullcode_to_traits) == smultixcan_pvalues.columns.shape[0]

# %% [markdown] tags=[]
# # Change/combine traits in S-MultiXcan results

# %% [markdown] tags=[]
# ## Get a list of EFO labels for PhenomeXcan traits

# %% tags=[]
traits_efo_labels = [
    t.get_efo_info().label
    if (t := phenomexcan_fullcode_to_traits[c]).get_efo_info() is not None
    else t.full_code
    for c in smultixcan_pvalues.columns
]

# %% tags=[]
len(traits_efo_labels)

# %% tags=[]
traits_efo_labels[:10]

# %% [markdown] tags=[]
# ## Get `min(p-value)` for same EFO labels

# %% tags=[]
smultixcan_pvalues_combined = smultixcan_pvalues.groupby(
    traits_efo_labels, axis=1
).min()

# %% tags=[]
smultixcan_pvalues_combined.shape

# %% tags=[]
smultixcan_pvalues_combined.head()

# %% [markdown] tags=[]
# ### Keep same order of genes and traits as in `z_scores` data

# %% tags=[]
assert set(smultixcan_pvalues_combined.index) == set(smultixcan_zscores_combined.index)

# %% tags=[]
assert set(smultixcan_pvalues_combined.columns) == set(
    smultixcan_zscores_combined.columns
)

# %% tags=[]
smultixcan_pvalues_combined = smultixcan_pvalues_combined.loc[
    smultixcan_zscores_combined.index, smultixcan_zscores_combined.columns
]

# %% tags=[]
smultixcan_pvalues_combined.shape

# %% [markdown] tags=[]
# ### Do we have NaN values?

# %% tags=[]
smultixcan_pvalues_combined.isna().any().any()

# %% tags=[]
smultixcan_pvalues_combined.isna().sum()

# %% tags=[]
smultixcan_pvalues_combined.dropna(axis=0).shape

# %% [markdown] tags=[]
# ### Are we getting those NaN values from `z-scores == 0`?

# %% tags=[]
zscores_zeros = (smultixcan_zscores_combined == 0.0).values
display(zscores_zeros.shape)

zscore_zeros_idx = np.where(zscores_zeros)
display(zscore_zeros_idx[0].shape)

# %% tags=[]
pvalues_nans = (smultixcan_pvalues_combined.isna()).values
display(pvalues_nans.shape)

pvalues_nans_idx = np.where(pvalues_nans)
display(pvalues_nans_idx[0].shape)

# %% tags=[]
np.array_equal(zscores_zeros, pvalues_nans)

# %% [markdown] tags=[]
# NaN pvalues do not exactly match zero z-scores. Let's see what's going on

# %% tags=[]
non_equal_idxs = np.where(zscores_zeros != pvalues_nans)
display(non_equal_idxs)

# %% tags=[]
non_equal_idxs[0].shape, non_equal_idxs[1].shape

# %% [markdown] tags=[]
# `z-scores` and `p-values` data version differ in just a few positions, so it is mostly because of `z_scores == 0.0`

# %% [markdown] tags=[]
# ### What's in the differing positions?

# %% tags=[]
zscores_values = smultixcan_zscores_combined.iloc[non_equal_idxs].stack()

# %% tags=[]
zscores_values.shape

# %% tags=[]
zscores_values.head()

# %% tags=[]
zscores_values.describe()

# %% tags=[]
zscores_values.sort_values(ascending=False)

# %% tags=[]
pvalues_values = smultixcan_pvalues_combined.iloc[non_equal_idxs].stack()

# %% tags=[]
pvalues_values.shape

# %% tags=[]
pvalues_values.head()

# %% tags=[]
pvalues_values.describe()

# %% tags=[]
pvalues_values.sort_values(ascending=False)

# %% [markdown] tags=[]
# ## Testing

# %% [markdown] tags=[]
# ### Stats

# %% tags=[]
_stats = smultixcan_pvalues_combined.stack().describe()
display(_stats.apply(str))

# %% tags=[]
assert _stats["min"] > 0.0

# %% tags=[]
assert _stats["max"] <= 1.0

# %% [markdown] tags=[]
# ### Same traits as in z-scores version

# %% tags=[]
assert smultixcan_pvalues_combined.index.equals(smultixcan_zscores_combined.index)

# %% tags=[]
assert smultixcan_pvalues_combined.columns.equals(smultixcan_zscores_combined.columns)

# %% [markdown] tags=[]
# ### EFO label (asthma) which combined three PhenomeXcan traits.

# %% tags=[]
_asthma_traits = [
    "22127-Doctor_diagnosed_asthma",
    "20002_1111-Noncancer_illness_code_selfreported_asthma",
    "J45-Diagnoses_main_ICD10_J45_Asthma",
]

# %% tags=[]
smultixcan_pvalues[_asthma_traits]

# %% tags=[]
_tmp = smultixcan_pvalues[_asthma_traits]
display(_tmp[_tmp.isna().any(axis=1)])

# %% tags=[]
_trait = "asthma"

_gene = "ENSG00000000419"
assert smultixcan_pvalues_combined.loc[_gene, _trait].round(3) == 0.421

_gene = "ENSG00000284526"
assert smultixcan_pvalues_combined.loc[_gene, _trait].round(3) == 0.643

_gene = "ENSG00000000938"
assert smultixcan_pvalues_combined.loc[_gene, _trait].round(3) == 0.020

_gene = "ENSG00000077327"
assert pd.isnull(smultixcan_pvalues_combined.loc[_gene, _trait])

# %% [markdown] tags=[]
# ### PhenomeXcan trait which has no EFO label.

# %% tags=[]
_trait = "100001_raw-Food_weight"

# %% tags=[]
smultixcan_pvalues[_trait]

# %% tags=[]
_gene = "ENSG00000284513"
assert smultixcan_pvalues_combined.loc[_gene, _trait].round(3) == 0.128

_gene = "ENSG00000000971"
assert smultixcan_pvalues_combined.loc[_gene, _trait].round(3) == 0.584

# %% [markdown] tags=[]
# # Save full (all traits, some with EFO, some not)

# %% tags=[]
smultixcan_pvalues_combined.shape

# %% tags=[]
smultixcan_pvalues_combined.head()

# %% [markdown] tags=[]
# ## Pickle (binary)

# %% tags=[]
output_file = conf.PHENOMEXCAN["SMULTIXCAN_EFO_PARTIAL_MASHR_PVALUES_FILE"]
display(output_file)

# %% tags=[]
smultixcan_pvalues_combined.to_pickle(output_file)

# %% [markdown] tags=[]
# ## TSV (text)

# %% tags=[]
# tsv format
output_text_file = output_file.with_suffix(".tsv.gz")
display(output_text_file)

# %% tags=[]
smultixcan_pvalues_combined.to_csv(
    output_text_file, sep="\t", index=True, float_format="%.5e"
)

# %% tags=[]
