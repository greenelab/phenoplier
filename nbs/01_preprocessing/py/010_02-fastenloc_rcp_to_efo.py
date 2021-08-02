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
# It uses the PhenomeXcan traits to EFO mapping files to group traits that end up having the same EFO label. This only combines the fastENLOC results (RCPs) by taking the maximum value across all traits with the same EFO label.

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
# # Load fastENLOC

# %% tags=[]
fastenloc_rcps = read_data(conf.PHENOMEXCAN["FASTENLOC_TORUS_RCP_FILE"])

# %% tags=[]
fastenloc_rcps.shape

# %% tags=[]
fastenloc_rcps.head()

# %% [markdown] tags=[]
# # Load S-MultiXcan z-scores (EFO-mapped)

# %% tags=[]
smultixcan_zscores_combined = read_data(
    conf.PHENOMEXCAN["SMULTIXCAN_EFO_PARTIAL_MASHR_ZSCORES_FILE"]
)

# %% tags=[]
smultixcan_zscores_combined.shape

# %% tags=[]
smultixcan_zscores_combined.head()

# %% [markdown] tags=[]
# # Get PhenomeXcan traits

# %% tags=[]
phenomexcan_fullcode_to_traits = {
    (trait_obj := Trait.get_trait(full_code=trait_name)).full_code: trait_obj
    for trait_name in fastenloc_rcps.columns
}

# %% tags=[]
len(phenomexcan_fullcode_to_traits)

# %% tags=[]
assert len(phenomexcan_fullcode_to_traits) == fastenloc_rcps.columns.shape[0]

# %% [markdown] tags=[]
# # Change/combine traits in S-MultiXcan results

# %% [markdown] tags=[]
# ## Get a list of EFO labels for PhenomeXcan traits

# %% tags=[]
traits_efo_labels = [
    t.get_efo_info().label
    if (t := phenomexcan_fullcode_to_traits[c]).get_efo_info() is not None
    else t.full_code
    for c in fastenloc_rcps.columns
]

# %% tags=[]
len(traits_efo_labels)

# %% tags=[]
traits_efo_labels[:10]

# %% [markdown] tags=[]
# ## Get `max(rcp)` for same EFO labels

# %% tags=[]
fastenloc_rcps_combined = fastenloc_rcps.groupby(traits_efo_labels, axis=1).max()

# %% tags=[]
fastenloc_rcps_combined.shape

# %% tags=[]
fastenloc_rcps_combined.head()

# %% [markdown] tags=[]
# ### Keep same order traits as in the S-MultiXcan results

# %% tags=[]
assert set(fastenloc_rcps_combined.columns) == set(smultixcan_zscores_combined.columns)

# %% tags=[]
fastenloc_rcps_combined = fastenloc_rcps_combined.loc[
    :, smultixcan_zscores_combined.columns
]

# %% tags=[]
fastenloc_rcps_combined.shape

# %% [markdown] tags=[]
# ### Do we have NaN values?

# %% tags=[]
fastenloc_rcps_combined.isna().any().any()

# %% tags=[]
fastenloc_rcps_combined.isna().sum()

# %% [markdown] tags=[]
# ## Testing

# %% [markdown] tags=[]
# ### Stats

# %% tags=[]
_stats = fastenloc_rcps_combined.stack().describe()
display(_stats.apply(str))

# %% tags=[]
assert _stats["min"] >= 0.0

# %% tags=[]
assert _stats["max"] <= 2.5

# %% [markdown] tags=[]
# ### Same traits as in z-scores version

# %% tags=[]
assert fastenloc_rcps_combined.columns.equals(smultixcan_zscores_combined.columns)

# %% [markdown] tags=[]
# ### EFO label (asthma) which combined three PhenomeXcan traits.

# %% tags=[]
_asthma_traits = [
    "22127-Doctor_diagnosed_asthma",
    "20002_1111-Noncancer_illness_code_selfreported_asthma",
    "J45-Diagnoses_main_ICD10_J45_Asthma",
]

# %% tags=[]
fastenloc_rcps[_asthma_traits]

# %% tags=[]
fastenloc_rcps[_asthma_traits].isna().sum(axis=1).sort_values(ascending=False).head()

# %% tags=[]
_trait = "asthma"

_gene = "ENSG00000000419"
assert fastenloc_rcps_combined.loc[_gene, _trait].round(3) == 0.001

_gene = "ENSG00000000457"
assert fastenloc_rcps_combined.loc[_gene, _trait].round(3) == 0.005

_gene = "ENSG00000284543"
assert fastenloc_rcps_combined.loc[_gene, _trait].round(4) == 0.0008

_gene = "ENSG00000253892"
assert pd.isnull(fastenloc_rcps_combined.loc[_gene, _trait])

# %% [markdown] tags=[]
# ### PhenomeXcan trait which has no EFO label.

# %% tags=[]
_trait = "100001_raw-Food_weight"

# %% tags=[]
fastenloc_rcps[_trait].sort_values(ascending=False)

# %% tags=[]
_gene = "ENSG00000280319"
assert fastenloc_rcps_combined.loc[_gene, _trait].round(3) == 0.360

_gene = "ENSG00000196071"
assert fastenloc_rcps_combined.loc[_gene, _trait].round(3) == 0.015

_gene = "ENSG00000284552"
assert pd.isnull(fastenloc_rcps_combined.loc[_gene, _trait])

# %% [markdown] tags=[]
# # Save full (all traits, some with EFO, some not)

# %% tags=[]
fastenloc_rcps_combined.shape

# %% tags=[]
fastenloc_rcps_combined.head()

# %% [markdown] tags=[]
# ## Pickle (binary)

# %% tags=[]
output_file = conf.PHENOMEXCAN["FASTENLOC_EFO_PARTIAL_TORUS_RCP_FILE"]
display(output_file)

# %% tags=[]
fastenloc_rcps_combined.to_pickle(output_file)

# %% [markdown] tags=[]
# ## TSV (text)

# %% tags=[]
# tsv format
output_text_file = output_file.with_suffix(".tsv.gz")
display(output_text_file)

# %% tags=[]
fastenloc_rcps_combined.to_csv(
    output_text_file, sep="\t", index=True, float_format="%.5e"
)

# %% tags=[]
