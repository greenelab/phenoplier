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
# It projects eMERGE S-MultiXcan results into the MultiPLIER latent space.
#
# The eMERGE TWAS results are not publicly available, so this won't work if you don't have access to the data.

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
from pathlib import Path

from IPython.display import display
import pandas as pd

import conf
from entity import Gene
from multiplier import MultiplierProjection

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
RESULTS_PROJ_OUTPUT_DIR = Path(conf.RESULTS["PROJECTIONS_DIR"])
RESULTS_PROJ_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_PROJ_OUTPUT_DIR)

# %% [markdown] tags=[]
# # Load eMERGE data (S-MultiXcan)

# %% tags=[]
smultixcan_results_filename = conf.EMERGE["SMULTIXCAN_MASHR_ZSCORES_FILE"]

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

# %% [markdown] tags=[]
# ## Gene IDs to Gene names

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
# ## Remove NaN values

# %% tags=[]
smultixcan_results = smultixcan_results.dropna(how="any")

# %% tags=[]
smultixcan_results.shape

# %% [markdown] tags=[]
# # Standardize S-MultiXcan results

# %% [markdown] tags=[]
# Here we adjust for highly polygenic traits (see notebook `005_00-data_analysis.ipynb`): we penalize those traits that have large effect sizes across several genes, such as antropometric traits.

# %% tags=[]
_tmp = smultixcan_results.apply(lambda x: x / x.sum())

# %% tags=[]
_tmp.shape

# %% tags=[]
assert _tmp.shape == smultixcan_results.shape

# %% tags=[]
# some testing
_trait = "513"
_gene = "SCYL3"
assert (
    _tmp.loc[_gene, _trait]
    == smultixcan_results.loc[_gene, _trait] / smultixcan_results[_trait].sum()
)

_trait = "165.1"
_gene = "DPM1"
assert (
    _tmp.loc[_gene, _trait]
    == smultixcan_results.loc[_gene, _trait] / smultixcan_results[_trait].sum()
)

_trait = "327"
_gene = "CFH"
assert (
    _tmp.loc[_gene, _trait]
    == smultixcan_results.loc[_gene, _trait] / smultixcan_results[_trait].sum()
)

_trait = "701.2"
_gene = "C1orf112"
assert (
    _tmp.loc[_gene, _trait]
    == smultixcan_results.loc[_gene, _trait] / smultixcan_results[_trait].sum()
)

# %% tags=[]
smultixcan_results = _tmp

# %% tags=[]
smultixcan_results.head()

# %% [markdown] tags=[]
# # Project S-MultiXcan data into MultiPLIER latent space

# %% tags=[]
mproj = MultiplierProjection()

# %% tags=[]
smultixcan_into_multiplier = mproj.transform(smultixcan_results)

# %% tags=[]
smultixcan_into_multiplier.shape

# %% tags=[]
smultixcan_into_multiplier.head()

# %% [markdown] tags=[]
# # Quick analysis

# %% tags=[]
(smultixcan_into_multiplier.loc["LV603"].sort_values(ascending=False).head(20))

# %% tags=[]
(smultixcan_into_multiplier.loc["LV136"].sort_values(ascending=False).head(20))

# %% tags=[]
(smultixcan_into_multiplier.loc["LV844"].sort_values(ascending=False).head(20))

# %% [markdown] tags=[]
# # Save

# %% tags=[]
output_file = Path(
    RESULTS_PROJ_OUTPUT_DIR, f"projection-{results_filename_stem}.pkl"
).resolve()

display(output_file)

# %% tags=[]
smultixcan_into_multiplier.to_pickle(output_file)

# %% tags=[]
