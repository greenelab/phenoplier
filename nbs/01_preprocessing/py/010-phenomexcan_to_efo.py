# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill
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

# %% [markdown]
# It uses the PhenomeXcan traits to EFO mapping files to group traits that end up having the same EFO label. Currently, this only combines the S-MultiXcan results (z-scores) using the [Stouffer method](https://en.wikipedia.org/wiki/Fisher%27s_method#Relation_to_Stouffer's_Z-score_method) (implemented in functions `get_weights` and `_combine_z_scores` below).

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
# # Load S-MultiXcan results

# %% tags=[]
smultixcan_zscores = read_data(conf.PHENOMEXCAN["SMULTIXCAN_MASHR_ZSCORES_FILE"])

# %% tags=[]
smultixcan_zscores.shape

# %% tags=[]
smultixcan_zscores.head()

# %% [markdown] tags=[]
# # Get PhenomeXcan traits

# %% tags=[]
phenomexcan_fullcode_to_traits = {
    (trait_obj := Trait.get_trait(full_code=trait_name)).full_code: trait_obj
    for trait_name in smultixcan_zscores.columns
}

# %% tags=[]
len(phenomexcan_fullcode_to_traits)

# %% tags=[]
assert len(phenomexcan_fullcode_to_traits) == smultixcan_zscores.columns.shape[0]

# %% [markdown] tags=[]
# # Change/combine traits in S-MultiXcan results

# %% tags=[]
traits_sample_size = pd.DataFrame(
    [
        {
            "fullcode": fc,
            "n_cases": t.n_cases,
            "n_controls": t.n_controls,
            "n": t.n,
        }
        for fc, t in phenomexcan_fullcode_to_traits.items()
    ]
)

# %% tags=[]
traits_sample_size.shape

# %% tags=[]
traits_sample_size.head()


# %% tags=[]
def get_weights(traits_fullcodes):
    """
    This function takes a list of PhenomeXcan traits that map to the same EFO label, and returns their weights using sample sizes
    from GWASs. In the case of binary traits (i.e. diseases) the formula is:
        (n_cases / n_controls) * sqrt(n)
    where n=n_cases+n_controls
    In case of continuous traits (such as height) it is just n
    """
    return np.array(
        [
            (t.n_cases / t.n_controls) * np.sqrt(t.n)
            if not pd.isnull((t := phenomexcan_fullcode_to_traits[trait_name]).n_cases)
            and not pd.isnull(t.n_controls)
            else t.n
            for trait_name in traits_fullcodes
        ]
    )


def _combine_z_scores(x):
    """
    Combines PhenomeXcan traits that map to the same EFO label using the Stouffer's Z-score method:
    https://en.wikipedia.org/wiki/Fisher%27s_method#Relation_to_Stouffer's_Z-score_method

    It uses weights for each traits, which are computed with function get_weights.

    Args:
        x: a pandas.DataFrame with PhenomeXcan traits in the columns, and genes in the rows. Values are z-scores of association in S-MultiXcan.

    Returns:
        pandas.Series for all genes and the single EFO label for which all traits in x map to. Values are the combined z-scores.
    """
    # combine z-scores using Stouffer's method
    weights = get_weights(x.columns)
    numerator = (x * weights).sum(1)
    denominator = np.sqrt(np.power(weights, 2).sum())
    new_data = numerator / denominator

    return pd.Series(
        data=new_data.values,
        index=x.index.copy(),
        name=x.columns[0],
    )


# %% [markdown] tags=[]
# ## Get a list of EFO labels for PhenomeXcan traits

# %% tags=[]
traits_efo_labels = [
    t.get_efo_info().label
    if (t := phenomexcan_fullcode_to_traits[c]).get_efo_info() is not None
    else t.full_code
    for c in smultixcan_zscores.columns
]

# %% tags=[]
len(traits_efo_labels)

# %% tags=[]
traits_efo_labels[:10]

# %% [markdown] tags=[]
# ## Combine z-scores for same EFO labels

# %% tags=[]
smultixcan_zscores_combined = smultixcan_zscores.groupby(
    traits_efo_labels, axis=1
).apply(_combine_z_scores)

# %% tags=[]
smultixcan_zscores_combined.shape

# %% tags=[]
smultixcan_zscores_combined.head()

# %% tags=[]
assert not smultixcan_zscores_combined.isna().any().any()

# %% [markdown] tags=[]
# ## Testing

# %% [markdown] tags=[]
# ### EFO label (asthma) which combined three PhenomeXcan traits.

# %% tags=[]
_asthma_traits = [
    "22127-Doctor_diagnosed_asthma",
    "20002_1111-Noncancer_illness_code_selfreported_asthma",
    "J45-Diagnoses_main_ICD10_J45_Asthma",
]

# %% tags=[]
smultixcan_zscores[_asthma_traits]

# %% tags=[]
traits_sample_size[traits_sample_size["fullcode"].isin(_asthma_traits)]

# %% tags=[]
_trait = "asthma"

_gene = "ENSG00000000419"
_weights = np.array(
    [
        ((41934.0 / 319207.0) * np.sqrt(361141)),
        ((11717.0 / 80070.0) * np.sqrt(91787)),
        ((1693.0 / 359501.0) * np.sqrt(361194)),
    ]
)
assert smultixcan_zscores_combined.loc[_gene, _trait].round(3) == (
    (_weights[1] * 0.327024 + _weights[0] * 0.707137 + _weights[2] * 0.805021)
    / np.sqrt(_weights[0] ** 2 + _weights[1] ** 2 + _weights[2] ** 2)
).round(3)

_gene = "ENSG00000284526"
assert smultixcan_zscores_combined.loc[_gene, _trait].round(3) == (
    (_weights[1] * 0.302116 + _weights[0] * 0.006106 + _weights[2] * 0.463360)
    / np.sqrt(_weights[0] ** 2 + _weights[1] ** 2 + _weights[2] ** 2)
).round(3)

# %% [markdown] tags=[]
# ### PhenomeXcan trait which has no EFO label.

# %% tags=[]
_trait = "100001_raw-Food_weight"

# %% tags=[]
traits_sample_size[traits_sample_size["fullcode"].isin((_trait,))]

# %% tags=[]
smultixcan_zscores[_trait]

# %% tags=[]
_gene = "ENSG00000284513"
_weights = np.array(
    [
        np.sqrt(51453),
    ]
)
assert smultixcan_zscores_combined.loc[_gene, _trait].round(3) == (
    (_weights[0] * 1.522281) / np.sqrt(_weights[0] ** 2)
).round(3)

_gene = "ENSG00000000971"
assert smultixcan_zscores_combined.loc[_gene, _trait].round(3) == (
    (_weights[0] * 0.548127) / np.sqrt(_weights[0] ** 2)
).round(3)

# %% [markdown] tags=[]
# # Save full (all traits, some with EFO, some not)

# %% tags=[]
smultixcan_zscores_combined.shape

# %% tags=[]
smultixcan_zscores_combined.head()

# %% [markdown] tags=[]
# ## Pickle (binary)

# %% tags=[]
output_file = conf.PHENOMEXCAN["SMULTIXCAN_EFO_PARTIAL_MASHR_ZSCORES_FILE"]
display(output_file)

# %% tags=[]
smultixcan_zscores_combined.to_pickle(output_file)

# %% [markdown] tags=[]
# ## TSV (text)

# %% tags=[]
# tsv format
output_text_file = output_file.with_suffix(".tsv.gz")
display(output_text_file)

# %% tags=[]
smultixcan_zscores_combined.to_csv(
    output_text_file, sep="\t", index=True, float_format="%.5e"
)

# %% tags=[]
