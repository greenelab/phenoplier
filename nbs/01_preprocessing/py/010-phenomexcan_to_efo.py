# ---
# jupyter:
#   jupytext:
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

# %% [markdown] papermill={"duration": 0.047347, "end_time": "2020-12-14T21:24:39.706211", "exception": false, "start_time": "2020-12-14T21:24:39.658864", "status": "completed"} tags=[]
# # Description

# %% [markdown]
# It uses the PhenomeXcan traits to EFO mapping files to group traits that end up having the same EFO label. Currently, this only combines the S-MultiXcan results (z-scores) using the [Stouffer method](https://en.wikipedia.org/wiki/Fisher%27s_method#Relation_to_Stouffer's_Z-score_method) (implemented in functions `get_weights` and `_combine_z_scores` below).

# %% [markdown] papermill={"duration": 0.031906, "end_time": "2020-12-14T21:24:39.770133", "exception": false, "start_time": "2020-12-14T21:24:39.738227", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.02319, "end_time": "2020-12-14T21:24:39.807557", "exception": false, "start_time": "2020-12-14T21:24:39.784367", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 0.36702, "end_time": "2020-12-14T21:24:40.187800", "exception": false, "start_time": "2020-12-14T21:24:39.820780", "status": "completed"} tags=[]
import numpy as np
import pandas as pd

import conf
from data.cache import read_data
from entity import Trait

# %% [markdown] papermill={"duration": 0.01365, "end_time": "2020-12-14T21:24:40.215577", "exception": false, "start_time": "2020-12-14T21:24:40.201927", "status": "completed"} tags=[]
# # Load S-MultiXcan results

# %% papermill={"duration": 0.301476, "end_time": "2020-12-14T21:24:40.530059", "exception": false, "start_time": "2020-12-14T21:24:40.228583", "status": "completed"} tags=[]
smultixcan_zscores = read_data(
    conf.PHENOMEXCAN["SMULTIXCAN_MASHR_ZSCORES_FILE"]
)

# %% papermill={"duration": 0.028251, "end_time": "2020-12-14T21:24:40.573779", "exception": false, "start_time": "2020-12-14T21:24:40.545528", "status": "completed"} tags=[]
smultixcan_zscores.shape

# %% papermill={"duration": 0.037367, "end_time": "2020-12-14T21:24:40.624901", "exception": false, "start_time": "2020-12-14T21:24:40.587534", "status": "completed"} tags=[]
smultixcan_zscores.head()

# %% [markdown] papermill={"duration": 0.013405, "end_time": "2020-12-14T21:24:40.652569", "exception": false, "start_time": "2020-12-14T21:24:40.639164", "status": "completed"} tags=[]
# # Get PhenomeXcan traits

# %% papermill={"duration": 0.557081, "end_time": "2020-12-14T21:24:41.222794", "exception": false, "start_time": "2020-12-14T21:24:40.665713", "status": "completed"} tags=[]
phenomexcan_fullcode_to_traits = {
    (trait_obj := Trait.get_trait(full_code=trait_name)).full_code: trait_obj
    for trait_name in smultixcan_zscores.columns
}

# %% papermill={"duration": 0.023326, "end_time": "2020-12-14T21:24:41.259604", "exception": false, "start_time": "2020-12-14T21:24:41.236278", "status": "completed"} tags=[]
len(phenomexcan_fullcode_to_traits)

# %% papermill={"duration": 0.02293, "end_time": "2020-12-14T21:24:41.296358", "exception": false, "start_time": "2020-12-14T21:24:41.273428", "status": "completed"} tags=[]
assert len(phenomexcan_fullcode_to_traits) == smultixcan_zscores.columns.shape[0]

# %% [markdown] papermill={"duration": 0.014088, "end_time": "2020-12-14T21:24:41.324768", "exception": false, "start_time": "2020-12-14T21:24:41.310680", "status": "completed"} tags=[]
# # Change/combine traits in S-MultiXcan results

# %% papermill={"duration": 0.029626, "end_time": "2020-12-14T21:24:41.367817", "exception": false, "start_time": "2020-12-14T21:24:41.338191", "status": "completed"} tags=[]
traits_sample_size = pd.DataFrame([
    {
        'fullcode': fc,
        'n_cases': t.n_cases,
        'n_controls': t.n_controls,
        'n': t.n,
    }
    for fc, t in phenomexcan_fullcode_to_traits.items()
])

# %% papermill={"duration": 0.023718, "end_time": "2020-12-14T21:24:41.405440", "exception": false, "start_time": "2020-12-14T21:24:41.381722", "status": "completed"} tags=[]
traits_sample_size.shape

# %% papermill={"duration": 0.027715, "end_time": "2020-12-14T21:24:41.447200", "exception": false, "start_time": "2020-12-14T21:24:41.419485", "status": "completed"} tags=[]
traits_sample_size.head()


# %% papermill={"duration": 0.025799, "end_time": "2020-12-14T21:24:41.488390", "exception": false, "start_time": "2020-12-14T21:24:41.462591", "status": "completed"} tags=[]
def get_weights(traits_fullcodes):
    """
    This function takes a list of PhenomeXcan traits that map to the same EFO label, and returns their weights using sample sizes
    from GWASs. In the case of binary traits (i.e. diseases) the formula is:
        (n_cases / n_controls) * sqrt(n)
    where n=n_cases+n_controls
    In case of continuous traits (such as height) it is just n
    """
    return np.array([
        (t.n_cases / t.n_controls) * np.sqrt(t.n)
        if not pd.isnull((t := phenomexcan_fullcode_to_traits[trait_name]).n_cases) and not pd.isnull(t.n_controls)
        else t.n
        for trait_name in traits_fullcodes
    ])

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


# %% [markdown] papermill={"duration": 0.015798, "end_time": "2020-12-14T21:24:41.519131", "exception": false, "start_time": "2020-12-14T21:24:41.503333", "status": "completed"} tags=[]
# ## Get a list of EFO labels for PhenomeXcan traits

# %% papermill={"duration": 2.685565, "end_time": "2020-12-14T21:24:44.219103", "exception": false, "start_time": "2020-12-14T21:24:41.533538", "status": "completed"} tags=[]
traits_efo_labels = [
    t.get_efo_info().label
    if (t := phenomexcan_fullcode_to_traits[c]).get_efo_info() is not None
    else t.full_code
    for c in smultixcan_zscores.columns
]

# %% papermill={"duration": 0.024726, "end_time": "2020-12-14T21:24:44.258361", "exception": false, "start_time": "2020-12-14T21:24:44.233635", "status": "completed"} tags=[]
len(traits_efo_labels)

# %% papermill={"duration": 0.025158, "end_time": "2020-12-14T21:24:44.298606", "exception": false, "start_time": "2020-12-14T21:24:44.273448", "status": "completed"} tags=[]
traits_efo_labels[:10]

# %% [markdown] papermill={"duration": 0.015088, "end_time": "2020-12-14T21:24:44.329103", "exception": false, "start_time": "2020-12-14T21:24:44.314015", "status": "completed"} tags=[]
# ## Combine z-scores for same EFO labels

# %% papermill={"duration": 6.719873, "end_time": "2020-12-14T21:24:51.063699", "exception": false, "start_time": "2020-12-14T21:24:44.343826", "status": "completed"} tags=[]
smultixcan_zscores_combined = smultixcan_zscores.groupby(traits_efo_labels, axis=1).apply(_combine_z_scores)

# %% papermill={"duration": 0.025116, "end_time": "2020-12-14T21:24:51.105728", "exception": false, "start_time": "2020-12-14T21:24:51.080612", "status": "completed"} tags=[]
smultixcan_zscores_combined.shape

# %% papermill={"duration": 0.037619, "end_time": "2020-12-14T21:24:51.158659", "exception": false, "start_time": "2020-12-14T21:24:51.121040", "status": "completed"} tags=[]
smultixcan_zscores_combined.head()

# %% papermill={"duration": 0.09069, "end_time": "2020-12-14T21:24:51.265422", "exception": false, "start_time": "2020-12-14T21:24:51.174732", "status": "completed"} tags=[]
assert not smultixcan_zscores_combined.isna().any().any()

# %% [markdown] papermill={"duration": 0.017824, "end_time": "2020-12-14T21:24:51.302999", "exception": false, "start_time": "2020-12-14T21:24:51.285175", "status": "completed"} tags=[]
# ## Testing

# %% [markdown] papermill={"duration": 0.015233, "end_time": "2020-12-14T21:24:51.333889", "exception": false, "start_time": "2020-12-14T21:24:51.318656", "status": "completed"} tags=[]
# ### EFO label (asthma) which combined three PhenomeXcan traits.

# %% papermill={"duration": 0.025447, "end_time": "2020-12-14T21:24:51.374735", "exception": false, "start_time": "2020-12-14T21:24:51.349288", "status": "completed"} tags=[]
_asthma_traits = [
    '22127-Doctor_diagnosed_asthma',
    '20002_1111-Noncancer_illness_code_selfreported_asthma',
    'J45-Diagnoses_main_ICD10_J45_Asthma'
]

# %% papermill={"duration": 0.030594, "end_time": "2020-12-14T21:24:51.421078", "exception": false, "start_time": "2020-12-14T21:24:51.390484", "status": "completed"} tags=[]
smultixcan_zscores[_asthma_traits]

# %% papermill={"duration": 0.030275, "end_time": "2020-12-14T21:24:51.467871", "exception": false, "start_time": "2020-12-14T21:24:51.437596", "status": "completed"} tags=[]
traits_sample_size[traits_sample_size['fullcode'].isin(_asthma_traits)]

# %% papermill={"duration": 0.030206, "end_time": "2020-12-14T21:24:51.514718", "exception": false, "start_time": "2020-12-14T21:24:51.484512", "status": "completed"} tags=[]
_trait = 'asthma'

_gene = 'ENSG00000000419'
_weights = np.array([
    ((41934.0 / 319207.0) * np.sqrt(361141)),
    ((11717.0 / 80070.0) * np.sqrt(91787)),
    ((1693.0 / 359501.0) * np.sqrt(361194)),
])
assert smultixcan_zscores_combined.loc[_gene, _trait].round(3) == (
        (
            _weights[1] * 0.327024
            + _weights[0] * 0.707137
            + _weights[2] * 0.805021
        ) / np.sqrt(_weights[0] ** 2 + _weights[1] ** 2 + _weights[2] ** 2)
    ).round(3)

_gene = 'ENSG00000284526'
assert smultixcan_zscores_combined.loc[_gene, _trait].round(3) == (
        (
            _weights[1] * 0.302116
            + _weights[0] * 0.006106
            + _weights[2] * 0.463360
        ) / np.sqrt(_weights[0] ** 2 + _weights[1] ** 2 + _weights[2] ** 2)
    ).round(3)

# %% [markdown] papermill={"duration": 0.017625, "end_time": "2020-12-14T21:24:51.549048", "exception": false, "start_time": "2020-12-14T21:24:51.531423", "status": "completed"} tags=[]
# ### PhenomeXcan trait which has no EFO label.

# %% papermill={"duration": 0.029284, "end_time": "2020-12-14T21:24:51.595943", "exception": false, "start_time": "2020-12-14T21:24:51.566659", "status": "completed"} tags=[]
_trait = '100001_raw-Food_weight'

# %% papermill={"duration": 0.033309, "end_time": "2020-12-14T21:24:51.659574", "exception": false, "start_time": "2020-12-14T21:24:51.626265", "status": "completed"} tags=[]
traits_sample_size[traits_sample_size['fullcode'].isin((_trait,))]

# %% papermill={"duration": 0.029512, "end_time": "2020-12-14T21:24:51.706284", "exception": false, "start_time": "2020-12-14T21:24:51.676772", "status": "completed"} tags=[]
smultixcan_zscores[_trait]

# %% papermill={"duration": 0.028189, "end_time": "2020-12-14T21:24:51.751962", "exception": false, "start_time": "2020-12-14T21:24:51.723773", "status": "completed"} tags=[]
_gene = 'ENSG00000284513'
_weights = np.array([
    np.sqrt(51453),
])
assert smultixcan_zscores_combined.loc[_gene, _trait].round(3) == (
        (
            _weights[0] * 1.522281
        ) / np.sqrt(_weights[0] ** 2)
    ).round(3)

_gene = 'ENSG00000000971'
assert smultixcan_zscores_combined.loc[_gene, _trait].round(3) == (
        (
            _weights[0] * 0.548127
        ) / np.sqrt(_weights[0] ** 2)
    ).round(3)

# %% [markdown] papermill={"duration": 0.01664, "end_time": "2020-12-14T21:24:51.785620", "exception": false, "start_time": "2020-12-14T21:24:51.768980", "status": "completed"} tags=[]
# # Save full (all traits, some with EFO, some not)

# %% papermill={"duration": 0.027261, "end_time": "2020-12-14T21:24:51.829640", "exception": false, "start_time": "2020-12-14T21:24:51.802379", "status": "completed"} tags=[]
smultixcan_zscores_combined.shape

# %% papermill={"duration": 0.038884, "end_time": "2020-12-14T21:24:51.886083", "exception": false, "start_time": "2020-12-14T21:24:51.847199", "status": "completed"} tags=[]
smultixcan_zscores_combined.head()

# %% [markdown] papermill={"duration": 0.01755, "end_time": "2020-12-14T21:24:51.921427", "exception": false, "start_time": "2020-12-14T21:24:51.903877", "status": "completed"} tags=[]
# ## Pickle (binary)

# %% papermill={"duration": 0.027635, "end_time": "2020-12-14T21:24:51.966659", "exception": false, "start_time": "2020-12-14T21:24:51.939024", "status": "completed"} tags=[]
output_file = conf.PHENOMEXCAN["SMULTIXCAN_EFO_PARTIAL_MASHR_ZSCORES_FILE"]
display(output_file)

# %% papermill={"duration": 3.895306, "end_time": "2020-12-14T21:24:55.880398", "exception": false, "start_time": "2020-12-14T21:24:51.985092", "status": "completed"} tags=[]
smultixcan_zscores_combined.to_pickle(output_file)

# %% [markdown] papermill={"duration": 0.01907, "end_time": "2020-12-14T21:24:55.951472", "exception": false, "start_time": "2020-12-14T21:24:55.932402", "status": "completed"} tags=[]
# ## TSV (text)

# %% papermill={"duration": 0.028317, "end_time": "2020-12-14T21:24:55.997571", "exception": false, "start_time": "2020-12-14T21:24:55.969254", "status": "completed"} tags=[]
# tsv format
output_text_file = output_file.with_suffix('.tsv.gz')
display(output_text_file)

# %% papermill={"duration": 392.240426, "end_time": "2020-12-14T21:31:28.256860", "exception": false, "start_time": "2020-12-14T21:24:56.016434", "status": "completed"} tags=[]
smultixcan_zscores_combined.to_csv(
    output_text_file,
    sep='\t',
    index=True,
    float_format="%.5e"
)

# %% papermill={"duration": 0.01856, "end_time": "2020-12-14T21:31:28.294646", "exception": false, "start_time": "2020-12-14T21:31:28.276086", "status": "completed"} tags=[]
