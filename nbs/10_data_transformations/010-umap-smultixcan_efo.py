# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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

# %% [markdown]
# # Description

# %% [markdown]
# It projects input data into a UMAP representation.

# %% [markdown]
# # Modules loading

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from pathlib import Path
from IPython.display import display

import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns

import conf
from utils import generate_result_set_name

# %% [markdown]
# # Settings

# %%
INPUT_FILEPATH = Path(
    conf.PHENOMEXCAN["SMULTIXCAN_EFO_PARTIAL_MASHR_ZSCORES_FILE"]
).resolve()
display(INPUT_FILEPATH)

input_filepath_stem = INPUT_FILEPATH.stem
display(input_filepath_stem)

# %%
# number of components to use in the dimensionality reduction step
DR_OPTIONS = {
    'n_components': [5, 10, 20, 30, 40, 50],
    'metric': 'euclidean',
    'n_neighbors': 15,
    'random_state': 0,
}

# %%
# output dir for this notebook
RESULTS_DIR = Path(
    conf.RESULTS["DATA_TRANSFORMATIONS_DIR"],
    'umap'
).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %%
# dictionary containing all options/settings (used to generate filenames)
ALL_OPTIONS = DR_OPTIONS.copy()
# ALL_OPTIONS['proj_percentile'] = PERCENTILE_NAME

display(ALL_OPTIONS)

# %% [markdown]
# # Load input file

# %%
data = pd.read_pickle(INPUT_FILEPATH).T

# %%
data.shape

# %%
data.head()


# %% [markdown]
# # UMAP

# %%
def get_umap_proj(orig_data, options):
    umap_obj = umap.UMAP(**{k:v for k, v in options.items() if k in DR_OPTIONS})
    umap_obj = umap_obj.fit(orig_data)
    umap_data = umap_obj.transform(orig_data)
    return pd.DataFrame(
        data=umap_data,
        index=orig_data.index.copy(),
        columns=[f'UMAP{i+1}' for i in range(umap_data.shape[1])]
    )


# %%
# for n_comp, n_neigh in product(DR_OPTIONS['n_components'], DR_OPTIONS['n_neighbors']):
for n_comp in DR_OPTIONS['n_components']:
    print(f'# components: {n_comp}')
    
    options = ALL_OPTIONS.copy()
    options['n_components'] = n_comp
    
    dr_data = get_umap_proj(data, options)
    
    display(dr_data.shape)
    assert dr_data.shape == (data.shape[0], n_comp)
    
    display(dr_data.iloc[:, 0:5].describe())
    
    # save
    output_file = Path(
        RESULTS_DIR,
        generate_result_set_name(
            options,
            prefix=f'umap-{input_filepath_stem}-',
            suffix='.pkl'
        )
    ).resolve()
    display(output_file)
    
    dr_data.to_pickle(output_file)
    
    print('\n')

# %%
