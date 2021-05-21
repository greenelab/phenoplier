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
# TODO

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import HTML
from tqdm import tqdm

from entity import Trait, Gene

import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
EXPERIMENT_NAME = "lv"
LIPIDS_GENE_SET = "gene_set_decrease"

# %% tags=[]
RANDOM_SEED = 0
N_PERMUTATIONS = 1000
N_TOP_TRAITS = 25

# %% tags=[]
OUTPUT_DIR = Path(
    conf.RESULTS["CRISPR_ANALYSES"]["BASE_DIR"],
    f"{EXPERIMENT_NAME}-{LIPIDS_GENE_SET}",
    "permutations",
)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
display(OUTPUT_DIR)

# %% [markdown] tags=[]
# # Data loading

# %% [markdown] tags=[]
# ## Traits list

# %% tags=[]
input_filepath = OUTPUT_DIR / "lipid_traits_list.pkl"
display(input_filepath)

# %% tags=[]
with open(input_filepath, "rb") as handle:
    lipids_related_traits = pickle.load(handle)

# %% tags=[]
lipids_related_traits

# %% [markdown] tags=[]
# ## Permutations

# %% tags=[]
input_filepath = OUTPUT_DIR / "permutation_results.pkl"
display(input_filepath)

# %% tags=[]
with open(input_filepath, "rb") as handle:
    permutation_results = pickle.load(handle)

# %% tags=[]
df = pd.Series([x for l in permutation_results for x in l])

# %% tags=[]
with pd.option_context(
    "display.max_rows", None, "display.max_columns", None, "max_colwidth", None
):
    display(df.value_counts())

# %% [markdown] tags=[]
# # p-value

# %% tags=[]
# in this case we are permisive to compute the p-value, and count cases where at least half of the important traits are among the top
pval = (
    sum(
        [
            (i / len(lipids_related_traits)) > 0.50
            for i in list(map(len, permutation_results))
        ]
    )
    + 1
) / (len(permutation_results) + 1)

display(pval)

# %% tags=[]
assert pval < 0.01
