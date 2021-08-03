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
# This notebooks checks which traits move from different partitions and clusters across the clustering tree.

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
from IPython.display import display
from pathlib import Path

import pandas as pd

from data.cache import read_data
from utils import generate_result_set_name
import conf

# %% [markdown] tags=[]
# # Settings

# %% [markdown] tags=[]
# # Load data

# %% [markdown]
# ## S-MultiXcan projection (`z_score_std`)

# %% tags=[]
INPUT_SUBSET = "z_score_std"

# %% tags=[]
INPUT_STEM = "projection-smultixcan-efo_partial-mashr-zscores"

# %% tags=[]
input_filepath = Path(
    conf.RESULTS["DATA_TRANSFORMATIONS_DIR"],
    INPUT_SUBSET,
    f"{INPUT_SUBSET}-{INPUT_STEM}.pkl",
).resolve()
display(input_filepath)

assert input_filepath.exists(), "Input file does not exist"

input_filepath_stem = input_filepath.stem
display(input_filepath_stem)

# %% tags=[]
data = read_data(input_filepath)

# %% tags=[]
data.shape

# %% tags=[]
data.head()

# %% [markdown]
# ## Clustering results

# %%
CONSENSUS_CLUSTERING_DIR = Path(
    conf.RESULTS["CLUSTERING_DIR"], "consensus_clustering"
).resolve()

display(CONSENSUS_CLUSTERING_DIR)

# %%
input_file = Path(CONSENSUS_CLUSTERING_DIR, "best_partitions_by_k.pkl").resolve()
display(input_file)

# %%
best_partitions = read_data(input_file)

# %%
# keep selected partitions only
best_partitions = best_partitions[best_partitions["selected"]]

# %%
best_partitions.shape

# %%
best_partitions.head()

# %% [markdown]
# # Within the "complex" branch

# %% [markdown]
# ## part15 k14 to part16 k14

# %%
part_one = best_partitions.loc[15, "partition"]
part_two = best_partitions.loc[16, "partition"]

# %%
part_one_clus = data.loc[part_one == 14].index
part_two_clus = data.loc[part_two == 14].index

# %%
part_one_clus

# %%
part_two_clus

# %%
part_one_clus.intersection(part_two_clus)

# %% [markdown]
# ## part25 k24 to part26 k15

# %%
part_one = best_partitions.loc[25, "partition"]
part_two = best_partitions.loc[26, "partition"]

# %%
part_one_clus = data.loc[part_one == 24].index
part_two_clus = data.loc[part_two == 15].index

# %%
part_one_clus

# %%
part_two_clus

# %%
part_one_clus.intersection(part_two_clus)

# %% [markdown]
# ## part26 k13 to part29 k21

# %%
part_one = best_partitions.loc[26, "partition"]
part_two = best_partitions.loc[29, "partition"]

# %%
part_one_clus = data.loc[part_one == 13].index
part_two_clus = data.loc[part_two == 21].index

# %%
part_one_clus

# %%
part_two_clus

# %%
part_one_clus.intersection(part_two_clus)

# %% [markdown]
# # Between branches

# %% [markdown]
# ## part22 k13 to part25 k23

# %%
part_one = best_partitions.loc[22, "partition"]
part_two = best_partitions.loc[25, "partition"]

# %%
part_one_clus = data.loc[part_one == 13].index
part_two_clus = data.loc[part_two == 23].index

# %%
part_one_clus

# %%
part_two_clus

# %%
part_one_clus.intersection(part_two_clus)

# %% [markdown]
# ## part26 k0 to part29 k21

# %%
part_one = best_partitions.loc[26, "partition"]
part_two = best_partitions.loc[29, "partition"]

# %%
part_one_clus = data.loc[part_one == 0].index
part_two_clus = data.loc[part_two == 21].index

# %%
part_one_clus

# %%
part_two_clus

# %%
part_one_clus.intersection(part_two_clus)

# %%
