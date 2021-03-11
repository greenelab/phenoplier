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
# It prepares the data to create a clustering tree visualization (using the R package `clustree`).

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
from IPython.display import display
from pathlib import Path

import numpy as np
import pandas as pd

from utils import generate_result_set_name
import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
CONSENSUS_CLUSTERING_DIR = Path(
    conf.RESULTS["CLUSTERING_DIR"], "consensus_clustering"
).resolve()

display(CONSENSUS_CLUSTERING_DIR)

# %% [markdown] tags=[]
# # Load data

# %% [markdown] tags=[]
# ## PCA

# %% tags=[]
INPUT_SUBSET = "pca"

# %% tags=[]
INPUT_STEM = "z_score_std-projection-smultixcan-efo_partial-mashr-zscores"

# %% tags=[]
DR_OPTIONS = {
    "n_components": 50,
    "svd_solver": "full",
    "random_state": 0,
}

# %% tags=[]
input_filepath = Path(
    conf.RESULTS["DATA_TRANSFORMATIONS_DIR"],
    INPUT_SUBSET,
    generate_result_set_name(
        DR_OPTIONS, prefix=f"{INPUT_SUBSET}-{INPUT_STEM}-", suffix=".pkl"
    ),
).resolve()
display(input_filepath)

assert input_filepath.exists(), "Input file does not exist"

input_filepath_stem = input_filepath.stem
display(input_filepath_stem)

# %% tags=[]
data_pca = pd.read_pickle(input_filepath).iloc[:, :5]

# %% tags=[]
data_pca.shape

# %% tags=[]
data_pca.head()

# %% [markdown] tags=[]
# ## UMAP

# %% tags=[]
INPUT_SUBSET = "umap"

# %% tags=[]
INPUT_STEM = "z_score_std-projection-smultixcan-efo_partial-mashr-zscores"

# %% tags=[]
DR_OPTIONS = {
    "n_components": 5,
    "metric": "euclidean",
    "n_neighbors": 15,
    "random_state": 0,
}

# %% tags=[]
input_filepath = Path(
    conf.RESULTS["DATA_TRANSFORMATIONS_DIR"],
    INPUT_SUBSET,
    generate_result_set_name(
        DR_OPTIONS, prefix=f"{INPUT_SUBSET}-{INPUT_STEM}-", suffix=".pkl"
    ),
).resolve()
display(input_filepath)

assert input_filepath.exists(), "Input file does not exist"

input_filepath_stem = input_filepath.stem
display(input_filepath_stem)

# %% tags=[]
data_umap = pd.read_pickle(input_filepath)

# %% tags=[]
data_umap.shape

# %% tags=[]
data_umap.head()

# %% [markdown] tags=[]
# # Load selected best partitions

# %% tags=[]
input_file = Path(CONSENSUS_CLUSTERING_DIR, "best_partitions_by_k.pkl").resolve()
display(input_file)

# %% tags=[]
best_partitions = pd.read_pickle(input_file)

# %% tags=[]
best_partitions.shape

# %% tags=[]
best_partitions.head()

# %% [markdown] tags=[]
# # Prepare data for clustrees

# %% tags=[]
clustrees_df = pd.concat((data_pca, data_umap), join="inner", axis=1)

# %% tags=[]
display(clustrees_df.shape)
assert clustrees_df.shape == (data_pca.shape[0], data_pca.shape[1] + data_umap.shape[1])

# %% tags=[]
clustrees_df.head()

# %% [markdown] tags=[]
# ## Add partitions

# %% tags=[]
_tmp = np.unique(
    [best_partitions.loc[k, "partition"].shape for k in best_partitions.index]
)
display(_tmp)
assert _tmp.shape[0] == 1
assert _tmp[0] == data_umap.shape[0] == data_pca.shape[0]

# %% tags=[]
assert not best_partitions.isna().any().any()

# %% tags=[]
# df = df.assign(**{f'k{k}': partitions.loc[k, 'partition'] for k in selected_k_values})
clustrees_df = clustrees_df.assign(
    **{
        f"k{k}": best_partitions.loc[k, "partition"]
        for k in best_partitions.index
        if best_partitions.loc[k, "selected"]
    }
)

# %% tags=[]
clustrees_df.index.rename("trait", inplace=True)

# %% tags=[]
clustrees_df.shape

# %% tags=[]
clustrees_df.head()

# %% tags=[]
# make sure partitions were assigned correctly
assert (
    np.unique(
        [
            clustrees_df[f"{k}"].value_counts().sum()
            for k in clustrees_df.columns[
                clustrees_df.columns.str.contains("^k[0-9]+$", regex=True)
            ]
        ]
    )[0]
    == data_pca.shape[0]
)

# %% [markdown] tags=[]
# # Save

# %% tags=[]
output_file = Path(CONSENSUS_CLUSTERING_DIR, "clustering_tree_data.tsv").resolve()
display(output_file)

# %% tags=[]
clustrees_df.to_csv(output_file, sep="\t")

# %% tags=[]
