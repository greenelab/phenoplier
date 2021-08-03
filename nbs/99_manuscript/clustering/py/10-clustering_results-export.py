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
# It exports clustering results to an Excel file.

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
from pathlib import Path
from IPython.display import display

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import conf
from entity import Trait
from utils import get_git_repository_path

# %% [markdown]
# # Settings

# %% tags=[]
DELIVERABLES_BASE_DIR = get_git_repository_path() / "data"
display(DELIVERABLES_BASE_DIR)

# %% tags=[]
OUTPUT_DIR = DELIVERABLES_BASE_DIR / "clustering" / "partitions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
display(OUTPUT_DIR)

# %%
N_TOP_PARTITIONS = 5

# %% [markdown] tags=[]
# # Load data

# %% [markdown] tags=[]
# ## PhenomeXcan (S-MultiXcan)

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

# %% tags=[]
data = pd.read_pickle(input_filepath)

# %% tags=[]
data.shape

# %% tags=[]
data.head()

# %% [markdown] tags=[]
# ## Best clustering partitions

# %%
CONSENSUS_CLUSTERING_DIR = Path(
    conf.RESULTS["CLUSTERING_DIR"], "consensus_clustering"
).resolve()

display(CONSENSUS_CLUSTERING_DIR)

# %% tags=[]
input_file = Path(CONSENSUS_CLUSTERING_DIR, "best_partitions_by_k.pkl").resolve()
display(input_file)

# %% tags=[]
best_partitions = pd.read_pickle(input_file)

# %%
best_partitions = best_partitions[best_partitions["selected"]]

# %% tags=[]
best_partitions.shape

# %% tags=[]
best_partitions.head()


# %% [markdown] tags=[]
# # Export clustering results

# %%
def get_trait_objs(phenotype_full_code):
    if Trait.is_efo_label(phenotype_full_code):
        traits = Trait.get_traits_from_efo(phenotype_full_code)
    else:
        traits = [Trait.get_trait(full_code=phenotype_full_code)]

    # sort by sample size
    return sorted(traits, key=lambda x: x.n_cases / x.n, reverse=True)


def get_trait_description(phenotype_full_code):
    traits = get_trait_objs(phenotype_full_code)

    return traits[0].description


def get_trait_n(phenotype_full_code):
    traits = get_trait_objs(phenotype_full_code)

    return traits[0].n


def get_trait_n_cases(phenotype_full_code):
    traits = get_trait_objs(phenotype_full_code)

    return traits[0].n_cases


def num_to_int_str(num):
    if pd.isnull(num):
        return ""

    return f"{num:,.0f}"


def get_part_clust(row):
    return f"{row.part_k} / {row.cluster_id}"


# %%
for part_k in best_partitions.reset_index().sort_values("k", ascending=False)[
    :N_TOP_PARTITIONS
]["k"]:
    print(part_k)

    part = best_partitions.loc[part_k, "partition"]
    cluster_id_sizes = pd.Series(part).value_counts()

    with pd.ExcelWriter(OUTPUT_DIR / f"clustering_part{part_k}.xlsx") as writer:
        for cluster_id, cluster_size in cluster_id_sizes.iteritems():
            cluster_traits = data.index[part == cluster_id]

            cluster_traits_df = [
                {
                    "Trait description": get_trait_description(t),
                    "Sample size": get_trait_n(t),
                    "Number of cases": get_trait_n_cases(t),
                }
                for t in cluster_traits
            ]
            cluster_traits_df = pd.DataFrame(cluster_traits_df)

            cluster_traits_df.to_excel(writer, sheet_name=f"Cluster {cluster_id}")

# %%
