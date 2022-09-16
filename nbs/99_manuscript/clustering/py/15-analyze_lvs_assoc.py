# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
#     formats: ipynb,py//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] tags=[]
# # Description

# %% [markdown] tags=[]
# **TODO:** update

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
import re
from pathlib import Path

import pandas as pd

from entity import Trait
import conf

# %% [markdown] tags=[]
# # Settings

# %%
# assert (
#     conf.MANUSCRIPT["BASE_DIR"] is not None
# ), "The manuscript directory was not configured"

# OUTPUT_FILE_PATH = conf.MANUSCRIPT["CONTENT_DIR"] / "50.00.supplementary_material.md"
# display(OUTPUT_FILE_PATH)
# assert OUTPUT_FILE_PATH.exists()

# %%
# # result_set is either phenomexcan or emerge
# LV_FILE_MARK_TEMPLATE = "<!-- {lv}:{result_set}_traits_assocs:{position} -->"

# %%
# TABLE_CAPTION = "Table: Significant trait associations of {lv_name} in {result_set_name}. {table_id}"

# %%
# TABLE_CAPTION_ID = "#tbl:sup:{result_set}_assocs:{lv_name_lower_case}"

# %%
# RESULT_SET_NAMES = {
#     "phenomexcan": "PhenomeXcan",
#     "emerge": "eMERGE",
# }

# %%
CLUSTERING_K = 29

# %% [markdown] tags=[]
# # Paths

# %%
CONSENSUS_CLUSTERING_DIR = Path(
    conf.RESULTS["CLUSTERING_DIR"], "consensus_clustering"
).resolve()

display(CONSENSUS_CLUSTERING_DIR)

# %%
CLUSTERING_LVS_DIR = (
    conf.RESULTS["CLUSTERING_INTERPRETATION"]["BASE_DIR"] / "cluster_lvs" / "part29"
)
display(CLUSTERING_LVS_DIR)
assert CLUSTERING_LVS_DIR.exists()

# %% [markdown] tags=[]
# # Load data

# %% [markdown]
# ## Clustering data

# %%
INPUT_SUBSET = "z_score_std"

# %%
INPUT_STEM = "projection-smultixcan-efo_partial-mashr-zscores"

# %%
input_filepath = Path(
    conf.RESULTS["DATA_TRANSFORMATIONS_DIR"],
    INPUT_SUBSET,
    f"{INPUT_SUBSET}-{INPUT_STEM}.pkl",
).resolve()
display(input_filepath)

assert input_filepath.exists(), "Input file does not exist"

input_filepath_stem = input_filepath.stem
display(input_filepath_stem)

# %%
input_data = pd.read_pickle(input_filepath)

# %%
input_data.shape

# %%
input_data.head()

# %% [markdown]
# ## Clustering partition

# %%
input_file = Path(CONSENSUS_CLUSTERING_DIR, "best_partitions_by_k.pkl").resolve()
display(input_file)

# %%
best_partitions = pd.read_pickle(input_file)

# %%
best_partitions.shape

# %%
best_partitions.head()

# %%
SELECTED_PARTITION = best_partitions.loc[CLUSTERING_K, "partition"]

# %%
SELECTED_PARTITION


# %%
def get_cluster_traits(cluster):
    cluster_traits = input_data[SELECTED_PARTITION == cluster].index

    traits_with_short_codes = [
        Trait.get_traits_from_efo(c)[0].code
        if Trait.is_efo_label(c)
        else Trait.get_trait(full_code=c).code
        for c in cluster_traits
    ]

    return traits_with_short_codes


# %% [markdown]
# ## PhenomeXcan LV-trait associations

# %%
input_filepath = Path(conf.RESULTS["GLS"] / "gls-summary-phenomexcan.pkl.gz")
display(input_filepath)

# %%
phenomexcan_lv_trait_assocs = pd.read_pickle(input_filepath)

# %%
phenomexcan_lv_trait_assocs.shape

# %%
phenomexcan_lv_trait_assocs.head()

# %% [markdown]
# ## eMERGE LV-trait associations

# %%
input_filepath = Path(conf.RESULTS["GLS"] / "gls-summary-emerge.pkl.gz")
display(input_filepath)

# %%
emerge_lv_trait_assocs = pd.read_pickle(input_filepath)

# %%
emerge_lv_trait_assocs.shape

# %%
emerge_lv_trait_assocs.head()

# %% [markdown]
# ## eMERGE traits info

# %%
input_filepath = conf.EMERGE["DESC_FILE_WITH_SAMPLE_SIZE"]
display(input_filepath)

# %%
emerge_traits_info = pd.read_csv(
    input_filepath,
    sep="\t",
    dtype={"phecode": str},
    usecols=[
        "phecode",
        "phenotype",
        "category",
        "eMERGE_III_EUR_case",
        "eMERGE_III_EUR_control",
    ],
)

# %%
emerge_traits_info["phecode"] = emerge_traits_info["phecode"].apply(
    lambda x: f"EUR_{x}"
)

# %%
emerge_traits_info = emerge_traits_info.set_index("phecode").sort_index()

# %%
emerge_traits_info = emerge_traits_info.rename(
    columns={
        "eMERGE_III_EUR_case": "eur_n_cases",
        "eMERGE_III_EUR_control": "eur_n_controls",
    }
)

# %%
emerge_traits_info.shape

# %%
emerge_traits_info.head()

# %%
assert emerge_traits_info.index.is_unique

# %% [markdown]
# ## LVs errors


# %%
lvs_typeIerr = pd.read_pickle(
    conf.RESULTS["GLS_NULL_SIMS"] / "lvs-null_sims-1000g_eur-prop_type_I_errors.pkl"
).set_index("lv")

# %%
lvs_typeIerr.shape

# %%
lvs_typeIerr.head()

# %%
lvs_flagged = set(lvs_typeIerr[lvs_typeIerr["5"] > 0.07].index)

# %%
len(lvs_flagged)

# %% [markdown]
# # Load trait associations


# %% [markdown]
# ## PhenomeXcan

# %%
data_signif = phenomexcan_lv_trait_assocs[
    (~phenomexcan_lv_trait_assocs["lv"].isin(lvs_flagged))
    & (phenomexcan_lv_trait_assocs["fdr"] < 0.05)
]

# %%
data_signif.shape

# %% [markdown]
# ## eMERGE

# %%
data_emerge_signif = emerge_lv_trait_assocs[
    (~emerge_lv_trait_assocs["lv"].isin(lvs_flagged))
    & (emerge_lv_trait_assocs["fdr"] < 0.05)
]

# %%
data_emerge_signif.shape

# %%
# with pd.option_context(
#     "display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None
# ):
#     lv = "LV66"
#     _tmp = data_signif[data_signif["lv"] == lv]
#     display(_tmp)

#     _tmp = data_emerge_signif[data_emerge_signif["lv"] == lv]
#     display(_tmp)

# %%
with pd.option_context(
    "display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None
):
    trait = "schizophrenia"
    _tmp = data_signif[data_signif["phenotype_desc"].str.lower().str.contains(trait)]
    display(_tmp)

    _tmp = data_emerge_signif[data_emerge_signif["lv"] == lv]
    display(_tmp)

# %% [markdown]
# # Autoimmune clusters

# %% [markdown]
# ## PhenomeXcan

# %%
cluster13_lvs = pd.read_pickle(
    CLUSTERING_LVS_DIR / "cluster_interpreter-part29_k13.pkl"
)["name"]

# %%
cluster13_lvs.shape

# %%
cluster13_lvs

# %%
cluster26_lvs = pd.read_pickle(
    CLUSTERING_LVS_DIR / "cluster_interpreter-part29_k26.pkl"
)["name"]

# %%
cluster8_lvs = pd.read_pickle(CLUSTERING_LVS_DIR / "cluster_interpreter-part29_k8.pkl")[
    "name"
]

# %%
all_clusters_lvs = set(cluster13_lvs) | set(cluster26_lvs) | set(cluster8_lvs)

# %%
len(all_clusters_lvs)

# %%
cluster13_traits = get_cluster_traits(13)
cluster26_traits = get_cluster_traits(26)
cluster8_traits = get_cluster_traits(8)

# %%
all_clusters_traits = (
    set(cluster13_traits) | set(cluster26_traits) | set(cluster8_traits)
)

# %%
len(all_clusters_traits)

# %%
groups_signif = data_signif[
    data_signif["lv"].isin(all_clusters_lvs)
    & data_signif["phenotype"].isin(all_clusters_traits)
].sort_values("fdr")

# %%
groups_signif.shape

# %%
groups_signif["lv"].unique()

# %%
groups_signif["phenotype"].unique()

# %% [markdown]
# ## eMERGE

# %%
groups_emerge_signif = data_emerge_signif[
    data_emerge_signif["lv"].isin(all_clusters_lvs)
].sort_values("fdr")

# %%
groups_emerge_signif.shape

# %%
groups_emerge_signif["lv"].unique()

# %% [markdown]
# ## Compare

# %%
# LVs shared
set(groups_signif["lv"].unique()).intersection(set(groups_emerge_signif["lv"].unique()))

# %%
with pd.option_context(
    "display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None
):
    display(groups_signif.shape)
    display(groups_signif)

# %%
with pd.option_context(
    "display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None
):
    display(groups_emerge_signif.shape)
    display(groups_emerge_signif)

# %% [markdown]
# # Cardiovascular clusters

# %% [markdown]
# ## PhenomeXcan

# %%
cluster17_lvs = pd.read_pickle(
    CLUSTERING_LVS_DIR / "cluster_interpreter-part29_k17.pkl"
)["name"]

# %%
cluster17_lvs.shape

# %%
cluster25_lvs = pd.read_pickle(
    CLUSTERING_LVS_DIR / "cluster_interpreter-part29_k25.pkl"
)["name"]

# %%
cluster21_lvs = pd.read_pickle(
    CLUSTERING_LVS_DIR / "cluster_interpreter-part29_k21.pkl"
)["name"]

# %%
cluster28_lvs = pd.read_pickle(
    CLUSTERING_LVS_DIR / "cluster_interpreter-part29_k28.pkl"
)["name"]

# %%
cluster11_lvs = pd.read_pickle(
    CLUSTERING_LVS_DIR / "cluster_interpreter-part29_k11.pkl"
)["name"]

# %%
cluster16_lvs = pd.read_pickle(
    CLUSTERING_LVS_DIR / "cluster_interpreter-part29_k16.pkl"
)["name"]

# %%
all_clusters_lvs = (
    set(cluster17_lvs)
    | set(cluster25_lvs)
    | set(cluster21_lvs)
    | set(cluster28_lvs)
    | set(cluster11_lvs)
    | set(cluster16_lvs)
)

# %%
len(all_clusters_lvs)

# %%
cluster17_traits = get_cluster_traits(17)
cluster25_traits = get_cluster_traits(25)
cluster21_traits = get_cluster_traits(21)
cluster28_traits = get_cluster_traits(28)
cluster11_traits = get_cluster_traits(11)
cluster16_traits = get_cluster_traits(16)

# %%
all_clusters_traits = (
    set(cluster17_traits)
    | set(cluster25_traits)
    | set(cluster21_traits)
    | set(cluster28_traits)
    | set(cluster11_traits)
    | set(cluster16_traits)
)

# %%
len(all_clusters_traits)

# %%
groups_signif = data_signif[
    data_signif["lv"].isin(all_clusters_lvs)
    & data_signif["phenotype"].isin(all_clusters_traits)
].sort_values("fdr")

# %%
groups_signif.shape

# %%
groups_signif["lv"].unique()

# %%
groups_signif["phenotype"].unique()

# %% [markdown]
# ## eMERGE

# %%
groups_emerge_signif = data_emerge_signif[
    data_emerge_signif["lv"].isin(all_clusters_lvs)
].sort_values("fdr")

# %%
groups_emerge_signif.shape

# %%
groups_emerge_signif["lv"].unique()

# %% [markdown]
# ## Compare

# %%
# LVs shared
set(groups_signif["lv"].unique()).intersection(set(groups_emerge_signif["lv"].unique()))

# %%
with pd.option_context(
    "display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None
):
    display(groups_signif.shape)
    display(groups_signif.head(20))

# %%
with pd.option_context(
    "display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None
):
    display(groups_emerge_signif.shape)
    display(groups_emerge_signif)

# %%
with pd.option_context(
    "display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None
):
    lv = "LV434"
    _tmp = groups_signif[groups_signif["lv"] == lv]
    display(_tmp)

    _tmp = groups_emerge_signif[groups_emerge_signif["lv"] == lv]
    display(_tmp)

# %%
with pd.option_context(
    "display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None
):
    lv = "LV434"
    _tmp = groups_signif[groups_signif["lv"] == lv]
    display(_tmp)

    _tmp = groups_emerge_signif[groups_emerge_signif["lv"] == lv]
    display(_tmp)

# %%
