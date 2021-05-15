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

# %% [markdown]
# TODO

# %% [markdown] tags=[]
# # Environment variables

# %% tags=[]
# from IPython.display import display

# import conf

# N_JOBS = conf.GENERAL["N_JOBS"]
# display(N_JOBS)

# %% tags=[]
# # %env MKL_NUM_THREADS=$N_JOBS
# # %env OPEN_BLAS_NUM_THREADS=$N_JOBS
# # %env NUMEXPR_NUM_THREADS=$N_JOBS
# # %env OMP_NUM_THREADS=$N_JOBS

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
from pathlib import Path

import numpy as np
import pandas as pd

import conf
from data.cache import read_data

# %% [markdown] tags=[]
# # Settings

# %%
# SELECTED_PARTITION_K = 44

# %%
# CONSENSUS_CLUSTERING_DIR = Path(
#     conf.RESULTS["CLUSTERING_DIR"], "consensus_clustering"
# ).resolve()

# display(CONSENSUS_CLUSTERING_DIR)

# %% [markdown]
# # Data loading

# %% [markdown]
# ## eMERGE S-MultiXcan results

# %%
# FIXME: path hardcoded
input_filepath = Path(
    "/home/miltondp/projects/labs/greenelab/phenoplier/base/data/emerge/gene_assoc/emerge-smultixcan-mashr-zscores.pkl"
).resolve()

# %% tags=[]
# input_filepath = Path(
#     conf.RESULTS["PROJECTIONS_DIR"],
#     "projection-emerge-smultixcan-mashr-zscores.pkl",
# ).resolve()
# display(input_filepath)
# assert input_filepath.exists()

# %%
pmbb_data = pd.read_pickle(input_filepath)

# %%
pmbb_data.shape

# %%
pmbb_data.head()

# %% [markdown] tags=[]
# ## PhenomeXcan S-MultiXcan results

# %%
input_filepath = conf.PHENOMEXCAN["SMULTIXCAN_EFO_PARTIAL_MASHR_ZSCORES_FILE"]
display(input_filepath)

# %% tags=[]
phenomexcan_data = pd.read_pickle(input_filepath)

# %% tags=[]
phenomexcan_data.shape

# %% tags=[]
phenomexcan_data.head()

# %% [markdown]
# ## PhenomeXcan traits info

# %%
phenomexcan_traits_info = read_data(conf.PHENOMEXCAN["RAPID_GWAS_PHENO_INFO_FILE"])

# %%
phenomexcan_traits_info.shape

# %%
phenomexcan_traits_info.head()

# %%
# read_data(conf.PHENOMEXCAN["GTEX_GWAS_PHENO_INFO_FILE"])

# %% [markdown]
# ## PMBB/eMERGE phecodes mappings to ICD10

# %%
# FIXME: hardcoded
# input_file = conf.PMBB["ICD10_PHECODE_MAPPING_FILE"]
input_file = Path(
    "/home/miltondp/projects/labs/greenelab/phenoplier/base/data/pmbb",
    "Phecode_to_UKB_ICD10_mapping.tsv",
).resolve()
display(input_file)

# %%
icd10_to_phecodes_map = pd.read_csv(input_file, sep="\t", dtype={"Phecode": str})

# %%
icd10_to_phecodes_map.shape

# %%
icd10_to_phecodes_map.head()

# %%
icd10_to_phecodes_map["Phecode"].is_unique

# %%
icd10_to_phecodes_map["PhenomeXcan_ICD10"].is_unique

# %% [markdown]
# The same ICD10 code from PhenomeXcan can map to multiple phecodes, and one phecode can belong to several ICD10 codes.

# %% [markdown]
# ## Phecodes descriptions (and sample size)

# %%
# FIXME: hardcoded
input_filepath = Path(
    "/home/miltondp/projects/labs/greenelab/phenoplier/base/data/emerge",
    "eMERGE_III_PMBB_GSA_v2_2020_phecode_AFR_EUR_cc50_counts_w_dictionary.txt",
).resolve()
display(input_filepath)

# %%
pmbb_traits_df = pd.read_csv(
    input_filepath,
    sep="\t",
    dtype={"phecode": str},
    usecols=["phecode", "phenotype", "category"],
)

# %%
pmbb_traits_df = pmbb_traits_df.rename(
    columns={
        "phenotype": "phecode_phenotype",
        "category": "phecode_category",
    }
)

# %%
pmbb_traits_df.shape

# %%
pmbb_traits_df.head()

# %% [markdown]
# ## PhenomeXcan-EFO mappings

# %%
phenomexcan_to_efo = read_data(conf.PHENOMEXCAN["TRAITS_FULLCODE_TO_EFO_MAP_FILE"])

# %%
phenomexcan_to_efo.shape

# %%
phenomexcan_to_efo.head()

# %% [markdown]
# # How many PhenomeXcan traits are in eMERGE results?

# %%
phenomexcan_icd10_traits = phenomexcan_traits_info[
    phenomexcan_traits_info["source"] == "icd10"
]

# %%
phenomexcan_icd10_traits.shape

# %%
phenomexcan_icd10_traits.head()

# %%
# See how many traits
icd10_in_phecodes = icd10_to_phecodes_map[
    icd10_to_phecodes_map["PhenomeXcan_ICD10"].isin(phenomexcan_icd10_traits.index)
]

# %%
icd10_in_phecodes.shape

# %% [markdown]
# 224 PhenomeXcan's traits from ICD10 have phecodes

# %%
icd10_in_phecodes.head()

# %%
# Which entries in the icd10-phecode mapping are not present in PhenomeXcan?
icd10_to_phecodes_map[
    ~icd10_to_phecodes_map["PhenomeXcan_ICD10"].isin(phenomexcan_icd10_traits.index)
]

# %% [markdown]
# # Map ICD10 to EFO in PhenomeXcan

# %%
# get the EFO map of PhenomeXcan's traits with ICD10 that have phecodes
efo_from_icd10 = phenomexcan_to_efo[
    phenomexcan_to_efo["ukb_code"].isin(icd10_in_phecodes["PhenomeXcan_ICD10"].values)
]

# %%
efo_from_icd10

# %% [markdown]
# 199 PhenomeXcan traits can be mapped to phecodes when we use the PhenomeXcan-EFO mappings. These mappings combine different PhenomeXcan traits, such as ICD10 codes and self-reported data.

# %% [markdown]
# # Create initial dataframe for test set

# %% [markdown]
# We create a test set with all 309 PMBB's traits available. It is important to include them all, since some ICD10 are mapped to several different phecodes, and one phecodes can represent different ICD10 codes.

# %%
test_df = pmbb_traits_df.copy()

# %%
test_df.shape

# %%
test_df.head()


# %% [markdown]
# ## Phecode to ICD10

# %%
def map_phecode_to_icd10_list(phecode):
    df = icd10_to_phecodes_map[icd10_to_phecodes_map["Phecode"] == phecode]
    if df.shape[0] == 0:
        return None

    return set(df["PhenomeXcan_ICD10"].values)


# %%
# some testing
assert map_phecode_to_icd10_list("008") == set(["A09"])
assert map_phecode_to_icd10_list("244.2") == set(["E03"])
assert map_phecode_to_icd10_list("244.4") == set(["E03"])

assert map_phecode_to_icd10_list("216") == set(["D22", "D23"])

assert map_phecode_to_icd10_list("722") is None

# %%
_new_df = []

for phecode in test_df["phecode"].unique():
    _new_map_values = map_phecode_to_icd10_list(phecode)
    if _new_map_values is None:
        continue

    for n in _new_map_values:
        _new_df.append({"phecode": phecode, "icd10": n})

_new_df = pd.DataFrame(_new_df)

# %%
# assign phecode
test_df = pd.merge(test_df, _new_df, on="phecode", how="left")

# %%
test_df.shape

# %%
test_df.head()


# %% [markdown]
# ## ICD10 to EFO

# %%
def map_icd10_to_efo(icd10_list):
    if icd10_list is None:
        return None

    df = efo_from_icd10[efo_from_icd10["ukb_code"] == icd10_list]
    efo_values = None

    if df.shape[0] > 0:
        efo_values = set(df["current_term_label"].values)

    return efo_values


# %%
# some testing
assert map_icd10_to_efo("M17") == set(["osteoarthritis, knee"])
assert map_icd10_to_efo("R30") == set(["dysuria"])
assert map_icd10_to_efo("I49") == set(["cardiac arrhythmia"])
assert map_icd10_to_efo("A41") == set(["sepsis"])

assert map_icd10_to_efo("T81") == set(["complication"])
assert map_icd10_to_efo("T88") == set(["complication"])

assert map_icd10_to_efo("C61") == set(["prostate carcinoma"])
assert map_icd10_to_efo("D07") == set(["urogenital neoplasm"])

assert map_icd10_to_efo(None) is None
assert map_icd10_to_efo("NonExistent") is None

# %%
_new_df = []

for phecode in test_df["icd10"].unique():
    _new_map_values = map_icd10_to_efo(phecode)
    if _new_map_values is None:
        continue

    for n in _new_map_values:
        _new_df.append({"icd10": phecode, "efo": n})

_new_df = pd.DataFrame(_new_df)

# %%
# assign phecode
test_df = pd.merge(test_df, _new_df, on="icd10", how="left")

# %%
test_df = test_df.dropna(subset=["icd10", "efo"], how="all")

# %%
test_df = test_df.drop_duplicates(subset=["icd10", "efo"])

# %%
test_df.shape

# %%
test_df.head()

# %% [markdown]
# ## ICD10/EFO to PhenomeXcan traits

# %%
assert test_df[pd.isnull(test_df["icd10"])].shape[0] == 0

# %%
test_df[pd.isnull(test_df["efo"])]

# %%
with pd.option_context(
    "display.max_rows", None, "display.max_columns", None, "max_colwidth", None
):
    _tmp = (
        efo_from_icd10.head()
    )  # [efo_from_icd10["current_term_label"] == "osteoarthritis, knee"]
    display(_tmp)


# %%
def map_to_phenomexcan_trait(row):
    icd10_list = row["icd10"]
    if pd.isnull(icd10_list):
        icd10_list = set()

    # get EFO from ICD10 hits
    # this has the effect to always find the same set of PhenomeXcan traits
    # regardness of whether only the icd10 or only the efo labels are given)
    icd10_hits = phenomexcan_to_efo[(phenomexcan_to_efo["ukb_code"] == icd10_list)][
        "current_term_label"
    ].tolist()

    efo_list = row["efo"]
    if pd.isnull(efo_list):
        efo_list = set()

    # get ICD10 from EFO hits
    efo_hits = phenomexcan_to_efo[
        (phenomexcan_to_efo["current_term_label"] == efo_list)
    ]["ukb_code"].tolist()

    phenomexcan_maps = phenomexcan_to_efo[
        (phenomexcan_to_efo["ukb_code"].isin([icd10_list] + efo_hits))
        | (phenomexcan_to_efo["current_term_label"].isin([efo_hits] + icd10_hits))
    ]

    if phenomexcan_maps.shape[0] == 0:
        return None

    return set(phenomexcan_maps["ukb_fullcode"].unique())


# %%
# some testing
row = {"icd10": "M17", "efo": None}
assert map_to_phenomexcan_trait(row) == set(
    ["M17-Diagnoses_main_ICD10_M17_Gonarthrosis_arthrosis_of_knee"]
)
row = {"icd10": None, "efo": "osteoarthritis, knee"}
assert map_to_phenomexcan_trait(row) == set(
    ["M17-Diagnoses_main_ICD10_M17_Gonarthrosis_arthrosis_of_knee"]
)
row = {"icd10": "M17", "efo": "osteoarthritis, knee"}
assert map_to_phenomexcan_trait(row) == set(
    ["M17-Diagnoses_main_ICD10_M17_Gonarthrosis_arthrosis_of_knee"]
)

row = {"icd10": None, "efo": "coronary artery disease"}
assert map_to_phenomexcan_trait(row) == set(
    [
        "CARDIoGRAM_C4D_CAD_ADDITIVE",
        "I25-Diagnoses_main_ICD10_I25_Chronic_ischaemic_heart_disease",
    ]
)
row = {"icd10": "I25", "efo": None}
assert map_to_phenomexcan_trait(row) == set(
    [
        "CARDIoGRAM_C4D_CAD_ADDITIVE",
        "I25-Diagnoses_main_ICD10_I25_Chronic_ischaemic_heart_disease",
    ]
)
row = {"icd10": "I25", "efo": "coronary artery disease"}
assert map_to_phenomexcan_trait(row) == set(
    [
        "CARDIoGRAM_C4D_CAD_ADDITIVE",
        "I25-Diagnoses_main_ICD10_I25_Chronic_ischaemic_heart_disease",
    ]
)

row = {"icd10": None, "efo": None}
assert map_to_phenomexcan_trait(row) is None

# %%
_new_df = []

for idx, row in test_df.iterrows():
    _new_map_values = map_to_phenomexcan_trait(row)
    if _new_map_values is None:
        continue

    for n in _new_map_values:
        _new_df.append({"icd10": row["icd10"], "efo": row["efo"], "phenomexcan": n})

_new_df = pd.DataFrame(_new_df)

# %%
assert (
    _new_df[
        _new_df["icd10"].isin(test_df[pd.isnull(test_df["efo"])]["icd10"].tolist())
    ].shape[0]
    == 0
)

# %%
# assign phenomexcan trait
test_df = pd.merge(test_df, _new_df, on=["icd10", "efo"], how="left")

# %%
_tmp = test_df[test_df["icd10"] == "I25"]
display(_tmp)
assert _tmp.shape[0] == 2

# %%
test_df.shape

# %%
test_df.head()

# %% [markdown]
# # Save

# %%
# FIXME: uncomment when merged with emerge branch
# conf.RESULTS["EMERGE_VALIDATION_DIR"].mkdir(exist_ok=True, parents=True)

# %% tags=[]
# FIXME: hardcoded path
# output_file = Path(
#     conf.RESULTS["EMERGE_VALIDATION_DIR"],
#     "emerge-test_set.pkl",
# ).resolve()
output_file = Path(
    "/home/miltondp/projects/labs/greenelab/phenoplier/base/data/emerge",
    "phecodes_phenomexcan_maps.tsv",
).resolve()

display(output_file)

# %% tags=[]
test_df.to_csv(output_file, sep="\t", index=False)

# %%
