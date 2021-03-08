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
# It creates a text file with mappings for all traits in PhenomeXcan (many of them are from UK Biobank, and a small set of 42 traits are from other studies) to EFO labels. It also adds a category for each trait, which now contains only one category: `disease` (or empty if not categorized).

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
import re
from shutil import copyfile

from IPython.display import display
import pandas as pd
import obonet

import conf
from data.cache import read_data
from entity import Trait, GTEXGWASTrait

# %% [markdown] tags=[]
# # Functions

# %% tags=[]
def get_parents(graph, node):
    for t in graph.successors(node):
        yield t


def _is_disease_single_node(node):
    return node == "EFO:0000408"


def is_disease(graph, node):
    if node not in graph.nodes:
        return False

    if _is_disease_single_node(node):
        return True

    for parent_node in get_parents(graph, node):
        if is_disease(graph, parent_node):
            return True

    return False


# %% [markdown] tags=[]
# # Load EFO Ontology

# %% tags=[]
url = conf.GENERAL["EFO_ONTOLOGY_OBO_FILE"]
graph = obonet.read_obo(url)

# %% tags=[]
# Number of nodes
len(graph)

# %% tags=[]
# Number of edges
graph.number_of_edges()

# %% tags=[]
assert graph.nodes["EFO:0000270"].get("name") == "asthma"

# %% [markdown] tags=[]
# # Load PhenomeXcan traits

# %% tags=[]
phenomexan_traits_names = read_data(
    conf.PHENOMEXCAN["SMULTIXCAN_MASHR_ZSCORES_FILE"]
).columns.tolist()

# %% tags=[]
len(phenomexan_traits_names)

# %% tags=[]
phenomexcan_traits = [Trait.get_trait(full_code=t) for t in phenomexan_traits_names]

# %% tags=[]
assert len(phenomexcan_traits) == len(phenomexan_traits_names)

# %% tags=[]
phenomexcan_code_to_full_code = {t.code: t.get_plain_name() for t in phenomexcan_traits}

# %% tags=[]
assert phenomexcan_code_to_full_code["50_raw"] == "50_raw-Standing_height"

# %% [markdown] tags=[]
# # Load UKB to EFO mappings

# %% tags=[]
ukb_to_efo = read_data(conf.UK_BIOBANK["UKBCODE_TO_EFO_MAP_FILE"])

# %% tags=[]
ukb_to_efo.shape

# %% tags=[]
ukb_to_efo

# %% tags=[]
ukb_to_efo = ukb_to_efo.rename(
    columns={
        "MAPPED_TERM_LABEL": "term_label",
        "MAPPING_TYPE": "mapping_type",
        "MAPPED_TERM_URI": "term_codes",
        "ICD10_CODE/SELF_REPORTED_TRAIT_FIELD_CODE": "ukb_code",
    }
)[["ukb_code", "term_label", "term_codes", "mapping_type"]]

# %% [markdown] tags=[]
# ## Add GTEx GWAS EFO terms

# %% tags=[]
all_gtex_gwas_phenos = [
    p for p in phenomexcan_traits if GTEXGWASTrait.is_phenotype_from_study(p.full_code)
]

# %% tags=[]
_tmp = len(all_gtex_gwas_phenos)
display(_tmp)
assert _tmp == 42

# %% tags=[]
all_gtex_gwas_phenos[:10]

# %% tags=[]
_old_shape = ukb_to_efo.shape

# %% tags=[]
ukb_to_efo = ukb_to_efo.append(
    pd.DataFrame(
        {
            "ukb_code": [ggp.full_code for ggp in all_gtex_gwas_phenos],
            "term_codes": [ggp.orig_efo_id for ggp in all_gtex_gwas_phenos],
        }
    ),
    ignore_index=True,
)

# %% tags=[]
# Fix wrong EFO codes
idx = ukb_to_efo[ukb_to_efo["ukb_code"] == "BCAC_ER_negative_BreastCancer_EUR"].index
ukb_to_efo.loc[idx, "term_codes"] = "EFO_1000650"

idx = ukb_to_efo[ukb_to_efo["ukb_code"] == "CARDIoGRAM_C4D_CAD_ADDITIVE"].index
ukb_to_efo.loc[idx, "term_codes"] = "EFO_0001645"

idx = ukb_to_efo[
    ukb_to_efo["ukb_code"] == "Astle_et_al_2016_Sum_basophil_neutrophil_counts"
].index
ukb_to_efo.loc[idx, "term_codes"] = "EFO_0009388"

idx = ukb_to_efo[
    ukb_to_efo["ukb_code"] == "Astle_et_al_2016_Sum_eosinophil_basophil_counts"
].index
ukb_to_efo.loc[idx, "term_codes"] = "EFO_0009389"

idx = ukb_to_efo[
    ukb_to_efo["ukb_code"] == "Astle_et_al_2016_Sum_neutrophil_eosinophil_counts"
].index
ukb_to_efo.loc[idx, "term_codes"] = "EFO_0009390"

# %% tags=[]
# ukb_to_efo_maps = ukb_to_efo_maps.dropna(subset=['efo_code'])

# %% tags=[]
_tmp = ukb_to_efo.shape
display(_tmp)
assert _tmp[0] == _old_shape[0] + 42

# %% [markdown] tags=[]
# ## Replace values and remove nans

# %% tags=[]
ukb_to_efo = ukb_to_efo.replace(
    {
        "term_codes": {
            "_": ":",
            "HP0011106": "HP:0011106",
        },
    },
    regex=True,
)

# %% tags=[]
ukb_to_efo = ukb_to_efo.dropna(how="all")

# %% tags=[]
ukb_to_efo = ukb_to_efo.dropna(subset=["term_codes"])

# %% tags=[]
assert ukb_to_efo[ukb_to_efo["term_codes"].isna()].shape[0] == 0

# %% tags=[]
assert ukb_to_efo[ukb_to_efo["term_codes"].str.contains("EFO:")].shape[0] > 0

# %% tags=[]
assert ukb_to_efo[ukb_to_efo["term_codes"].str.contains("HP:")].shape[0] > 0

# %% tags=[]
ukb_to_efo.shape

# %% tags=[]
ukb_to_efo.head()


# %% [markdown] tags=[]
# # Add PhenomeXcan code/full code

# %% tags=[]
def _get_fullcode(code):
    if code in phenomexcan_code_to_full_code:
        return phenomexcan_code_to_full_code[code]

    return None


# %% tags=[]
ukb_to_efo = ukb_to_efo.assign(ukb_fullcode=ukb_to_efo["ukb_code"].apply(_get_fullcode))

# %% tags=[]
ukb_to_efo.shape

# %% tags=[]
ukb_to_efo.head()

# %% tags=[]
# remove entries for which we couldn't map a ukb full code
ukb_to_efo = ukb_to_efo.dropna(subset=["ukb_fullcode"])

# %% tags=[]
ukb_to_efo.shape

# %% tags=[]
ukb_to_efo.isna().sum()

# %% tags=[]
# for these ones we need to query the original EFO ontology
ukb_to_efo[ukb_to_efo["term_label"].isna()]

# %% [markdown] tags=[]
# # Load EFO labels and xrefs

# %% tags=[]
term_id_to_label = (
    read_data(conf.GENERAL["TERM_ID_LABEL_FILE"])[["term_id", "label"]]
    .dropna()
    .set_index("term_id")["label"]
    .to_dict()
)

# %% tags=[]
len(term_id_to_label)

# %% tags=[]
# see if efo code with missing label in term_id_to_label is here
assert term_id_to_label["EFO:0009628"] == "abnormal result of function studies"
assert term_id_to_label["EFO:0005606"] == "family history of breast cancer"

# %% tags=[]
assert term_id_to_label["EFO:0004616"] == "osteoarthritis, knee"

# %% tags=[]
# get current labels for old EFO codes
term_id_xrefs = read_data(
    conf.GENERAL["TERM_ID_XREFS_FILE"]
)  # [['label', 'EFO']].dropna().set_index('EFO')[['label']]

# %% tags=[]
term_id_xrefs.dtypes

# %% tags=[]
term_id_xrefs.shape

# %% tags=[]
term_id_xrefs.head()

# %% tags=[]
# see if for an old efo code we get the current efo label
new_efo_code = term_id_xrefs[term_id_xrefs["target_id"] == "EFO:1000673"].index[0]
display(new_efo_code)
assert term_id_to_label[new_efo_code] == "autoimmune bullous skin disease"

# %% [markdown] tags=[]
# # Add new EFO label

# %% [markdown] tags=[]
# ## Functions

# %% tags=[]
term_pattern = re.compile(r"\w+:\w+")


def _add_term_labels(row):
    term_ids = row["term_codes"]

    matches = term_pattern.findall(term_ids)

    labels = []
    for m in matches:
        if m in term_id_to_label:
            new_label = term_id_to_label[m]
        else:
            other_xrefs = term_id_xrefs[term_id_xrefs["target_id"] == m]
            if other_xrefs.shape[0] == 1:
                new_label = term_id_to_label[other_xrefs.index[0]]
            elif other_xrefs.shape[0] > 1:
                new_label = term_id_to_label[other_xrefs.index[0]]
            elif not pd.isnull(row["term_label"]):
                new_label = row["term_label"]
            else:
                continue

        labels.append(new_label.lower())

    return " AND ".join(labels)


# %% [markdown] tags=[]
# ## Apply

# %% tags=[]
ukb_to_efo = ukb_to_efo.assign(
    current_term_label=ukb_to_efo.apply(_add_term_labels, axis=1)
)

# %% tags=[]
ukb_to_efo.shape

# %% tags=[]
ukb_to_efo.head()


# %% [markdown] tags=[]
# # Add categories

# %% [markdown] tags=[]
# It only adds Disease for now

# %% tags=[]
def _get_disease_category(row):
    term_ids = row["term_codes"]

    matches = term_pattern.findall(term_ids)

    for m in matches:
        if is_disease(graph, m):
            return "disease"

    return None


# %% tags=[]
ukb_to_efo = ukb_to_efo.assign(category=ukb_to_efo.apply(_get_disease_category, axis=1))

# %% tags=[]
ukb_to_efo.head()

# %% tags=[]
ukb_to_efo[ukb_to_efo["category"] == "disease"].shape

# %% tags=[]
_tmp = ukb_to_efo[ukb_to_efo["category"] == "disease"]
_tmp["current_term_label"].value_counts()

# %% [markdown] tags=[]
# # Testing

# %% tags=[]
# asthma exists
_tmp = ukb_to_efo[ukb_to_efo["current_term_label"].str.lower().str.contains("asthma")]
display(_tmp)
assert _tmp.shape[0] >= 4

# %% tags=[]
# check if old EFO labels are updated in orig_efo_names
# _tmp = ukb_to_efo.dropna()
_tmp = ukb_to_efo[ukb_to_efo["term_codes"].str.contains("EFO:1000673")]
display(_tmp)
_tmp = _tmp.iloc[0]
assert _tmp["term_label"] == "bullous skin disease"
assert _tmp["current_term_label"] == "autoimmune bullous skin disease"

# %% tags=[]
_tmp = ukb_to_efo.isna().sum()
assert _tmp.loc["ukb_fullcode"] == 0
assert _tmp.loc["term_codes"] == 0
assert _tmp.loc["current_term_label"] == 0

# %% tags=[]
# check all nan term labels now have non-empty current labels
_tmp = ukb_to_efo[ukb_to_efo["term_label"].isna()]
display(_tmp)
assert _tmp[_tmp["current_term_label"].isna()].shape[0] == 0
assert _tmp[_tmp["current_term_label"].str.strip().str.len() == 0].shape[0] == 0

# %% tags=[]
ukb_to_efo[(ukb_to_efo["current_term_label"].str.strip().str.len() == 0)]

# %% tags=[]
# check there are no null/empty current term labels
assert (
    ukb_to_efo[
        ukb_to_efo["current_term_label"].isna()
        | (ukb_to_efo["current_term_label"].str.strip().str.len() == 0)
    ].shape[0]
    == 0
)

# %% tags=[]
# How many entries have an term_label that differs from current_term_label?
ukb_to_efo[ukb_to_efo["term_label"] != ukb_to_efo["current_term_label"]]

# %% tags=[]
# How many entries comprise more than one EFO codes?
_tmp = ukb_to_efo[ukb_to_efo["current_term_label"].str.contains(" AND ")]
display(_tmp.shape)
display(_tmp)

# %% tags=[]
ukb_to_efo["mapping_type"].value_counts()

# %% tags=[]
ukb_to_efo[ukb_to_efo["mapping_type"] == "Exact"]["current_term_label"].value_counts()

# %% tags=[]
ukb_to_efo["current_term_label"].value_counts()

# %% tags=[]
ukb_to_efo[ukb_to_efo["current_term_label"] == "emotional symptom measurement"]

# %% tags=[]
ukb_to_efo[ukb_to_efo["ukb_fullcode"].duplicated(False)]

# %% tags=[]
ukb_to_efo[ukb_to_efo["ukb_fullcode"].duplicated(False)]["ukb_fullcode"].tolist()

# %% tags=[]
# Fix duplicated ukb_fullcode entries with different current_term_labels
idx = ukb_to_efo[
    ukb_to_efo["ukb_fullcode"]
    == "T79-Diagnoses_main_ICD10_T79_Certain_early_complications_of_trauma_not_elsewhere_classified"
].index
ukb_to_efo.loc[idx, "current_term_label"] = "complication"

# %% tags=[]
assert ukb_to_efo[ukb_to_efo["mapping_type"] == "Exact"]["ukb_fullcode"].is_unique

# %% tags=[]
_tmp = (
    ukb_to_efo.groupby(["ukb_fullcode", "current_term_label"])
    .count()
    .reset_index()[["ukb_fullcode", "current_term_label"]]
)
assert not _tmp.duplicated().any()

# %% [markdown] tags=[]
# # Save

# %% [markdown] tags=[]
# ## In main data folder

# %% tags=[]
outfile = conf.PHENOMEXCAN["TRAITS_FULLCODE_TO_EFO_MAP_FILE"]
display(outfile)

ukb_to_efo.to_csv(outfile, sep="\t", index=False)

# %% [markdown] tags=[]
# ## In libs/data folder

# %% [markdown] tags=[]
# Since this file (`outfile`) is used by the Trait class to return EFO codes/labels for PhenomeXcan traits, it is copied also to a source code folder and it is supposed to be versioned.

# %% tags=[]
display(Trait.UKB_TO_EFO_MAP_FILE)

# %% tags=[]
copyfile(
    outfile,
    Trait.UKB_TO_EFO_MAP_FILE,
)

# %% tags=[]
