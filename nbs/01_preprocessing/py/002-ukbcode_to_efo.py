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

# %% [markdown] papermill={"duration": 0.034842, "end_time": "2020-12-17T20:01:25.725418", "exception": false, "start_time": "2020-12-17T20:01:25.690576", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.030839, "end_time": "2020-12-17T20:01:25.787348", "exception": false, "start_time": "2020-12-17T20:01:25.756509", "status": "completed"} tags=[]
# It creates a text file with mappings for all traits in PhenomeXcan (many of them are from UK Biobank, and a small set of 42 traits are from other studies) to EFO labels. It also adds a category for each trait, which now contains only one category: `disease` (or empty if not categorized).

# %% [markdown] papermill={"duration": 0.030965, "end_time": "2020-12-17T20:01:25.848807", "exception": false, "start_time": "2020-12-17T20:01:25.817842", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.041664, "end_time": "2020-12-17T20:01:25.921422", "exception": false, "start_time": "2020-12-17T20:01:25.879758", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 0.571741, "end_time": "2020-12-17T20:01:26.524769", "exception": false, "start_time": "2020-12-17T20:01:25.953028", "status": "completed"} tags=[]
import re
from shutil import copyfile

from IPython.display import display
import pandas as pd
import obonet

import conf
from data.cache import read_data
from entity import Trait, GTEXGWASTrait

# %% [markdown] papermill={"duration": 0.030796, "end_time": "2020-12-17T20:01:26.589510", "exception": false, "start_time": "2020-12-17T20:01:26.558714", "status": "completed"} tags=[]
# # Functions

# %% papermill={"duration": 0.043912, "end_time": "2020-12-17T20:01:26.664018", "exception": false, "start_time": "2020-12-17T20:01:26.620106", "status": "completed"} tags=[]
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


# %% [markdown] papermill={"duration": 0.031296, "end_time": "2020-12-17T20:01:26.726979", "exception": false, "start_time": "2020-12-17T20:01:26.695683", "status": "completed"} tags=[]
# # Load EFO Ontology

# %% papermill={"duration": 4.857699, "end_time": "2020-12-17T20:01:31.616228", "exception": false, "start_time": "2020-12-17T20:01:26.758529", "status": "completed"} tags=[]
url = conf.GENERAL["EFO_ONTOLOGY_OBO_FILE"]
graph = obonet.read_obo(url)

# %% papermill={"duration": 0.047403, "end_time": "2020-12-17T20:01:31.702406", "exception": false, "start_time": "2020-12-17T20:01:31.655003", "status": "completed"} tags=[]
# Number of nodes
len(graph)

# %% papermill={"duration": 0.078905, "end_time": "2020-12-17T20:01:31.812727", "exception": false, "start_time": "2020-12-17T20:01:31.733822", "status": "completed"} tags=[]
# Number of edges
graph.number_of_edges()

# %% papermill={"duration": 0.04272, "end_time": "2020-12-17T20:01:31.890675", "exception": false, "start_time": "2020-12-17T20:01:31.847955", "status": "completed"} tags=[]
assert graph.nodes["EFO:0000270"].get("name") == "asthma"

# %% [markdown] papermill={"duration": 0.030726, "end_time": "2020-12-17T20:01:31.953948", "exception": false, "start_time": "2020-12-17T20:01:31.923222", "status": "completed"} tags=[]
# # Load PhenomeXcan traits

# %% papermill={"duration": 0.306715, "end_time": "2020-12-17T20:01:32.291384", "exception": false, "start_time": "2020-12-17T20:01:31.984669", "status": "completed"} tags=[]
phenomexan_traits_names = read_data(
    conf.PHENOMEXCAN["SMULTIXCAN_MASHR_ZSCORES_FILE"]
).columns.tolist()

# %% papermill={"duration": 0.043385, "end_time": "2020-12-17T20:01:32.368967", "exception": false, "start_time": "2020-12-17T20:01:32.325582", "status": "completed"} tags=[]
len(phenomexan_traits_names)

# %% papermill={"duration": 0.541655, "end_time": "2020-12-17T20:01:32.941571", "exception": false, "start_time": "2020-12-17T20:01:32.399916", "status": "completed"} tags=[]
phenomexcan_traits = [Trait.get_trait(full_code=t) for t in phenomexan_traits_names]

# %% papermill={"duration": 0.042574, "end_time": "2020-12-17T20:01:33.016205", "exception": false, "start_time": "2020-12-17T20:01:32.973631", "status": "completed"} tags=[]
assert len(phenomexcan_traits) == len(phenomexan_traits_names)

# %% papermill={"duration": 0.06318, "end_time": "2020-12-17T20:01:33.110359", "exception": false, "start_time": "2020-12-17T20:01:33.047179", "status": "completed"} tags=[]
phenomexcan_code_to_full_code = {t.code: t.get_plain_name() for t in phenomexcan_traits}

# %% papermill={"duration": 0.042185, "end_time": "2020-12-17T20:01:33.184112", "exception": false, "start_time": "2020-12-17T20:01:33.141927", "status": "completed"} tags=[]
assert phenomexcan_code_to_full_code["50_raw"] == "50_raw-Standing_height"

# %% [markdown] papermill={"duration": 0.03135, "end_time": "2020-12-17T20:01:33.246622", "exception": false, "start_time": "2020-12-17T20:01:33.215272", "status": "completed"} tags=[]
# # Load UKB to EFO mappings

# %% papermill={"duration": 0.047078, "end_time": "2020-12-17T20:01:33.327199", "exception": false, "start_time": "2020-12-17T20:01:33.280121", "status": "completed"} tags=[]
ukb_to_efo = read_data(conf.UK_BIOBANK["UKBCODE_TO_EFO_MAP_FILE"])

# %% papermill={"duration": 0.043362, "end_time": "2020-12-17T20:01:33.402310", "exception": false, "start_time": "2020-12-17T20:01:33.358948", "status": "completed"} tags=[]
ukb_to_efo.shape

# %% papermill={"duration": 0.051304, "end_time": "2020-12-17T20:01:33.486070", "exception": false, "start_time": "2020-12-17T20:01:33.434766", "status": "completed"} tags=[]
ukb_to_efo

# %% papermill={"duration": 0.044048, "end_time": "2020-12-17T20:01:33.563074", "exception": false, "start_time": "2020-12-17T20:01:33.519026", "status": "completed"} tags=[]
ukb_to_efo = ukb_to_efo.rename(
    columns={
        "MAPPED_TERM_LABEL": "term_label",
        "MAPPING_TYPE": "mapping_type",
        "MAPPED_TERM_URI": "term_codes",
        "ICD10_CODE/SELF_REPORTED_TRAIT_FIELD_CODE": "ukb_code",
    }
)[["ukb_code", "term_label", "term_codes", "mapping_type"]]

# %% [markdown] papermill={"duration": 0.031342, "end_time": "2020-12-17T20:01:33.626145", "exception": false, "start_time": "2020-12-17T20:01:33.594803", "status": "completed"} tags=[]
# ## Add GTEx GWAS EFO terms

# %% papermill={"duration": 0.046127, "end_time": "2020-12-17T20:01:33.703853", "exception": false, "start_time": "2020-12-17T20:01:33.657726", "status": "completed"} tags=[]
all_gtex_gwas_phenos = [
    p for p in phenomexcan_traits if GTEXGWASTrait.is_phenotype_from_study(p.full_code)
]

# %% papermill={"duration": 0.044355, "end_time": "2020-12-17T20:01:33.779960", "exception": false, "start_time": "2020-12-17T20:01:33.735605", "status": "completed"} tags=[]
_tmp = len(all_gtex_gwas_phenos)
display(_tmp)
assert _tmp == 42

# %% papermill={"duration": 0.045002, "end_time": "2020-12-17T20:01:33.857426", "exception": false, "start_time": "2020-12-17T20:01:33.812424", "status": "completed"} tags=[]
all_gtex_gwas_phenos[:10]

# %% papermill={"duration": 0.043162, "end_time": "2020-12-17T20:01:33.933833", "exception": false, "start_time": "2020-12-17T20:01:33.890671", "status": "completed"} tags=[]
_old_shape = ukb_to_efo.shape

# %% papermill={"duration": 0.046175, "end_time": "2020-12-17T20:01:34.012966", "exception": false, "start_time": "2020-12-17T20:01:33.966791", "status": "completed"} tags=[]
ukb_to_efo = ukb_to_efo.append(
    pd.DataFrame(
        {
            "ukb_code": [ggp.full_code for ggp in all_gtex_gwas_phenos],
            "term_codes": [ggp.orig_efo_id for ggp in all_gtex_gwas_phenos],
        }
    ),
    ignore_index=True,
)

# %% papermill={"duration": 0.046475, "end_time": "2020-12-17T20:01:34.091300", "exception": false, "start_time": "2020-12-17T20:01:34.044825", "status": "completed"} tags=[]
# Fix wrong EFO codes
idx = ukb_to_efo[ukb_to_efo["ukb_code"] == "BCAC_ER_negative_BreastCancer_EUR"].index
ukb_to_efo.loc[idx, "term_codes"] = "EFO_1000650"

idx = ukb_to_efo[ukb_to_efo["ukb_code"] == "CARDIoGRAM_C4D_CAD_ADDITIVE"].index
ukb_to_efo.loc[idx, "term_codes"] = "EFO_0001645"

# %% papermill={"duration": 0.043406, "end_time": "2020-12-17T20:01:34.167926", "exception": false, "start_time": "2020-12-17T20:01:34.124520", "status": "completed"} tags=[]
# ukb_to_efo_maps = ukb_to_efo_maps.dropna(subset=['efo_code'])

# %% papermill={"duration": 0.044387, "end_time": "2020-12-17T20:01:34.244473", "exception": false, "start_time": "2020-12-17T20:01:34.200086", "status": "completed"} tags=[]
_tmp = ukb_to_efo.shape
display(_tmp)
assert _tmp[0] == _old_shape[0] + 42

# %% [markdown] papermill={"duration": 0.032357, "end_time": "2020-12-17T20:01:34.311841", "exception": false, "start_time": "2020-12-17T20:01:34.279484", "status": "completed"} tags=[]
# ## Replace values and remove nans

# %% papermill={"duration": 0.0491, "end_time": "2020-12-17T20:01:34.393470", "exception": false, "start_time": "2020-12-17T20:01:34.344370", "status": "completed"} tags=[]
ukb_to_efo = ukb_to_efo.replace(
    {
        "term_codes": {
            #             '\|\|': ', ',
            "_": ":",
            "HP0011106": "HP:0011106",
        },
        #         'efo_name': {
        #             '\|\|': ', ',
        #         }
    },
    regex=True,
)

# %% papermill={"duration": 0.046403, "end_time": "2020-12-17T20:01:34.473661", "exception": false, "start_time": "2020-12-17T20:01:34.427258", "status": "completed"} tags=[]
ukb_to_efo = ukb_to_efo.dropna(how="all")

# %% papermill={"duration": 0.045483, "end_time": "2020-12-17T20:01:34.552196", "exception": false, "start_time": "2020-12-17T20:01:34.506713", "status": "completed"} tags=[]
ukb_to_efo = ukb_to_efo.dropna(subset=["term_codes"])

# %% papermill={"duration": 0.048554, "end_time": "2020-12-17T20:01:34.652194", "exception": false, "start_time": "2020-12-17T20:01:34.603640", "status": "completed"} tags=[]
assert ukb_to_efo[ukb_to_efo["term_codes"].isna()].shape[0] == 0

# %% papermill={"duration": 0.04517, "end_time": "2020-12-17T20:01:34.731098", "exception": false, "start_time": "2020-12-17T20:01:34.685928", "status": "completed"} tags=[]
assert ukb_to_efo[ukb_to_efo["term_codes"].str.contains("EFO:")].shape[0] > 0

# %% papermill={"duration": 0.045098, "end_time": "2020-12-17T20:01:34.809244", "exception": false, "start_time": "2020-12-17T20:01:34.764146", "status": "completed"} tags=[]
assert ukb_to_efo[ukb_to_efo["term_codes"].str.contains("HP:")].shape[0] > 0

# %% papermill={"duration": 0.045071, "end_time": "2020-12-17T20:01:34.887157", "exception": false, "start_time": "2020-12-17T20:01:34.842086", "status": "completed"} tags=[]
ukb_to_efo.shape

# %% papermill={"duration": 0.047989, "end_time": "2020-12-17T20:01:34.968964", "exception": false, "start_time": "2020-12-17T20:01:34.920975", "status": "completed"} tags=[]
ukb_to_efo.head()


# %% [markdown] papermill={"duration": 0.032287, "end_time": "2020-12-17T20:01:35.034489", "exception": false, "start_time": "2020-12-17T20:01:35.002202", "status": "completed"} tags=[]
# # Add PhenomeXcan code/full code

# %% papermill={"duration": 0.043687, "end_time": "2020-12-17T20:01:35.111083", "exception": false, "start_time": "2020-12-17T20:01:35.067396", "status": "completed"} tags=[]
def _get_fullcode(code):
    if code in phenomexcan_code_to_full_code:
        return phenomexcan_code_to_full_code[code]

    return None


# %% papermill={"duration": 0.0458, "end_time": "2020-12-17T20:01:35.189980", "exception": false, "start_time": "2020-12-17T20:01:35.144180", "status": "completed"} tags=[]
ukb_to_efo = ukb_to_efo.assign(ukb_fullcode=ukb_to_efo["ukb_code"].apply(_get_fullcode))

# %% papermill={"duration": 0.047405, "end_time": "2020-12-17T20:01:35.271743", "exception": false, "start_time": "2020-12-17T20:01:35.224338", "status": "completed"} tags=[]
ukb_to_efo.shape

# %% papermill={"duration": 0.049349, "end_time": "2020-12-17T20:01:35.355083", "exception": false, "start_time": "2020-12-17T20:01:35.305734", "status": "completed"} tags=[]
ukb_to_efo.head()

# %% papermill={"duration": 0.046273, "end_time": "2020-12-17T20:01:35.435194", "exception": false, "start_time": "2020-12-17T20:01:35.388921", "status": "completed"} tags=[]
# remove entries for which we couldn't map a ukb full code
ukb_to_efo = ukb_to_efo.dropna(subset=["ukb_fullcode"])

# %% papermill={"duration": 0.168633, "end_time": "2020-12-17T20:01:35.637406", "exception": false, "start_time": "2020-12-17T20:01:35.468773", "status": "completed"} tags=[]
ukb_to_efo.shape

# %% papermill={"duration": 0.049726, "end_time": "2020-12-17T20:01:35.737457", "exception": false, "start_time": "2020-12-17T20:01:35.687731", "status": "completed"} tags=[]
ukb_to_efo.isna().sum()

# %% papermill={"duration": 0.054065, "end_time": "2020-12-17T20:01:35.826835", "exception": false, "start_time": "2020-12-17T20:01:35.772770", "status": "completed"} tags=[]
# for these ones we need to query the original EFO ontology
ukb_to_efo[ukb_to_efo["term_label"].isna()]

# %% [markdown] papermill={"duration": 0.034458, "end_time": "2020-12-17T20:01:35.895993", "exception": false, "start_time": "2020-12-17T20:01:35.861535", "status": "completed"} tags=[]
# # Load EFO labels and xrefs

# %% papermill={"duration": 0.093885, "end_time": "2020-12-17T20:01:36.024370", "exception": false, "start_time": "2020-12-17T20:01:35.930485", "status": "completed"} tags=[]
term_id_to_label = (
    read_data(conf.GENERAL["TERM_ID_LABEL_FILE"])[["term_id", "label"]]
    .dropna()
    .set_index("term_id")["label"]
    .to_dict()
)

# %% papermill={"duration": 0.047036, "end_time": "2020-12-17T20:01:36.112076", "exception": false, "start_time": "2020-12-17T20:01:36.065040", "status": "completed"} tags=[]
len(term_id_to_label)

# %% papermill={"duration": 0.046463, "end_time": "2020-12-17T20:01:36.194475", "exception": false, "start_time": "2020-12-17T20:01:36.148012", "status": "completed"} tags=[]
# see if efo code with missing label in term_id_to_label is here
assert term_id_to_label["EFO:0009628"] == "abnormal result of function studies"
assert term_id_to_label["EFO:0005606"] == "family history of breast cancer"

# %% papermill={"duration": 0.048516, "end_time": "2020-12-17T20:01:36.277394", "exception": false, "start_time": "2020-12-17T20:01:36.228878", "status": "completed"} tags=[]
assert term_id_to_label["EFO:0004616"] == "osteoarthritis, knee"

# %% papermill={"duration": 0.303598, "end_time": "2020-12-17T20:01:36.615598", "exception": false, "start_time": "2020-12-17T20:01:36.312000", "status": "completed"} tags=[]
# get current labels for old EFO codes
term_id_xrefs = read_data(
    conf.GENERAL["TERM_ID_XREFS_FILE"]
)  # [['label', 'EFO']].dropna().set_index('EFO')[['label']]

# %% papermill={"duration": 0.047579, "end_time": "2020-12-17T20:01:36.705071", "exception": false, "start_time": "2020-12-17T20:01:36.657492", "status": "completed"} tags=[]
term_id_xrefs.dtypes

# %% papermill={"duration": 0.04728, "end_time": "2020-12-17T20:01:36.787028", "exception": false, "start_time": "2020-12-17T20:01:36.739748", "status": "completed"} tags=[]
term_id_xrefs.shape

# %% papermill={"duration": 0.049799, "end_time": "2020-12-17T20:01:36.871540", "exception": false, "start_time": "2020-12-17T20:01:36.821741", "status": "completed"} tags=[]
term_id_xrefs.head()

# %% papermill={"duration": 0.067742, "end_time": "2020-12-17T20:01:36.975318", "exception": false, "start_time": "2020-12-17T20:01:36.907576", "status": "completed"} tags=[]
# see if for an old efo code we get the current efo label
new_efo_code = term_id_xrefs[term_id_xrefs["target_id"] == "EFO:1000673"].index[0]
display(new_efo_code)
assert term_id_to_label[new_efo_code] == "autoimmune bullous skin disease"

# %% [markdown] papermill={"duration": 0.034581, "end_time": "2020-12-17T20:01:37.045934", "exception": false, "start_time": "2020-12-17T20:01:37.011353", "status": "completed"} tags=[]
# # Add new EFO label

# %% [markdown] papermill={"duration": 0.035624, "end_time": "2020-12-17T20:01:37.116863", "exception": false, "start_time": "2020-12-17T20:01:37.081239", "status": "completed"} tags=[]
# ## Functions

# %% papermill={"duration": 0.048235, "end_time": "2020-12-17T20:01:37.200339", "exception": false, "start_time": "2020-12-17T20:01:37.152104", "status": "completed"} tags=[]
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

# %% [markdown] papermill={"duration": 0.037245, "end_time": "2020-12-17T20:01:37.272934", "exception": false, "start_time": "2020-12-17T20:01:37.235689", "status": "completed"} tags=[]
# ## Apply

# %% papermill={"duration": 0.091414, "end_time": "2020-12-17T20:01:37.401728", "exception": false, "start_time": "2020-12-17T20:01:37.310314", "status": "completed"} tags=[]
ukb_to_efo = ukb_to_efo.assign(
    current_term_label=ukb_to_efo.apply(_add_term_labels, axis=1)
)

# %% papermill={"duration": 0.047946, "end_time": "2020-12-17T20:01:37.485404", "exception": false, "start_time": "2020-12-17T20:01:37.437458", "status": "completed"} tags=[]
ukb_to_efo.shape

# %% papermill={"duration": 0.051521, "end_time": "2020-12-17T20:01:37.573219", "exception": false, "start_time": "2020-12-17T20:01:37.521698", "status": "completed"} tags=[]
ukb_to_efo.head()


# %% [markdown] papermill={"duration": 0.035094, "end_time": "2020-12-17T20:01:37.644184", "exception": false, "start_time": "2020-12-17T20:01:37.609090", "status": "completed"} tags=[]
# # Add categories

# %% [markdown] papermill={"duration": 0.035183, "end_time": "2020-12-17T20:01:37.714893", "exception": false, "start_time": "2020-12-17T20:01:37.679710", "status": "completed"} tags=[]
# It only adds Disease for now

# %% papermill={"duration": 0.047305, "end_time": "2020-12-17T20:01:37.797922", "exception": false, "start_time": "2020-12-17T20:01:37.750617", "status": "completed"} tags=[]
def _get_disease_category(row):
    term_ids = row["term_codes"]

    matches = term_pattern.findall(term_ids)

    for m in matches:
        if is_disease(graph, m):
            return "disease"

    return None


# %% papermill={"duration": 0.068073, "end_time": "2020-12-17T20:01:37.902256", "exception": false, "start_time": "2020-12-17T20:01:37.834183", "status": "completed"} tags=[]
ukb_to_efo = ukb_to_efo.assign(category=ukb_to_efo.apply(_get_disease_category, axis=1))

# %% papermill={"duration": 0.052715, "end_time": "2020-12-17T20:01:37.990475", "exception": false, "start_time": "2020-12-17T20:01:37.937760", "status": "completed"} tags=[]
ukb_to_efo.head()

# %% papermill={"duration": 0.04909, "end_time": "2020-12-17T20:01:38.076153", "exception": false, "start_time": "2020-12-17T20:01:38.027063", "status": "completed"} tags=[]
ukb_to_efo[ukb_to_efo["category"] == "disease"].shape

# %% papermill={"duration": 0.052329, "end_time": "2020-12-17T20:01:38.166117", "exception": false, "start_time": "2020-12-17T20:01:38.113788", "status": "completed"} tags=[]
_tmp = ukb_to_efo[ukb_to_efo["category"] == "disease"]
_tmp["current_term_label"].value_counts()

# %% [markdown] papermill={"duration": 0.035933, "end_time": "2020-12-17T20:01:38.239573", "exception": false, "start_time": "2020-12-17T20:01:38.203640", "status": "completed"} tags=[]
# # Testing

# %% papermill={"duration": 0.055548, "end_time": "2020-12-17T20:01:38.333371", "exception": false, "start_time": "2020-12-17T20:01:38.277823", "status": "completed"} tags=[]
# asthma exists
_tmp = ukb_to_efo[ukb_to_efo["current_term_label"].str.lower().str.contains("asthma")]
display(_tmp)
assert _tmp.shape[0] >= 4

# %% papermill={"duration": 0.054623, "end_time": "2020-12-17T20:01:38.425151", "exception": false, "start_time": "2020-12-17T20:01:38.370528", "status": "completed"} tags=[]
# check if old EFO labels are updated in orig_efo_names
# _tmp = ukb_to_efo.dropna()
_tmp = ukb_to_efo[ukb_to_efo["term_codes"].str.contains("EFO:1000673")]
display(_tmp)
_tmp = _tmp.iloc[0]
assert _tmp["term_label"] == "bullous skin disease"
assert _tmp["current_term_label"] == "autoimmune bullous skin disease"

# %% papermill={"duration": 0.05033, "end_time": "2020-12-17T20:01:38.513298", "exception": false, "start_time": "2020-12-17T20:01:38.462968", "status": "completed"} tags=[]
_tmp = ukb_to_efo.isna().sum()
assert _tmp.loc["ukb_fullcode"] == 0
assert _tmp.loc["term_codes"] == 0
assert _tmp.loc["current_term_label"] == 0

# %% papermill={"duration": 0.062271, "end_time": "2020-12-17T20:01:38.612219", "exception": false, "start_time": "2020-12-17T20:01:38.549948", "status": "completed"} tags=[]
# check all nan term labels now have non-empty current labels
_tmp = ukb_to_efo[ukb_to_efo["term_label"].isna()]
display(_tmp)
assert _tmp[_tmp["current_term_label"].isna()].shape[0] == 0
assert _tmp[_tmp["current_term_label"].str.strip().str.len() == 0].shape[0] == 0

# %% papermill={"duration": 0.053675, "end_time": "2020-12-17T20:01:38.705480", "exception": false, "start_time": "2020-12-17T20:01:38.651805", "status": "completed"} tags=[]
ukb_to_efo[(ukb_to_efo["current_term_label"].str.strip().str.len() == 0)]

# %% papermill={"duration": 0.050921, "end_time": "2020-12-17T20:01:38.795255", "exception": false, "start_time": "2020-12-17T20:01:38.744334", "status": "completed"} tags=[]
# check there are no null/empty current term labels
assert (
    ukb_to_efo[
        ukb_to_efo["current_term_label"].isna()
        | (ukb_to_efo["current_term_label"].str.strip().str.len() == 0)
    ].shape[0]
    == 0
)

# %% papermill={"duration": 0.056902, "end_time": "2020-12-17T20:01:38.891815", "exception": false, "start_time": "2020-12-17T20:01:38.834913", "status": "completed"} tags=[]
# How many entries have an term_label that differs from current_term_label?
ukb_to_efo[ukb_to_efo["term_label"] != ukb_to_efo["current_term_label"]]

# %% papermill={"duration": 0.06374, "end_time": "2020-12-17T20:01:38.995214", "exception": false, "start_time": "2020-12-17T20:01:38.931474", "status": "completed"} tags=[]
# How many entries comprise more than one EFO codes?
_tmp = ukb_to_efo[ukb_to_efo["current_term_label"].str.contains(" AND ")]
display(_tmp.shape)
display(_tmp)

# %% papermill={"duration": 0.053417, "end_time": "2020-12-17T20:01:39.089595", "exception": false, "start_time": "2020-12-17T20:01:39.036178", "status": "completed"} tags=[]
ukb_to_efo["mapping_type"].value_counts()

# %% papermill={"duration": 0.054891, "end_time": "2020-12-17T20:01:39.186328", "exception": false, "start_time": "2020-12-17T20:01:39.131437", "status": "completed"} tags=[]
ukb_to_efo[ukb_to_efo["mapping_type"] == "Exact"]["current_term_label"].value_counts()

# %% papermill={"duration": 0.056323, "end_time": "2020-12-17T20:01:39.282767", "exception": false, "start_time": "2020-12-17T20:01:39.226444", "status": "completed"} tags=[]
ukb_to_efo["current_term_label"].value_counts()

# %% papermill={"duration": 0.058268, "end_time": "2020-12-17T20:01:39.382594", "exception": false, "start_time": "2020-12-17T20:01:39.324326", "status": "completed"} tags=[]
ukb_to_efo[ukb_to_efo["current_term_label"] == "emotional symptom measurement"]

# %% papermill={"duration": 0.057769, "end_time": "2020-12-17T20:01:39.481865", "exception": false, "start_time": "2020-12-17T20:01:39.424096", "status": "completed"} tags=[]
ukb_to_efo[ukb_to_efo["ukb_fullcode"].duplicated(False)]

# %% papermill={"duration": 0.054316, "end_time": "2020-12-17T20:01:39.577598", "exception": false, "start_time": "2020-12-17T20:01:39.523282", "status": "completed"} tags=[]
ukb_to_efo[ukb_to_efo["ukb_fullcode"].duplicated(False)]["ukb_fullcode"].tolist()

# %% papermill={"duration": 0.053176, "end_time": "2020-12-17T20:01:39.672793", "exception": false, "start_time": "2020-12-17T20:01:39.619617", "status": "completed"} tags=[]
# Fix duplicated ukb_fullcode entries with different current_term_labels
idx = ukb_to_efo[
    ukb_to_efo["ukb_fullcode"]
    == "T79-Diagnoses_main_ICD10_T79_Certain_early_complications_of_trauma_not_elsewhere_classified"
].index
ukb_to_efo.loc[idx, "current_term_label"] = "complication"

# %% papermill={"duration": 0.053801, "end_time": "2020-12-17T20:01:39.767730", "exception": false, "start_time": "2020-12-17T20:01:39.713929", "status": "completed"} tags=[]
assert ukb_to_efo[ukb_to_efo["mapping_type"] == "Exact"]["ukb_fullcode"].is_unique

# %% papermill={"duration": 0.057468, "end_time": "2020-12-17T20:01:39.866241", "exception": false, "start_time": "2020-12-17T20:01:39.808773", "status": "completed"} tags=[]
_tmp = (
    ukb_to_efo.groupby(["ukb_fullcode", "current_term_label"])
    .count()
    .reset_index()[["ukb_fullcode", "current_term_label"]]
)
assert not _tmp.duplicated().any()

# %% [markdown] papermill={"duration": 0.040165, "end_time": "2020-12-17T20:01:39.947513", "exception": false, "start_time": "2020-12-17T20:01:39.907348", "status": "completed"} tags=[]
# # Save

# %% [markdown] papermill={"duration": 0.040128, "end_time": "2020-12-17T20:01:40.028310", "exception": false, "start_time": "2020-12-17T20:01:39.988182", "status": "completed"} tags=[]
# ## In main data folder

# %% papermill={"duration": 0.057511, "end_time": "2020-12-17T20:01:40.126350", "exception": false, "start_time": "2020-12-17T20:01:40.068839", "status": "completed"} tags=[]
outfile = conf.PHENOMEXCAN["TRAITS_FULLCODE_TO_EFO_MAP_FILE"]
display(outfile)

ukb_to_efo.to_csv(outfile, sep="\t", index=False)

# %% [markdown] papermill={"duration": 0.040877, "end_time": "2020-12-17T20:01:40.208895", "exception": false, "start_time": "2020-12-17T20:01:40.168018", "status": "completed"} tags=[]
# ## In libs/data folder

# %% [markdown] papermill={"duration": 0.043404, "end_time": "2020-12-17T20:01:40.293817", "exception": false, "start_time": "2020-12-17T20:01:40.250413", "status": "completed"} tags=[]
# Since this file (`outfile`) is used by the Trait class to return EFO codes/labels for PhenomeXcan traits, it is copied also to a source code folder and it is supposed to be versioned.

# %% papermill={"duration": 0.053226, "end_time": "2020-12-17T20:01:40.388711", "exception": false, "start_time": "2020-12-17T20:01:40.335485", "status": "completed"} tags=[]
display(Trait.UKB_TO_EFO_MAP_FILE)

# %% papermill={"duration": 0.053364, "end_time": "2020-12-17T20:01:40.482975", "exception": false, "start_time": "2020-12-17T20:01:40.429611", "status": "completed"} tags=[]
copyfile(
    outfile,
    Trait.UKB_TO_EFO_MAP_FILE,
)
