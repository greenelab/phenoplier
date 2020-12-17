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

# %% [markdown] papermill={"duration": 0.068364, "end_time": "2020-12-14T21:24:23.427012", "exception": false, "start_time": "2020-12-14T21:24:23.358648", "status": "completed"} tags=[]
# # Description

# %% [markdown]
# It creates a text file with mappings for all traits in PhenomeXcan (many of them are from UK Biobank, and a small set of 42 traits are from other studies) to EFO labels. It also adds a category for each trait, which now contains only one category: `disease` (or empty if not categorized).

# %% [markdown] papermill={"duration": 0.031032, "end_time": "2020-12-14T21:24:23.491076", "exception": false, "start_time": "2020-12-14T21:24:23.460044", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.042071, "end_time": "2020-12-14T21:24:23.564672", "exception": false, "start_time": "2020-12-14T21:24:23.522601", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 0.382904, "end_time": "2020-12-14T21:24:23.979193", "exception": false, "start_time": "2020-12-14T21:24:23.596289", "status": "completed"} tags=[]
import pandas as pd

import conf
from data.cache import read_data
from entity import Trait


# %% [markdown] papermill={"duration": 0.030998, "end_time": "2020-12-14T21:24:24.044044", "exception": false, "start_time": "2020-12-14T21:24:24.013046", "status": "completed"} tags=[]
# # Functions

# %% papermill={"duration": 0.041088, "end_time": "2020-12-14T21:24:24.116129", "exception": false, "start_time": "2020-12-14T21:24:24.075041", "status": "completed"} tags=[]
def get_parents(graph, node):
    for t in graph.successors(node):
        yield t

def _is_disease_single_node(node):
    return node == 'EFO:0000408'

def is_disease(graph, node):
    if node not in graph.nodes:
        return False
    
    if _is_disease_single_node(node):
        return True
    
    for parent_node in get_parents(graph, node):
        if is_disease(graph, parent_node):
            return True
    
    return False


# %% [markdown] papermill={"duration": 0.030818, "end_time": "2020-12-14T21:24:24.178525", "exception": false, "start_time": "2020-12-14T21:24:24.147707", "status": "completed"} tags=[]
# # Load EFO Ontology

# %% papermill={"duration": 0.21541, "end_time": "2020-12-14T21:24:24.424283", "exception": false, "start_time": "2020-12-14T21:24:24.208873", "status": "completed"} tags=[]
import obonet

# %% papermill={"duration": 4.790495, "end_time": "2020-12-14T21:24:29.248624", "exception": false, "start_time": "2020-12-14T21:24:24.458129", "status": "completed"} tags=[]
url = conf.GENERAL["EFO_ONTOLOGY_OBO_FILE"]
graph = obonet.read_obo(url)

# %% papermill={"duration": 0.046482, "end_time": "2020-12-14T21:24:29.333747", "exception": false, "start_time": "2020-12-14T21:24:29.287265", "status": "completed"} tags=[]
# Number of nodes
len(graph)

# %% papermill={"duration": 0.078536, "end_time": "2020-12-14T21:24:29.444135", "exception": false, "start_time": "2020-12-14T21:24:29.365599", "status": "completed"} tags=[]
# Number of edges
graph.number_of_edges()

# %% papermill={"duration": 0.043067, "end_time": "2020-12-14T21:24:29.521520", "exception": false, "start_time": "2020-12-14T21:24:29.478453", "status": "completed"} tags=[]
assert graph.nodes['EFO:0000270'].get('name') == 'asthma'

# %% [markdown] papermill={"duration": 0.030502, "end_time": "2020-12-14T21:24:29.584244", "exception": false, "start_time": "2020-12-14T21:24:29.553742", "status": "completed"} tags=[]
# # Load PhenomeXcan traits

# %% papermill={"duration": 0.302417, "end_time": "2020-12-14T21:24:29.917577", "exception": false, "start_time": "2020-12-14T21:24:29.615160", "status": "completed"} tags=[]
phenomexan_traits_names = read_data(
    conf.PHENOMEXCAN["SMULTIXCAN_MASHR_ZSCORES_FILE"]
).columns.tolist()

# %% papermill={"duration": 0.043262, "end_time": "2020-12-14T21:24:29.994357", "exception": false, "start_time": "2020-12-14T21:24:29.951095", "status": "completed"} tags=[]
len(phenomexan_traits_names)

# %% papermill={"duration": 0.659404, "end_time": "2020-12-14T21:24:30.685506", "exception": false, "start_time": "2020-12-14T21:24:30.026102", "status": "completed"} tags=[]
phenomexcan_traits = [
    Trait.get_trait(full_code=t)
    for t in phenomexan_traits_names
]

# %% papermill={"duration": 0.042602, "end_time": "2020-12-14T21:24:30.759676", "exception": false, "start_time": "2020-12-14T21:24:30.717074", "status": "completed"} tags=[]
assert len(phenomexcan_traits) == len(phenomexan_traits_names)

# %% papermill={"duration": 0.064919, "end_time": "2020-12-14T21:24:30.857479", "exception": false, "start_time": "2020-12-14T21:24:30.792560", "status": "completed"} tags=[]
phenomexcan_code_to_full_code = {
    t.code: t.get_plain_name()
    for t in phenomexcan_traits
}

# %% papermill={"duration": 0.042902, "end_time": "2020-12-14T21:24:30.933008", "exception": false, "start_time": "2020-12-14T21:24:30.890106", "status": "completed"} tags=[]
assert phenomexcan_code_to_full_code['50_raw'] == '50_raw-Standing_height'

# %% [markdown] papermill={"duration": 0.030861, "end_time": "2020-12-14T21:24:30.996071", "exception": false, "start_time": "2020-12-14T21:24:30.965210", "status": "completed"} tags=[]
# # Load UKB to EFO mappings

# %% papermill={"duration": 0.046541, "end_time": "2020-12-14T21:24:31.073660", "exception": false, "start_time": "2020-12-14T21:24:31.027119", "status": "completed"} tags=[]
ukb_to_efo = read_data(conf.UK_BIOBANK["UKBCODE_TO_EFO_MAP_FILE"])

# %% papermill={"duration": 0.043984, "end_time": "2020-12-14T21:24:31.149710", "exception": false, "start_time": "2020-12-14T21:24:31.105726", "status": "completed"} tags=[]
ukb_to_efo.shape

# %% papermill={"duration": 0.050958, "end_time": "2020-12-14T21:24:31.232359", "exception": false, "start_time": "2020-12-14T21:24:31.181401", "status": "completed"} tags=[]
ukb_to_efo

# %% papermill={"duration": 0.045238, "end_time": "2020-12-14T21:24:31.311066", "exception": false, "start_time": "2020-12-14T21:24:31.265828", "status": "completed"} tags=[]
ukb_to_efo = ukb_to_efo.rename(columns={
    'MAPPED_TERM_LABEL': 'term_label',
    'MAPPING_TYPE': 'mapping_type',
    'MAPPED_TERM_URI': 'term_codes',
    'ICD10_CODE/SELF_REPORTED_TRAIT_FIELD_CODE': 'ukb_code',
})[['ukb_code', 'term_label', 'term_codes', 'mapping_type']]

# %% [markdown] papermill={"duration": 0.031126, "end_time": "2020-12-14T21:24:31.374511", "exception": false, "start_time": "2020-12-14T21:24:31.343385", "status": "completed"} tags=[]
# ## Add GTEx GWAS EFO terms

# %% papermill={"duration": 0.043399, "end_time": "2020-12-14T21:24:31.449190", "exception": false, "start_time": "2020-12-14T21:24:31.405791", "status": "completed"} tags=[]
from entity import Trait, GTEXGWASTrait

# %% papermill={"duration": 0.04776, "end_time": "2020-12-14T21:24:31.531077", "exception": false, "start_time": "2020-12-14T21:24:31.483317", "status": "completed"} tags=[]
all_gtex_gwas_phenos = [p for p in phenomexcan_traits if GTEXGWASTrait.is_phenotype_from_study(p.full_code)]

# %% papermill={"duration": 0.04402, "end_time": "2020-12-14T21:24:31.607132", "exception": false, "start_time": "2020-12-14T21:24:31.563112", "status": "completed"} tags=[]
_tmp = len(all_gtex_gwas_phenos)
display(_tmp)
assert _tmp == 42

# %% papermill={"duration": 0.044172, "end_time": "2020-12-14T21:24:31.683607", "exception": false, "start_time": "2020-12-14T21:24:31.639435", "status": "completed"} tags=[]
all_gtex_gwas_phenos[:10]

# %% papermill={"duration": 0.043858, "end_time": "2020-12-14T21:24:31.760217", "exception": false, "start_time": "2020-12-14T21:24:31.716359", "status": "completed"} tags=[]
_old_shape = ukb_to_efo.shape

# %% papermill={"duration": 0.045994, "end_time": "2020-12-14T21:24:31.839747", "exception": false, "start_time": "2020-12-14T21:24:31.793753", "status": "completed"} tags=[]
ukb_to_efo = ukb_to_efo.append(pd.DataFrame({
    'ukb_code': [ggp.full_code for ggp in all_gtex_gwas_phenos],
    'term_codes': [ggp.orig_efo_id for ggp in all_gtex_gwas_phenos],
}), ignore_index=True)

# %% papermill={"duration": 0.045478, "end_time": "2020-12-14T21:24:31.919137", "exception": false, "start_time": "2020-12-14T21:24:31.873659", "status": "completed"} tags=[]
# Fix wrong EFO codes
idx = ukb_to_efo[ukb_to_efo['ukb_code'] == 'BCAC_ER_negative_BreastCancer_EUR'].index
ukb_to_efo.loc[idx, 'term_codes'] = 'EFO_1000650'

idx = ukb_to_efo[ukb_to_efo['ukb_code'] == 'CARDIoGRAM_C4D_CAD_ADDITIVE'].index
ukb_to_efo.loc[idx, 'term_codes'] = 'EFO_0001645'

# %% papermill={"duration": 0.043949, "end_time": "2020-12-14T21:24:31.995858", "exception": false, "start_time": "2020-12-14T21:24:31.951909", "status": "completed"} tags=[]
# ukb_to_efo_maps = ukb_to_efo_maps.dropna(subset=['efo_code'])

# %% papermill={"duration": 0.045119, "end_time": "2020-12-14T21:24:32.074613", "exception": false, "start_time": "2020-12-14T21:24:32.029494", "status": "completed"} tags=[]
_tmp = ukb_to_efo.shape
display(_tmp)
assert _tmp[0] == _old_shape[0] + 42

# %% [markdown] papermill={"duration": 0.032492, "end_time": "2020-12-14T21:24:32.139801", "exception": false, "start_time": "2020-12-14T21:24:32.107309", "status": "completed"} tags=[]
# ## Replace values and remove nans

# %% papermill={"duration": 0.050455, "end_time": "2020-12-14T21:24:32.223094", "exception": false, "start_time": "2020-12-14T21:24:32.172639", "status": "completed"} tags=[]
ukb_to_efo = ukb_to_efo.replace(
    {
        'term_codes': {
#             '\|\|': ', ',
            '_': ':',
            'HP0011106': 'HP:0011106'
        },
#         'efo_name': {
#             '\|\|': ', ',
#         }
    },
    regex=True
)

# %% papermill={"duration": 0.046355, "end_time": "2020-12-14T21:24:32.302842", "exception": false, "start_time": "2020-12-14T21:24:32.256487", "status": "completed"} tags=[]
ukb_to_efo = ukb_to_efo.dropna(how='all')

# %% papermill={"duration": 0.045652, "end_time": "2020-12-14T21:24:32.381940", "exception": false, "start_time": "2020-12-14T21:24:32.336288", "status": "completed"} tags=[]
ukb_to_efo = ukb_to_efo.dropna(subset=['term_codes'])

# %% papermill={"duration": 0.050745, "end_time": "2020-12-14T21:24:32.484673", "exception": false, "start_time": "2020-12-14T21:24:32.433928", "status": "completed"} tags=[]
assert ukb_to_efo[ukb_to_efo['term_codes'].isna()].shape[0] == 0

# %% papermill={"duration": 0.045583, "end_time": "2020-12-14T21:24:32.563975", "exception": false, "start_time": "2020-12-14T21:24:32.518392", "status": "completed"} tags=[]
assert ukb_to_efo[ukb_to_efo['term_codes'].str.contains('EFO:')].shape[0] > 0

# %% papermill={"duration": 0.045494, "end_time": "2020-12-14T21:24:32.642424", "exception": false, "start_time": "2020-12-14T21:24:32.596930", "status": "completed"} tags=[]
assert ukb_to_efo[ukb_to_efo['term_codes'].str.contains('HP:')].shape[0] > 0

# %% papermill={"duration": 0.046165, "end_time": "2020-12-14T21:24:32.722818", "exception": false, "start_time": "2020-12-14T21:24:32.676653", "status": "completed"} tags=[]
ukb_to_efo.shape

# %% papermill={"duration": 0.048948, "end_time": "2020-12-14T21:24:32.805334", "exception": false, "start_time": "2020-12-14T21:24:32.756386", "status": "completed"} tags=[]
ukb_to_efo.head()


# %% [markdown] papermill={"duration": 0.032853, "end_time": "2020-12-14T21:24:32.871741", "exception": false, "start_time": "2020-12-14T21:24:32.838888", "status": "completed"} tags=[]
# # Add PhenomeXcan code/full code

# %% papermill={"duration": 0.044527, "end_time": "2020-12-14T21:24:32.949744", "exception": false, "start_time": "2020-12-14T21:24:32.905217", "status": "completed"} tags=[]
def _get_fullcode(code):
    if code in phenomexcan_code_to_full_code:
        return phenomexcan_code_to_full_code[code]

    return None


# %% papermill={"duration": 0.047032, "end_time": "2020-12-14T21:24:33.030662", "exception": false, "start_time": "2020-12-14T21:24:32.983630", "status": "completed"} tags=[]
ukb_to_efo = ukb_to_efo.assign(
    ukb_fullcode=ukb_to_efo['ukb_code'].apply(_get_fullcode)
)

# %% papermill={"duration": 0.045802, "end_time": "2020-12-14T21:24:33.109811", "exception": false, "start_time": "2020-12-14T21:24:33.064009", "status": "completed"} tags=[]
ukb_to_efo.shape

# %% papermill={"duration": 0.049465, "end_time": "2020-12-14T21:24:33.192764", "exception": false, "start_time": "2020-12-14T21:24:33.143299", "status": "completed"} tags=[]
ukb_to_efo.head()

# %% papermill={"duration": 0.046841, "end_time": "2020-12-14T21:24:33.273347", "exception": false, "start_time": "2020-12-14T21:24:33.226506", "status": "completed"} tags=[]
# remove entries for which we couldn't map a ukb full code
ukb_to_efo = ukb_to_efo.dropna(subset=['ukb_fullcode'])

# %% papermill={"duration": 0.045889, "end_time": "2020-12-14T21:24:33.352897", "exception": false, "start_time": "2020-12-14T21:24:33.307008", "status": "completed"} tags=[]
ukb_to_efo.shape

# %% papermill={"duration": 0.047055, "end_time": "2020-12-14T21:24:33.433506", "exception": false, "start_time": "2020-12-14T21:24:33.386451", "status": "completed"} tags=[]
ukb_to_efo.isna().sum()

# %% papermill={"duration": 0.059609, "end_time": "2020-12-14T21:24:33.527960", "exception": false, "start_time": "2020-12-14T21:24:33.468351", "status": "completed"} tags=[]
# for these ones we need to query the original EFO ontology
ukb_to_efo[ukb_to_efo['term_label'].isna()]

# %% [markdown] papermill={"duration": 0.03529, "end_time": "2020-12-14T21:24:33.599391", "exception": false, "start_time": "2020-12-14T21:24:33.564101", "status": "completed"} tags=[]
# # Load EFO labels and xrefs

# %% papermill={"duration": 0.094413, "end_time": "2020-12-14T21:24:33.728463", "exception": false, "start_time": "2020-12-14T21:24:33.634050", "status": "completed"} tags=[]
term_id_to_label = read_data(
    conf.GENERAL["TERM_ID_LABEL_FILE"]
)[['term_id', 'label']].dropna().set_index('term_id')['label'].to_dict()

# %% papermill={"duration": 0.047694, "end_time": "2020-12-14T21:24:33.814061", "exception": false, "start_time": "2020-12-14T21:24:33.766367", "status": "completed"} tags=[]
len(term_id_to_label)

# %% papermill={"duration": 0.046866, "end_time": "2020-12-14T21:24:33.896066", "exception": false, "start_time": "2020-12-14T21:24:33.849200", "status": "completed"} tags=[]
# see if efo code with missing label in term_id_to_label is here
assert term_id_to_label['EFO:0009628'] == 'abnormal result of function studies'
assert term_id_to_label['EFO:0005606'] == 'family history of breast cancer'

# %% papermill={"duration": 0.046792, "end_time": "2020-12-14T21:24:33.977983", "exception": false, "start_time": "2020-12-14T21:24:33.931191", "status": "completed"} tags=[]
assert term_id_to_label['EFO:0004616'] == 'osteoarthritis, knee'

# %% papermill={"duration": 0.306919, "end_time": "2020-12-14T21:24:34.320685", "exception": false, "start_time": "2020-12-14T21:24:34.013766", "status": "completed"} tags=[]
# get current labels for old EFO codes
term_id_xrefs = read_data(
    conf.GENERAL["TERM_ID_XREFS_FILE"]
)#[['label', 'EFO']].dropna().set_index('EFO')[['label']]

# %% papermill={"duration": 0.047999, "end_time": "2020-12-14T21:24:34.411502", "exception": false, "start_time": "2020-12-14T21:24:34.363503", "status": "completed"} tags=[]
term_id_xrefs.dtypes

# %% papermill={"duration": 0.050494, "end_time": "2020-12-14T21:24:34.496866", "exception": false, "start_time": "2020-12-14T21:24:34.446372", "status": "completed"} tags=[]
term_id_xrefs.shape

# %% papermill={"duration": 0.05031, "end_time": "2020-12-14T21:24:34.583325", "exception": false, "start_time": "2020-12-14T21:24:34.533015", "status": "completed"} tags=[]
term_id_xrefs.head()

# %% papermill={"duration": 0.068648, "end_time": "2020-12-14T21:24:34.687801", "exception": false, "start_time": "2020-12-14T21:24:34.619153", "status": "completed"} tags=[]
# see if for an old efo code we get the current efo label
new_efo_code = term_id_xrefs[term_id_xrefs['target_id'] == 'EFO:1000673'].index[0]
display(new_efo_code)
assert term_id_to_label[new_efo_code] == 'autoimmune bullous skin disease'

# %% [markdown] papermill={"duration": 0.035634, "end_time": "2020-12-14T21:24:34.759833", "exception": false, "start_time": "2020-12-14T21:24:34.724199", "status": "completed"} tags=[]
# # Add new EFO label

# %% papermill={"duration": 0.049529, "end_time": "2020-12-14T21:24:34.845247", "exception": false, "start_time": "2020-12-14T21:24:34.795718", "status": "completed"} tags=[]
import re
term_pattern = re.compile('\w+:\w+')

def _add_term_labels(row):
    term_ids = row['term_codes']
    
    matches = term_pattern.findall(term_ids)
    
    labels = []
    for m in matches:
        if m in term_id_to_label:
            new_label = term_id_to_label[m]
        else:
            other_xrefs = term_id_xrefs[term_id_xrefs['target_id'] == m]
            if other_xrefs.shape[0] == 1:
                new_label = term_id_to_label[other_xrefs.index[0]]
            elif other_xrefs.shape[0] > 1:
                new_label = term_id_to_label[other_xrefs.index[0]]
            elif not pd.isnull(row['term_label']):
                new_label = row['term_label']
            else:
                continue
    
        labels.append(new_label.lower())
    
    return ' AND '.join(labels)


# %% papermill={"duration": 0.093161, "end_time": "2020-12-14T21:24:35.168464", "exception": false, "start_time": "2020-12-14T21:24:35.075303", "status": "completed"} tags=[]
ukb_to_efo = ukb_to_efo.assign(
    current_term_label=ukb_to_efo.apply(_add_term_labels, axis=1)
)

# %% papermill={"duration": 0.048117, "end_time": "2020-12-14T21:24:35.252156", "exception": false, "start_time": "2020-12-14T21:24:35.204039", "status": "completed"} tags=[]
ukb_to_efo.shape

# %% papermill={"duration": 0.052045, "end_time": "2020-12-14T21:24:35.340648", "exception": false, "start_time": "2020-12-14T21:24:35.288603", "status": "completed"} tags=[]
ukb_to_efo.head()


# %% [markdown] papermill={"duration": 0.035862, "end_time": "2020-12-14T21:24:35.413784", "exception": false, "start_time": "2020-12-14T21:24:35.377922", "status": "completed"} tags=[]
# # Add categories

# %% [markdown] papermill={"duration": 0.038166, "end_time": "2020-12-14T21:24:35.488832", "exception": false, "start_time": "2020-12-14T21:24:35.450666", "status": "completed"} tags=[]
# It only adds Disease for now

# %% papermill={"duration": 0.049049, "end_time": "2020-12-14T21:24:35.575465", "exception": false, "start_time": "2020-12-14T21:24:35.526416", "status": "completed"} tags=[]
def _get_disease_category(row):
    term_ids = row['term_codes']
    
    matches = term_pattern.findall(term_ids)
    
    labels = []
    for m in matches:
        if is_disease(graph, m):
            return 'disease'
    
    return None


# %% papermill={"duration": 0.070816, "end_time": "2020-12-14T21:24:35.682881", "exception": false, "start_time": "2020-12-14T21:24:35.612065", "status": "completed"} tags=[]
ukb_to_efo = ukb_to_efo.assign(category=ukb_to_efo.apply(_get_disease_category, axis=1))

# %% papermill={"duration": 0.053911, "end_time": "2020-12-14T21:24:35.773530", "exception": false, "start_time": "2020-12-14T21:24:35.719619", "status": "completed"} tags=[]
ukb_to_efo.head()

# %% papermill={"duration": 0.050223, "end_time": "2020-12-14T21:24:35.861970", "exception": false, "start_time": "2020-12-14T21:24:35.811747", "status": "completed"} tags=[]
ukb_to_efo[ukb_to_efo['category'] == 'disease'].shape

# %% papermill={"duration": 0.051226, "end_time": "2020-12-14T21:24:35.949668", "exception": false, "start_time": "2020-12-14T21:24:35.898442", "status": "completed"} tags=[]
_tmp = ukb_to_efo[ukb_to_efo['category'] == 'disease']
_tmp['current_term_label'].value_counts()

# %% [markdown] papermill={"duration": 0.036316, "end_time": "2020-12-14T21:24:36.022534", "exception": false, "start_time": "2020-12-14T21:24:35.986218", "status": "completed"} tags=[]
# # Testing

# %% papermill={"duration": 0.054541, "end_time": "2020-12-14T21:24:36.113175", "exception": false, "start_time": "2020-12-14T21:24:36.058634", "status": "completed"} tags=[]
# asthma exists
_tmp = ukb_to_efo[ukb_to_efo['current_term_label'].str.lower().str.contains('asthma')]
display(_tmp)
assert _tmp.shape[0] >= 4

# %% papermill={"duration": 0.05549, "end_time": "2020-12-14T21:24:36.205794", "exception": false, "start_time": "2020-12-14T21:24:36.150304", "status": "completed"} tags=[]
# check if old EFO labels are updated in orig_efo_names
# _tmp = ukb_to_efo.dropna()
_tmp = ukb_to_efo[ukb_to_efo['term_codes'].str.contains('EFO:1000673')]
display(_tmp)
_tmp = _tmp.iloc[0]
assert _tmp['term_label'] == 'bullous skin disease'
assert _tmp['current_term_label'] == 'autoimmune bullous skin disease'

# %% papermill={"duration": 0.050357, "end_time": "2020-12-14T21:24:36.294533", "exception": false, "start_time": "2020-12-14T21:24:36.244176", "status": "completed"} tags=[]
_tmp = ukb_to_efo.isna().sum()
assert _tmp.loc['ukb_fullcode'] == 0
assert _tmp.loc['term_codes'] == 0
assert _tmp.loc['current_term_label'] == 0

# %% papermill={"duration": 0.06156, "end_time": "2020-12-14T21:24:36.393800", "exception": false, "start_time": "2020-12-14T21:24:36.332240", "status": "completed"} tags=[]
# check all nan term labels now have non-empty current labels
_tmp = ukb_to_efo[ukb_to_efo['term_label'].isna()]
display(_tmp)
assert _tmp[_tmp['current_term_label'].isna()].shape[0] == 0
assert _tmp[_tmp['current_term_label'].str.strip().str.len() == 0].shape[0] == 0

# %% papermill={"duration": 0.055272, "end_time": "2020-12-14T21:24:36.488141", "exception": false, "start_time": "2020-12-14T21:24:36.432869", "status": "completed"} tags=[]
ukb_to_efo[(ukb_to_efo['current_term_label'].str.strip().str.len() == 0)]

# %% papermill={"duration": 0.052713, "end_time": "2020-12-14T21:24:36.580833", "exception": false, "start_time": "2020-12-14T21:24:36.528120", "status": "completed"} tags=[]
# check there are no null/empty current term labels
assert ukb_to_efo[
    ukb_to_efo['current_term_label'].isna() | 
    (ukb_to_efo['current_term_label'].str.strip().str.len() == 0)
].shape[0] == 0

# %% papermill={"duration": 0.058794, "end_time": "2020-12-14T21:24:36.679161", "exception": false, "start_time": "2020-12-14T21:24:36.620367", "status": "completed"} tags=[]
# How many entries have an term_label that differs from current_term_label?
ukb_to_efo[ukb_to_efo['term_label'] != ukb_to_efo['current_term_label']]

# %% papermill={"duration": 0.064808, "end_time": "2020-12-14T21:24:36.782826", "exception": false, "start_time": "2020-12-14T21:24:36.718018", "status": "completed"} tags=[]
# How many entries comprise more than one EFO codes?
_tmp = ukb_to_efo[ukb_to_efo['current_term_label'].str.contains(' AND ')]
display(_tmp.shape)
display(_tmp)

# %% papermill={"duration": 0.054422, "end_time": "2020-12-14T21:24:36.878778", "exception": false, "start_time": "2020-12-14T21:24:36.824356", "status": "completed"} tags=[]
ukb_to_efo['mapping_type'].value_counts()

# %% papermill={"duration": 0.055562, "end_time": "2020-12-14T21:24:36.974567", "exception": false, "start_time": "2020-12-14T21:24:36.919005", "status": "completed"} tags=[]
ukb_to_efo[ukb_to_efo['mapping_type'] == 'Exact']['current_term_label'].value_counts()

# %% papermill={"duration": 0.054585, "end_time": "2020-12-14T21:24:37.073994", "exception": false, "start_time": "2020-12-14T21:24:37.019409", "status": "completed"} tags=[]
ukb_to_efo['current_term_label'].value_counts()

# %% papermill={"duration": 0.059395, "end_time": "2020-12-14T21:24:37.174540", "exception": false, "start_time": "2020-12-14T21:24:37.115145", "status": "completed"} tags=[]
ukb_to_efo[ukb_to_efo['current_term_label'] == 'emotional symptom measurement']

# %% papermill={"duration": 0.059722, "end_time": "2020-12-14T21:24:37.277125", "exception": false, "start_time": "2020-12-14T21:24:37.217403", "status": "completed"} tags=[]
ukb_to_efo[ukb_to_efo['ukb_fullcode'].duplicated(False)]

# %% papermill={"duration": 0.055363, "end_time": "2020-12-14T21:24:37.373572", "exception": false, "start_time": "2020-12-14T21:24:37.318209", "status": "completed"} tags=[]
ukb_to_efo[ukb_to_efo['ukb_fullcode'].duplicated(False)]['ukb_fullcode'].tolist()

# %% papermill={"duration": 0.054069, "end_time": "2020-12-14T21:24:37.469428", "exception": false, "start_time": "2020-12-14T21:24:37.415359", "status": "completed"} tags=[]
# Fix duplicated ukb_fullcode entries with different current_term_labels
idx = ukb_to_efo[ukb_to_efo['ukb_fullcode'] == 'T79-Diagnoses_main_ICD10_T79_Certain_early_complications_of_trauma_not_elsewhere_classified'].index
ukb_to_efo.loc[idx, 'current_term_label'] = 'complication'

# %% papermill={"duration": 0.055716, "end_time": "2020-12-14T21:24:37.568994", "exception": false, "start_time": "2020-12-14T21:24:37.513278", "status": "completed"} tags=[]
assert ukb_to_efo[ukb_to_efo['mapping_type'] == 'Exact']['ukb_fullcode'].is_unique

# %% papermill={"duration": 0.058752, "end_time": "2020-12-14T21:24:37.669188", "exception": false, "start_time": "2020-12-14T21:24:37.610436", "status": "completed"} tags=[]
_tmp = ukb_to_efo.groupby(['ukb_fullcode', 'current_term_label']).count().reset_index()[['ukb_fullcode', 'current_term_label']]
assert not _tmp.duplicated().any()

# %% [markdown] papermill={"duration": 0.041566, "end_time": "2020-12-14T21:24:37.752274", "exception": false, "start_time": "2020-12-14T21:24:37.710708", "status": "completed"} tags=[]
# # Save

# %% papermill={"duration": 0.059995, "end_time": "2020-12-14T21:24:37.853320", "exception": false, "start_time": "2020-12-14T21:24:37.793325", "status": "completed"} tags=[]
outfile = conf.PHENOMEXCAN["TRAITS_FULLCODE_TO_EFO_MAP_FILE"]
display(outfile)

ukb_to_efo.to_csv(outfile, sep='\t', index=False)

# %% papermill={"duration": 0.041717, "end_time": "2020-12-14T21:24:37.936675", "exception": false, "start_time": "2020-12-14T21:24:37.894958", "status": "completed"} tags=[]
