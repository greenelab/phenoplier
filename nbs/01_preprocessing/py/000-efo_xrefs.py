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

# %% [markdown] papermill={"duration": 0.038284, "end_time": "2020-12-14T21:24:14.404156", "exception": false, "start_time": "2020-12-14T21:24:14.365872", "status": "completed"} tags=[]
# # Description

# %% [markdown]
# It extracts from the EFO ontology all the xrefs from efo labels to other ontologies/datasets (such as Disease Ontology, ICD9, etc).

# %% [markdown] papermill={"duration": 0.012823, "end_time": "2020-12-14T21:24:14.429845", "exception": false, "start_time": "2020-12-14T21:24:14.417022", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.023627, "end_time": "2020-12-14T21:24:14.465328", "exception": false, "start_time": "2020-12-14T21:24:14.441701", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 0.36795, "end_time": "2020-12-14T21:24:14.845672", "exception": false, "start_time": "2020-12-14T21:24:14.477722", "status": "completed"} tags=[]
from collections import defaultdict

import pandas as pd
import networkx
import obonet

import conf


# %% [markdown] papermill={"duration": 0.01148, "end_time": "2020-12-14T21:24:14.869442", "exception": false, "start_time": "2020-12-14T21:24:14.857962", "status": "completed"} tags=[]
# # Functions

# %% papermill={"duration": 0.023396, "end_time": "2020-12-14T21:24:14.904419", "exception": false, "start_time": "2020-12-14T21:24:14.881023", "status": "completed"} tags=[]
def groupby(data, sep=":"):
    if data is None:
        return {}
    res = defaultdict(set)
    for d in data:
        ds = d.split(sep)
        res[ds[0]].add(d)
    return res


# %% papermill={"duration": 0.023823, "end_time": "2020-12-14T21:24:14.940490", "exception": false, "start_time": "2020-12-14T21:24:14.916667", "status": "completed"} tags=[]
test_data = [
    "DOID:2841",
    "HP:0002099",
    "ICD10:J45",
    "ICD10:J45.90",
    "ICD9:493",
    "ICD9:493.81",
    "ICD9:493.9",
    "KEGG:05310",
]

# %% papermill={"duration": 0.024736, "end_time": "2020-12-14T21:24:14.978061", "exception": false, "start_time": "2020-12-14T21:24:14.953325", "status": "completed"} tags=[]
_tmp = groupby(test_data)
assert _tmp is not None
assert len(_tmp) == 5

assert len(_tmp["DOID"]) == 1
assert len(_tmp["HP"]) == 1
assert len(_tmp["ICD10"]) == 2
assert len(_tmp["ICD9"]) == 3
assert len(_tmp["KEGG"]) == 1

# %% papermill={"duration": 0.028662, "end_time": "2020-12-14T21:24:15.019378", "exception": false, "start_time": "2020-12-14T21:24:14.990716", "status": "completed"} tags=[]
_tmp


# %% papermill={"duration": 0.024462, "end_time": "2020-12-14T21:24:15.056586", "exception": false, "start_time": "2020-12-14T21:24:15.032124", "status": "completed"} tags=[]
def get_parents(node):
    for t in graph.successors(node):
        yield t


def _is_disease_single_node(node):
    return node == "EFO:0000408"


def is_disease(node):
    if _is_disease_single_node(node):
        return True

    for parent_node in get_parents(node):
        if is_disease(parent_node):
            return True

    return False


# %% [markdown] papermill={"duration": 0.011866, "end_time": "2020-12-14T21:24:15.080899", "exception": false, "start_time": "2020-12-14T21:24:15.069033", "status": "completed"} tags=[]
# # Read the EFO ontology

# %% papermill={"duration": 4.77742, "end_time": "2020-12-14T21:24:19.870138", "exception": false, "start_time": "2020-12-14T21:24:15.092718", "status": "completed"} tags=[]
url = conf.GENERAL["EFO_ONTOLOGY_OBO_FILE"]
graph = obonet.read_obo(url)

# %% papermill={"duration": 0.028453, "end_time": "2020-12-14T21:24:19.914798", "exception": false, "start_time": "2020-12-14T21:24:19.886345", "status": "completed"} tags=[]
# Number of nodes
len(graph)

# %% papermill={"duration": 0.06168, "end_time": "2020-12-14T21:24:19.990643", "exception": false, "start_time": "2020-12-14T21:24:19.928963", "status": "completed"} tags=[]
# Number of edges
graph.number_of_edges()

# %% papermill={"duration": 0.025705, "end_time": "2020-12-14T21:24:20.030513", "exception": false, "start_time": "2020-12-14T21:24:20.004808", "status": "completed"} tags=[]
assert graph.nodes["EFO:0000270"].get("name") == "asthma"

# %% [markdown] papermill={"duration": 0.013662, "end_time": "2020-12-14T21:24:20.058162", "exception": false, "start_time": "2020-12-14T21:24:20.044500", "status": "completed"} tags=[]
# # Map EFO to other references

# %% papermill={"duration": 0.032709, "end_time": "2020-12-14T21:24:20.103558", "exception": false, "start_time": "2020-12-14T21:24:20.070849", "status": "completed"} tags=[]
efo_terms = {node_id for node_id in graph.nodes.keys()}

# %% papermill={"duration": 0.028124, "end_time": "2020-12-14T21:24:20.173254", "exception": false, "start_time": "2020-12-14T21:24:20.145130", "status": "completed"} tags=[]
len(efo_terms)

# %% papermill={"duration": 0.025146, "end_time": "2020-12-14T21:24:20.211081", "exception": false, "start_time": "2020-12-14T21:24:20.185935", "status": "completed"} tags=[]
graph.nodes["EFO:0000270"]

# %% [markdown] papermill={"duration": 0.012643, "end_time": "2020-12-14T21:24:20.236739", "exception": false, "start_time": "2020-12-14T21:24:20.224096", "status": "completed"} tags=[]
# ## EFO to label

# %% papermill={"duration": 0.061563, "end_time": "2020-12-14T21:24:20.311421", "exception": false, "start_time": "2020-12-14T21:24:20.249858", "status": "completed"} tags=[]
efo_full_data = []

for efo in efo_terms:
    efo_data = {}

    efo_data["term_id"] = efo
    efo_data["label"] = graph.nodes[efo].get("name")

    efo_full_data.append(efo_data)

# %% papermill={"duration": 0.05122, "end_time": "2020-12-14T21:24:20.378692", "exception": false, "start_time": "2020-12-14T21:24:20.327472", "status": "completed"} tags=[]
efo_label = pd.DataFrame(efo_full_data).set_index("term_id")

# %% papermill={"duration": 0.028423, "end_time": "2020-12-14T21:24:20.423820", "exception": false, "start_time": "2020-12-14T21:24:20.395397", "status": "completed"} tags=[]
efo_label.shape

# %% papermill={"duration": 0.032472, "end_time": "2020-12-14T21:24:20.471179", "exception": false, "start_time": "2020-12-14T21:24:20.438707", "status": "completed"} tags=[]
assert efo_label.index.is_unique

# %% papermill={"duration": 0.031068, "end_time": "2020-12-14T21:24:20.519174", "exception": false, "start_time": "2020-12-14T21:24:20.488106", "status": "completed"} tags=[]
efo_label.head()

# %% papermill={"duration": 0.025496, "end_time": "2020-12-14T21:24:20.558890", "exception": false, "start_time": "2020-12-14T21:24:20.533394", "status": "completed"} tags=[]
assert efo_label.loc["EFO:0000270", "label"] == "asthma"

# %% papermill={"duration": 0.128279, "end_time": "2020-12-14T21:24:20.700288", "exception": false, "start_time": "2020-12-14T21:24:20.572009", "status": "completed"} tags=[]
outfile = conf.GENERAL["TERM_ID_LABEL_FILE"]
display(outfile)

efo_label.to_csv(outfile, sep="\t")

# %% [markdown] papermill={"duration": 0.013222, "end_time": "2020-12-14T21:24:20.727011", "exception": false, "start_time": "2020-12-14T21:24:20.713789", "status": "completed"} tags=[]
# ## Map xrefs

# %% papermill={"duration": 0.164207, "end_time": "2020-12-14T21:24:20.905051", "exception": false, "start_time": "2020-12-14T21:24:20.740844", "status": "completed"} tags=[]
efo_full_data = []

for efo in efo_terms:
    efo_data = {}

    efo_data["term_id"] = efo

    for xref_id, xref_data in groupby(graph.nodes[efo].get("xref")).items():
        efo_data["target_id_type"] = xref_id

        for xref in xref_data:
            efo_data["target_id"] = xref
            efo_full_data.append(efo_data.copy())

# %% papermill={"duration": 0.123818, "end_time": "2020-12-14T21:24:21.042557", "exception": false, "start_time": "2020-12-14T21:24:20.918739", "status": "completed"} tags=[]
efo_full_data = pd.DataFrame(efo_full_data).set_index("term_id")

# %% papermill={"duration": 0.031714, "end_time": "2020-12-14T21:24:21.092822", "exception": false, "start_time": "2020-12-14T21:24:21.061108", "status": "completed"} tags=[]
efo_full_data.shape

# %% papermill={"duration": 0.029515, "end_time": "2020-12-14T21:24:21.138757", "exception": false, "start_time": "2020-12-14T21:24:21.109242", "status": "completed"} tags=[]
efo_full_data.head()

# %% papermill={"duration": 0.0266, "end_time": "2020-12-14T21:24:21.179695", "exception": false, "start_time": "2020-12-14T21:24:21.153095", "status": "completed"} tags=[]
graph.nodes["EFO:0002669"]

# %% papermill={"duration": 0.03887, "end_time": "2020-12-14T21:24:21.233631", "exception": false, "start_time": "2020-12-14T21:24:21.194761", "status": "completed"} tags=[]
efo_full_data.loc["EFO:0002669"]

# %% papermill={"duration": 0.034922, "end_time": "2020-12-14T21:24:21.284308", "exception": false, "start_time": "2020-12-14T21:24:21.249386", "status": "completed"} tags=[]
# some testing
assert efo_full_data.loc["EFO:0002669"].shape[0] == 3

_tmp = efo_full_data.loc["EFO:0002669"].sort_values("target_id")

assert _tmp.iloc[0]["target_id_type"] == "NCIt"
assert _tmp.iloc[0]["target_id"] == "NCIt:C2381"

assert _tmp.iloc[1]["target_id_type"] == "SNOMEDCT"
assert _tmp.iloc[1]["target_id"] == "SNOMEDCT:118259007"

assert _tmp.iloc[2]["target_id_type"] == "SNOMEDCT"
assert _tmp.iloc[2]["target_id"] == "SNOMEDCT:387045004"

# %% papermill={"duration": 0.408311, "end_time": "2020-12-14T21:24:21.709317", "exception": false, "start_time": "2020-12-14T21:24:21.301006", "status": "completed"} tags=[]
outfile = conf.GENERAL["TERM_ID_XREFS_FILE"]
display(outfile)

efo_full_data.to_csv(outfile, sep="\t")

# %% papermill={"duration": 0.014078, "end_time": "2020-12-14T21:24:21.738052", "exception": false, "start_time": "2020-12-14T21:24:21.723974", "status": "completed"} tags=[]
