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
# It extracts from the EFO ontology all the xrefs from efo labels to other ontologies/datasets (such as Disease Ontology, ICD9, etc).

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
from collections import defaultdict

from IPython.display import display
import pandas as pd
import obonet

import conf


# %% [markdown] tags=[]
# # Functions

# %% tags=[]
def groupby(data, sep=":"):
    if data is None:
        return {}
    res = defaultdict(set)
    for d in data:
        ds = d.split(sep)
        res[ds[0]].add(d)
    return res


# %% tags=[]
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

# %% tags=[]
_tmp = groupby(test_data)
assert _tmp is not None
assert len(_tmp) == 5

assert len(_tmp["DOID"]) == 1
assert len(_tmp["HP"]) == 1
assert len(_tmp["ICD10"]) == 2
assert len(_tmp["ICD9"]) == 3
assert len(_tmp["KEGG"]) == 1

# %% tags=[]
_tmp


# %% tags=[]
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


# %% [markdown] tags=[]
# # Read the EFO ontology

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
# # Map EFO to other references

# %% tags=[]
efo_terms = {node_id for node_id in graph.nodes.keys()}

# %% tags=[]
len(efo_terms)

# %% tags=[]
graph.nodes["EFO:0000270"]

# %% [markdown] tags=[]
# ## EFO to label

# %% tags=[]
efo_full_data = []

for efo in efo_terms:
    efo_data = {}

    efo_data["term_id"] = efo
    efo_data["label"] = graph.nodes[efo].get("name")

    efo_full_data.append(efo_data)

# %% tags=[]
efo_label = pd.DataFrame(efo_full_data).set_index("term_id")

# %% tags=[]
efo_label.shape

# %% tags=[]
assert efo_label.index.is_unique

# %% tags=[]
efo_label.head()

# %% tags=[]
assert efo_label.loc["EFO:0000270", "label"] == "asthma"

# %% tags=[]
outfile = conf.GENERAL["TERM_ID_LABEL_FILE"]
display(outfile)

efo_label.to_csv(outfile, sep="\t")

# %% [markdown] tags=[]
# ## Map xrefs

# %% tags=[]
efo_full_data = []

for efo in efo_terms:
    efo_data = {}

    efo_data["term_id"] = efo

    for xref_id, xref_data in groupby(graph.nodes[efo].get("xref")).items():
        efo_data["target_id_type"] = xref_id

        for xref in xref_data:
            efo_data["target_id"] = xref
            efo_full_data.append(efo_data.copy())

# %% tags=[]
efo_full_data = pd.DataFrame(efo_full_data).set_index("term_id")

# %% tags=[]
efo_full_data.shape

# %% tags=[]
efo_full_data.head()

# %% tags=[]
graph.nodes["EFO:0002669"]

# %% tags=[]
efo_full_data.loc["EFO:0002669"]

# %% tags=[]
# some testing
assert efo_full_data.loc["EFO:0002669"].shape[0] == 3

_tmp = efo_full_data.loc["EFO:0002669"].sort_values("target_id")

assert _tmp.iloc[0]["target_id_type"] == "NCIt"
assert _tmp.iloc[0]["target_id"] == "NCIt:C2381"

assert _tmp.iloc[1]["target_id_type"] == "SNOMEDCT"
assert _tmp.iloc[1]["target_id"] == "SNOMEDCT:118259007"

assert _tmp.iloc[2]["target_id_type"] == "SNOMEDCT"
assert _tmp.iloc[2]["target_id"] == "SNOMEDCT:387045004"

# %% tags=[]
outfile = conf.GENERAL["TERM_ID_XREFS_FILE"]
display(outfile)

efo_full_data.to_csv(outfile, sep="\t")

# %% tags=[]
