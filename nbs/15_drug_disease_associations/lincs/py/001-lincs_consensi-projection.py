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
# This notebook process the LINCS data consensus signatures from [here](https://figshare.com/articles/dataset/Consensus_signatures_for_LINCS_L1000_perturbations/3085426/1).

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
from pathlib import Path

import pandas as pd

import conf
from entity import Gene

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
OUTPUT_DATA_DIR = Path(conf.RESULTS["DRUG_DISEASE_ANALYSES"], "lincs")
display(OUTPUT_DATA_DIR)
OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown] tags=[]
# # Load LINCS consensus signatures

# %% tags=[]
input_file = conf.LINCS["CONSENSUS_SIGNATURES_FILE"]

display(input_file)

# %% tags=[]
lincs_data = pd.read_csv(input_file, sep="\t", index_col="perturbagen").T

# %% tags=[]
lincs_data.shape

# %% tags=[]
lincs_data.head()

# %% tags=[]
assert lincs_data.index.is_unique

# %% tags=[]
# check that DrugBank ids are consistent
_tmp = lincs_data.columns.map(len).unique()
assert _tmp.shape[0] == 1

# %% tags=[]
assert lincs_data.columns.is_unique

# %% [markdown] tags=[]
# ## Gene IDs to Gene names

# %% tags=[]
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

pandas2ri.activate()

# %% tags=[]
clusterProfiler = importr("clusterProfiler")

# %% tags=[]
_now_mapped_genes = clusterProfiler.bitr(
    lincs_data.index.tolist(),
    fromType="ENTREZID",
    toType="ENSEMBL",
    OrgDb="org.Hs.eg.db",
)

# %% tags=[]
_now_mapped_genes.shape

# %% tags=[]
# some genes have entrez ids that map to several ensembl id
display(_now_mapped_genes[_now_mapped_genes["ENTREZID"].duplicated(keep=False)])

# %% tags=[]
_now_mapped_genes = _now_mapped_genes.assign(
    in_phenomexcan=_now_mapped_genes["ENSEMBL"].apply(
        lambda x: x in Gene.GENE_ID_TO_NAME_MAP
    )
)

# %% tags=[]
_now_mapped_genes[_now_mapped_genes["in_phenomexcan"]].shape

# %% tags=[]
_now_mapped_genes.head()

# %% tags=[]
_now_mapped_genes = _now_mapped_genes[
    _now_mapped_genes["in_phenomexcan"]
].drop_duplicates(subset=["ENTREZID"])

# %% tags=[]
_now_mapped_genes.shape

# %% tags=[]
_now_mapped_genes.head()

# %% tags=[]
_now_mapped_genes_dict = _now_mapped_genes.set_index("ENTREZID").to_dict()["ENSEMBL"]

# %% tags=[]
lincs_data = lincs_data.loc[_now_mapped_genes_dict.keys()].rename(
    index=_now_mapped_genes_dict
)

# %% tags=[]
lincs_data.head()

# %% tags=[]
# make sure we have ensembl id only in the index
_tmp = pd.Series(lincs_data.index.map(len)).value_counts()
display(_tmp)
assert _tmp.shape[0] == 1

# %% tags=[]
lincs_data.head()

# %% [markdown] tags=[]
# ## Remove NaN values

# %% tags=[]
assert not lincs_data.isna().any().any()

# %% [markdown] tags=[]
# ## Save

# %% tags=[]
output_file = Path(OUTPUT_DATA_DIR, "lincs-data.pkl").resolve()
display(output_file)

# %% tags=[]
lincs_data.to_pickle(output_file)

# %% [markdown] tags=[]
# # Project into MultiPLIER

# %% tags=[]
from multiplier import MultiplierProjection

# %% tags=[]
mproj = MultiplierProjection()

# %% tags=[]
lincs_projection = mproj.transform(lincs_data)

# %% tags=[]
lincs_projection.shape

# %% tags=[]
lincs_projection.head()

# %%
assert not lincs_projection.isna().any().any()

# %% [markdown] tags=[]
# ## Save

# %% tags=[]
output_file = Path(OUTPUT_DATA_DIR, "lincs-projection.pkl").resolve()
display(output_file)

# %% tags=[]
lincs_projection.to_pickle(output_file)

# %% tags=[]
