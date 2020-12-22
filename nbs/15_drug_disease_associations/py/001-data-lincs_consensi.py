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

# %% [markdown] papermill={"duration": 0.044577, "end_time": "2020-12-18T22:38:21.345879", "exception": false, "start_time": "2020-12-18T22:38:21.301302", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.011764, "end_time": "2020-12-18T22:38:21.398073", "exception": false, "start_time": "2020-12-18T22:38:21.386309", "status": "completed"} tags=[]
# **TODO**: should probably be moved to preprocessing folder.

# %% [markdown] papermill={"duration": 0.011799, "end_time": "2020-12-18T22:38:21.422136", "exception": false, "start_time": "2020-12-18T22:38:21.410337", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.022467, "end_time": "2020-12-18T22:38:21.456126", "exception": false, "start_time": "2020-12-18T22:38:21.433659", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 0.192251, "end_time": "2020-12-18T22:38:21.659996", "exception": false, "start_time": "2020-12-18T22:38:21.467745", "status": "completed"} tags=[]
from pathlib import Path

# import numpy as np
import pandas as pd

import conf
# from multiplier import MultiplierProjection
from entity import Gene
# from data.cache import read_data
# from data.hdf5 import simplify_trait_fullcode, HDF5_FILE_PATTERN

# %% [markdown] papermill={"duration": 0.011416, "end_time": "2020-12-18T22:38:21.683356", "exception": false, "start_time": "2020-12-18T22:38:21.671940", "status": "completed"} tags=[]
# # Settings

# %%
OUTPUT_DIR = conf.RESULTS["DRUG_DISEASE_ANALYSES"]
display(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
OUTPUT_DATA_DIR = Path(OUTPUT_DIR, "data")
display(OUTPUT_DATA_DIR)
OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# %%
OUTPUT_RAW_DATA_DIR = Path(OUTPUT_DATA_DIR, "raw")
display(OUTPUT_RAW_DATA_DIR)
OUTPUT_RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# %%
OUTPUT_PROJ_DATA_DIR = Path(OUTPUT_DATA_DIR, "proj")
display(OUTPUT_PROJ_DATA_DIR)
OUTPUT_PROJ_DATA_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# # Load LINCS consensi drugbank (Daniel)

# %%
# TODO: hardcoded
input_file = Path(
    conf.DATA_DIR,
    "hetionet",
    "lincs-v2.0",
    "consensi-drugbank.tsv.bz2"
).resolve()

display(input_file)

# %%
lincs_data = pd.read_csv(input_file, sep='\t', index_col='perturbagen').T

# %%
lincs_data.shape

# %%
lincs_data.head()

# %%
assert lincs_data.index.is_unique

# %%
# drubback ids are consistent
_tmp = lincs_data.columns.map(len).unique()
assert _tmp.shape[0] == 1

# %%
assert lincs_data.columns.is_unique

# %% [markdown]
# ## Gene IDs to Gene names

# %%
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
pandas2ri.activate()

# %%
clusterProfiler = importr("clusterProfiler")

# %%
# lincs_data = lincs_data.rename(columns=GENE_ENTREZ_ID_TO_SYMBOL).rename(columns=GENE_SYMBOL_TO_ENSEMBL_ID).T

# %%
# _gene_id_len = lincs_data.index.map(len)

# %%
# _not_mapped_genes = lincs_data[_gene_id_len != 15].index.copy()

# %%
# _not_mapped_genes

# %%
_now_mapped_genes = clusterProfiler.bitr(lincs_data.index.tolist(), fromType='ENTREZID', toType='ENSEMBL', OrgDb="org.Hs.eg.db")

# %%
_now_mapped_genes.shape

# %%
# some genes have entrez ids that map to several ensembl id
display(_now_mapped_genes[_now_mapped_genes['ENTREZID'].duplicated(keep=False)])

# %%
_now_mapped_genes = _now_mapped_genes.assign(in_phenomexcan=_now_mapped_genes['ENSEMBL'].apply(lambda x: x in Gene.GENE_ID_TO_NAME_MAP))

# %%
_now_mapped_genes[_now_mapped_genes['in_phenomexcan']].shape

# %%
_now_mapped_genes.head()

# %%
_now_mapped_genes = _now_mapped_genes[_now_mapped_genes['in_phenomexcan']].drop_duplicates(subset=['ENTREZID'])

# %%
_now_mapped_genes.shape

# %%
_now_mapped_genes.head()

# %%
_now_mapped_genes_dict = _now_mapped_genes.set_index('ENTREZID').to_dict()['ENSEMBL']

# %%
lincs_data = lincs_data.loc[_now_mapped_genes_dict.keys()].rename(index=_now_mapped_genes_dict)

# %%
lincs_data.head()

# %%
# make sure we have ensembl id only in the index
_tmp = pd.Series(lincs_data.index.map(len)).value_counts()
display(_tmp)
assert _tmp.shape[0] == 1

# %%
lincs_data.head()

# %% [markdown]
# ## Remove NaN values

# %%
assert not lincs_data.isna().any().any()

# %% [markdown]
# ## Save

# %%
output_file = Path(
    OUTPUT_RAW_DATA_DIR,
    "lincs-data.pkl"
).resolve()
display(output_file)

# %%
lincs_data.to_pickle(output_file)

# %% [markdown]
# # Project into MultiPLIER

# %%
from multiplier import MultiplierProjection

# %%
mproj = MultiplierProjection()

# %%
lincs_projection = mproj.transform(lincs_data)

# %%
lincs_projection.shape

# %%
lincs_projection.head()

# %% [markdown]
# ## Save

# %%
output_file = Path(
    OUTPUT_PROJ_DATA_DIR,
    "lincs-projection.pkl"
).resolve()
display(output_file)

# %%
lincs_projection.to_pickle(output_file)

# %% [markdown]
# # Get reconstructed data

# %%
# lincs_data_recon = MultiplierProjection._read_model_z().rename(index=GENE_SYMBOL_TO_ENSEMBL_ID).dot(lincs_projection)

# %%
# lincs_data_recon.shape

# %%
# lincs_data_recon.head()

# %%
