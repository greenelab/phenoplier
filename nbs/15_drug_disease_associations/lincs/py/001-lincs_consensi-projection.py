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
# **TODO**: should probably be moved to preprocessing folder.

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
from pathlib import Path

# import numpy as np
import pandas as pd

import conf

# from multiplier import MultiplierProjection
from entity import Gene

# from data.cache import read_data
# from data.hdf5 import simplify_trait_fullcode, HDF5_FILE_PATTERN

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
OUTPUT_DIR = conf.RESULTS["DRUG_DISEASE_ANALYSES"]
display(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% tags=[]
OUTPUT_DATA_DIR = Path(OUTPUT_DIR, "data")
display(OUTPUT_DATA_DIR)
OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# %% tags=[]
OUTPUT_RAW_DATA_DIR = Path(OUTPUT_DATA_DIR, "raw")
display(OUTPUT_RAW_DATA_DIR)
OUTPUT_RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# %% tags=[]
OUTPUT_PROJ_DATA_DIR = Path(OUTPUT_DATA_DIR, "proj")
display(OUTPUT_PROJ_DATA_DIR)
OUTPUT_PROJ_DATA_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown] tags=[]
# # Load LINCS consensi drugbank (Daniel)

# %% tags=[]
# TODO: hardcoded
input_file = Path(
    conf.DATA_DIR, "hetionet", "lincs-v2.0", "consensi-drugbank.tsv.bz2"
).resolve()

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
# drubback ids are consistent
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
# lincs_data = lincs_data.rename(columns=GENE_ENTREZ_ID_TO_SYMBOL).rename(columns=GENE_SYMBOL_TO_ENSEMBL_ID).T

# %% tags=[]
# _gene_id_len = lincs_data.index.map(len)

# %% tags=[]
# _not_mapped_genes = lincs_data[_gene_id_len != 15].index.copy()

# %% tags=[]
# _not_mapped_genes

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
output_file = Path(OUTPUT_RAW_DATA_DIR, "lincs-data.pkl").resolve()
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

# %% [markdown] tags=[]
# ## Save

# %% tags=[]
output_file = Path(OUTPUT_PROJ_DATA_DIR, "lincs-projection.pkl").resolve()
display(output_file)

# %% tags=[]
lincs_projection.to_pickle(output_file)

# %% [markdown] tags=[]
# # Get reconstructed data

# %% tags=[]
# lincs_data_recon = MultiplierProjection._read_model_z().rename(index=GENE_SYMBOL_TO_ENSEMBL_ID).dot(lincs_projection)

# %% tags=[]
# lincs_data_recon.shape

# %% tags=[]
# lincs_data_recon.head()

# %% tags=[]
