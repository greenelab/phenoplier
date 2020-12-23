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

# %% [markdown] papermill={"duration": 0.036847, "end_time": "2020-12-23T17:59:44.910201", "exception": false, "start_time": "2020-12-23T17:59:44.873354", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.01954, "end_time": "2020-12-23T17:59:44.949431", "exception": false, "start_time": "2020-12-23T17:59:44.929891", "status": "completed"} tags=[]
# **TODO**: should probably be moved to preprocessing folder.

# %% [markdown] papermill={"duration": 0.019372, "end_time": "2020-12-23T17:59:44.988666", "exception": false, "start_time": "2020-12-23T17:59:44.969294", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.030808, "end_time": "2020-12-23T17:59:45.039115", "exception": false, "start_time": "2020-12-23T17:59:45.008307", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 0.543113, "end_time": "2020-12-23T17:59:45.601825", "exception": false, "start_time": "2020-12-23T17:59:45.058712", "status": "completed"} tags=[]
from pathlib import Path

# import numpy as np
import pandas as pd

import conf

# from multiplier import MultiplierProjection
from entity import Gene

# from data.cache import read_data
# from data.hdf5 import simplify_trait_fullcode, HDF5_FILE_PATTERN

# %% [markdown] papermill={"duration": 0.021813, "end_time": "2020-12-23T17:59:45.646617", "exception": false, "start_time": "2020-12-23T17:59:45.624804", "status": "completed"} tags=[]
# # Settings

# %% papermill={"duration": 0.036862, "end_time": "2020-12-23T17:59:45.704068", "exception": false, "start_time": "2020-12-23T17:59:45.667206", "status": "completed"} tags=[]
OUTPUT_DIR = conf.RESULTS["DRUG_DISEASE_ANALYSES"]
display(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% papermill={"duration": 0.031739, "end_time": "2020-12-23T17:59:45.758022", "exception": false, "start_time": "2020-12-23T17:59:45.726283", "status": "completed"} tags=[]
OUTPUT_DATA_DIR = Path(OUTPUT_DIR, "data")
display(OUTPUT_DATA_DIR)
OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# %% papermill={"duration": 0.038548, "end_time": "2020-12-23T17:59:45.819245", "exception": false, "start_time": "2020-12-23T17:59:45.780697", "status": "completed"} tags=[]
OUTPUT_RAW_DATA_DIR = Path(OUTPUT_DATA_DIR, "raw")
display(OUTPUT_RAW_DATA_DIR)
OUTPUT_RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# %% papermill={"duration": 0.034022, "end_time": "2020-12-23T17:59:45.879585", "exception": false, "start_time": "2020-12-23T17:59:45.845563", "status": "completed"} tags=[]
OUTPUT_PROJ_DATA_DIR = Path(OUTPUT_DATA_DIR, "proj")
display(OUTPUT_PROJ_DATA_DIR)
OUTPUT_PROJ_DATA_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown] papermill={"duration": 0.020194, "end_time": "2020-12-23T17:59:45.922357", "exception": false, "start_time": "2020-12-23T17:59:45.902163", "status": "completed"} tags=[]
# # Load LINCS consensi drugbank (Daniel)

# %% papermill={"duration": 0.033279, "end_time": "2020-12-23T17:59:45.975703", "exception": false, "start_time": "2020-12-23T17:59:45.942424", "status": "completed"} tags=[]
# TODO: hardcoded
input_file = Path(
    conf.DATA_DIR, "hetionet", "lincs-v2.0", "consensi-drugbank.tsv.bz2"
).resolve()

display(input_file)

# %% papermill={"duration": 6.341891, "end_time": "2020-12-23T17:59:52.341308", "exception": false, "start_time": "2020-12-23T17:59:45.999417", "status": "completed"} tags=[]
lincs_data = pd.read_csv(input_file, sep="\t", index_col="perturbagen").T

# %% papermill={"duration": 0.038749, "end_time": "2020-12-23T17:59:52.420231", "exception": false, "start_time": "2020-12-23T17:59:52.381482", "status": "completed"} tags=[]
lincs_data.shape

# %% papermill={"duration": 0.054224, "end_time": "2020-12-23T17:59:52.515848", "exception": false, "start_time": "2020-12-23T17:59:52.461624", "status": "completed"} tags=[]
lincs_data.head()

# %% papermill={"duration": 0.036989, "end_time": "2020-12-23T17:59:52.579939", "exception": false, "start_time": "2020-12-23T17:59:52.542950", "status": "completed"} tags=[]
assert lincs_data.index.is_unique

# %% papermill={"duration": 0.073849, "end_time": "2020-12-23T17:59:52.697701", "exception": false, "start_time": "2020-12-23T17:59:52.623852", "status": "completed"} tags=[]
# drubback ids are consistent
_tmp = lincs_data.columns.map(len).unique()
assert _tmp.shape[0] == 1

# %% papermill={"duration": 0.038076, "end_time": "2020-12-23T17:59:52.761595", "exception": false, "start_time": "2020-12-23T17:59:52.723519", "status": "completed"} tags=[]
assert lincs_data.columns.is_unique

# %% [markdown] papermill={"duration": 0.02401, "end_time": "2020-12-23T17:59:52.811513", "exception": false, "start_time": "2020-12-23T17:59:52.787503", "status": "completed"} tags=[]
# ## Gene IDs to Gene names

# %% papermill={"duration": 0.400361, "end_time": "2020-12-23T17:59:53.235188", "exception": false, "start_time": "2020-12-23T17:59:52.834827", "status": "completed"} tags=[]
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

pandas2ri.activate()

# %% papermill={"duration": 4.504344, "end_time": "2020-12-23T17:59:57.760360", "exception": false, "start_time": "2020-12-23T17:59:53.256016", "status": "completed"} tags=[]
clusterProfiler = importr("clusterProfiler")

# %% papermill={"duration": 0.043901, "end_time": "2020-12-23T17:59:57.839489", "exception": false, "start_time": "2020-12-23T17:59:57.795588", "status": "completed"} tags=[]
# lincs_data = lincs_data.rename(columns=GENE_ENTREZ_ID_TO_SYMBOL).rename(columns=GENE_SYMBOL_TO_ENSEMBL_ID).T

# %% papermill={"duration": 0.044084, "end_time": "2020-12-23T17:59:57.928243", "exception": false, "start_time": "2020-12-23T17:59:57.884159", "status": "completed"} tags=[]
# _gene_id_len = lincs_data.index.map(len)

# %% papermill={"duration": 0.037882, "end_time": "2020-12-23T17:59:57.992535", "exception": false, "start_time": "2020-12-23T17:59:57.954653", "status": "completed"} tags=[]
# _not_mapped_genes = lincs_data[_gene_id_len != 15].index.copy()

# %% papermill={"duration": 0.074281, "end_time": "2020-12-23T17:59:58.110711", "exception": false, "start_time": "2020-12-23T17:59:58.036430", "status": "completed"} tags=[]
# _not_mapped_genes

# %% papermill={"duration": 0.65365, "end_time": "2020-12-23T17:59:58.795278", "exception": false, "start_time": "2020-12-23T17:59:58.141628", "status": "completed"} tags=[]
_now_mapped_genes = clusterProfiler.bitr(
    lincs_data.index.tolist(),
    fromType="ENTREZID",
    toType="ENSEMBL",
    OrgDb="org.Hs.eg.db",
)

# %% papermill={"duration": 0.051759, "end_time": "2020-12-23T17:59:58.882413", "exception": false, "start_time": "2020-12-23T17:59:58.830654", "status": "completed"} tags=[]
_now_mapped_genes.shape

# %% papermill={"duration": 0.053269, "end_time": "2020-12-23T17:59:58.993145", "exception": false, "start_time": "2020-12-23T17:59:58.939876", "status": "completed"} tags=[]
# some genes have entrez ids that map to several ensembl id
display(_now_mapped_genes[_now_mapped_genes["ENTREZID"].duplicated(keep=False)])

# %% papermill={"duration": 1.528715, "end_time": "2020-12-23T18:00:00.556149", "exception": false, "start_time": "2020-12-23T17:59:59.027434", "status": "completed"} tags=[]
_now_mapped_genes = _now_mapped_genes.assign(
    in_phenomexcan=_now_mapped_genes["ENSEMBL"].apply(
        lambda x: x in Gene.GENE_ID_TO_NAME_MAP
    )
)

# %% papermill={"duration": 0.052244, "end_time": "2020-12-23T18:00:00.643276", "exception": false, "start_time": "2020-12-23T18:00:00.591032", "status": "completed"} tags=[]
_now_mapped_genes[_now_mapped_genes["in_phenomexcan"]].shape

# %% papermill={"duration": 0.060632, "end_time": "2020-12-23T18:00:00.749218", "exception": false, "start_time": "2020-12-23T18:00:00.688586", "status": "completed"} tags=[]
_now_mapped_genes.head()

# %% papermill={"duration": 0.042846, "end_time": "2020-12-23T18:00:00.819718", "exception": false, "start_time": "2020-12-23T18:00:00.776872", "status": "completed"} tags=[]
_now_mapped_genes = _now_mapped_genes[
    _now_mapped_genes["in_phenomexcan"]
].drop_duplicates(subset=["ENTREZID"])

# %% papermill={"duration": 0.039809, "end_time": "2020-12-23T18:00:00.892459", "exception": false, "start_time": "2020-12-23T18:00:00.852650", "status": "completed"} tags=[]
_now_mapped_genes.shape

# %% papermill={"duration": 0.063656, "end_time": "2020-12-23T18:00:00.993545", "exception": false, "start_time": "2020-12-23T18:00:00.929889", "status": "completed"} tags=[]
_now_mapped_genes.head()

# %% papermill={"duration": 0.082962, "end_time": "2020-12-23T18:00:01.107730", "exception": false, "start_time": "2020-12-23T18:00:01.024768", "status": "completed"} tags=[]
_now_mapped_genes_dict = _now_mapped_genes.set_index("ENTREZID").to_dict()["ENSEMBL"]

# %% papermill={"duration": 0.125291, "end_time": "2020-12-23T18:00:01.272338", "exception": false, "start_time": "2020-12-23T18:00:01.147047", "status": "completed"} tags=[]
lincs_data = lincs_data.loc[_now_mapped_genes_dict.keys()].rename(
    index=_now_mapped_genes_dict
)

# %% papermill={"duration": 0.111345, "end_time": "2020-12-23T18:00:01.432290", "exception": false, "start_time": "2020-12-23T18:00:01.320945", "status": "completed"} tags=[]
lincs_data.head()

# %% papermill={"duration": 0.078877, "end_time": "2020-12-23T18:00:01.776510", "exception": false, "start_time": "2020-12-23T18:00:01.697633", "status": "completed"} tags=[]
# make sure we have ensembl id only in the index
_tmp = pd.Series(lincs_data.index.map(len)).value_counts()
display(_tmp)
assert _tmp.shape[0] == 1

# %% papermill={"duration": 0.089339, "end_time": "2020-12-23T18:00:01.905209", "exception": false, "start_time": "2020-12-23T18:00:01.815870", "status": "completed"} tags=[]
lincs_data.head()

# %% [markdown] papermill={"duration": 0.027059, "end_time": "2020-12-23T18:00:01.959515", "exception": false, "start_time": "2020-12-23T18:00:01.932456", "status": "completed"} tags=[]
# ## Remove NaN values

# %% papermill={"duration": 0.045542, "end_time": "2020-12-23T18:00:02.032083", "exception": false, "start_time": "2020-12-23T18:00:01.986541", "status": "completed"} tags=[]
assert not lincs_data.isna().any().any()

# %% [markdown] papermill={"duration": 0.025209, "end_time": "2020-12-23T18:00:02.083337", "exception": false, "start_time": "2020-12-23T18:00:02.058128", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.04023, "end_time": "2020-12-23T18:00:02.151139", "exception": false, "start_time": "2020-12-23T18:00:02.110909", "status": "completed"} tags=[]
output_file = Path(OUTPUT_RAW_DATA_DIR, "lincs-data.pkl").resolve()
display(output_file)

# %% papermill={"duration": 0.110899, "end_time": "2020-12-23T18:00:02.291906", "exception": false, "start_time": "2020-12-23T18:00:02.181007", "status": "completed"} tags=[]
lincs_data.to_pickle(output_file)

# %% [markdown] papermill={"duration": 0.026448, "end_time": "2020-12-23T18:00:02.346307", "exception": false, "start_time": "2020-12-23T18:00:02.319859", "status": "completed"} tags=[]
# # Project into MultiPLIER

# %% papermill={"duration": 0.057628, "end_time": "2020-12-23T18:00:02.430151", "exception": false, "start_time": "2020-12-23T18:00:02.372523", "status": "completed"} tags=[]
from multiplier import MultiplierProjection

# %% papermill={"duration": 0.05842, "end_time": "2020-12-23T18:00:02.517458", "exception": false, "start_time": "2020-12-23T18:00:02.459038", "status": "completed"} tags=[]
mproj = MultiplierProjection()

# %% papermill={"duration": 2.322917, "end_time": "2020-12-23T18:00:04.889131", "exception": false, "start_time": "2020-12-23T18:00:02.566214", "status": "completed"} tags=[]
lincs_projection = mproj.transform(lincs_data)

# %% papermill={"duration": 0.079896, "end_time": "2020-12-23T18:00:05.025899", "exception": false, "start_time": "2020-12-23T18:00:04.946003", "status": "completed"} tags=[]
lincs_projection.shape

# %% papermill={"duration": 0.073552, "end_time": "2020-12-23T18:00:05.137120", "exception": false, "start_time": "2020-12-23T18:00:05.063568", "status": "completed"} tags=[]
lincs_projection.head()

# %% [markdown] papermill={"duration": 0.057015, "end_time": "2020-12-23T18:00:05.223461", "exception": false, "start_time": "2020-12-23T18:00:05.166446", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.069707, "end_time": "2020-12-23T18:00:05.350231", "exception": false, "start_time": "2020-12-23T18:00:05.280524", "status": "completed"} tags=[]
output_file = Path(OUTPUT_PROJ_DATA_DIR, "lincs-projection.pkl").resolve()
display(output_file)

# %% papermill={"duration": 0.075996, "end_time": "2020-12-23T18:00:05.475118", "exception": false, "start_time": "2020-12-23T18:00:05.399122", "status": "completed"} tags=[]
lincs_projection.to_pickle(output_file)

# %% [markdown] papermill={"duration": 0.050132, "end_time": "2020-12-23T18:00:05.563642", "exception": false, "start_time": "2020-12-23T18:00:05.513510", "status": "completed"} tags=[]
# # Get reconstructed data

# %% papermill={"duration": 0.06718, "end_time": "2020-12-23T18:00:05.672551", "exception": false, "start_time": "2020-12-23T18:00:05.605371", "status": "completed"} tags=[]
# lincs_data_recon = MultiplierProjection._read_model_z().rename(index=GENE_SYMBOL_TO_ENSEMBL_ID).dot(lincs_projection)

# %% papermill={"duration": 0.080537, "end_time": "2020-12-23T18:00:05.794090", "exception": false, "start_time": "2020-12-23T18:00:05.713553", "status": "completed"} tags=[]
# lincs_data_recon.shape

# %% papermill={"duration": 0.074678, "end_time": "2020-12-23T18:00:05.918868", "exception": false, "start_time": "2020-12-23T18:00:05.844190", "status": "completed"} tags=[]
# lincs_data_recon.head()

# %% papermill={"duration": 0.097779, "end_time": "2020-12-23T18:00:06.058773", "exception": false, "start_time": "2020-12-23T18:00:05.960994", "status": "completed"} tags=[]
