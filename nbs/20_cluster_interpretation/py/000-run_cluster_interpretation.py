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
# TODO: this notebook analyze all clusters from the selected partitions

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
# import re
import shutil
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# import numpy as np
import pandas as pd
import papermill as pm

# import matplotlib.pyplot as plt
# import seaborn as sns
# from IPython.display import HTML

# from clustering.methods import ClusterInterpreter
# from data.recount2 import LVAnalysis
# from data.cache import read_data
# from utils import generate_result_set_name
import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=["parameters"]
# select which partitions' clusters will be analyzed
PARTITION_Ks = [45, 41, 38, 28]

# %% [markdown] tags=[]
# # Load best partitions

# %% tags=[]
# output dir for this notebook
CONSENSUS_CLUSTERING_DIR = Path(
    conf.RESULTS["CLUSTERING_DIR"], "consensus_clustering"
).resolve()

display(CONSENSUS_CLUSTERING_DIR)

# %% tags=[]
input_file = Path(CONSENSUS_CLUSTERING_DIR, "best_partitions_by_k.pkl").resolve()
display(input_file)

# %% tags=[]
best_partitions = pd.read_pickle(input_file)

# %% tags=[]
assert best_partitions.index.is_unique

# %% tags=[]
best_partitions.shape

# %% tags=[]
best_partitions.head()

# %% [markdown] tags=[]
# # Select top k partitions

# %% tags=[]
# I take the top 4 partitions (according to their number of clusters)
selected_partition_ks = best_partitions[best_partitions["selected"]].index.sort_values(
    ascending=False
)[:4]
display(selected_partition_ks)

# %% [markdown] tags=[]
# # Run interpretation

# %% tags=[]
CLUSTER_ANALYSIS_OUTPUT_DIR = Path(
    conf.RESULTS["CLUSTERING_INTERPRETATION_OUTPUT_DIR"],
    "cluster_analyses",
).resolve()
display(CLUSTER_ANALYSIS_OUTPUT_DIR)

# %% tags=[]
CLUSTER_ANALYSIS_OUTPUT_DIR.mkdir(exist_ok=True)


# %% tags=[]
def run_notebook(input_nb, output_nb, parameters):
    pm.execute_notebook(
        input_nb,
        output_nb,
        progress_bar=False,
        parameters=parameters,
    )


# %% tags=[]
for part_k in selected_partition_ks:
    print(f"Partition k:{part_k}", flush=True)

    output_folder = Path(CLUSTER_ANALYSIS_OUTPUT_DIR, f"part{part_k}").resolve()
    shutil.rmtree(output_folder, ignore_errors=True)
    output_folder.mkdir()

    part = best_partitions.loc[part_k, "partition"]
    part_clusters = pd.Series(part).value_counts()

    # always skip the biggest cluster in each partition
    for c_size_idx, c in enumerate(part_clusters.index[1:]):
        print(f"  Cluster: {c}", flush=True)

        input_nb = Path(
            conf.RESULTS["CLUSTERING_INTERPRETATION_OUTPUT_DIR"],
            "interpret_cluster.run.ipynb",
        ).resolve()

        output_nb = Path(
            output_folder, f"{c_size_idx:02}-part{part_k}_k{c}.ipynb"
        ).resolve()

        parameters = dict(PARTITION_K=part_k, PARTITION_CLUSTER_ID=c)

        run_notebook(input_nb, output_nb, parameters)

# %% tags=[]
