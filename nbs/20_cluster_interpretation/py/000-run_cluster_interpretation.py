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
# This notebook reads clustering results taking the top 4 partitions with more clusters, and analyzes each cluster providing a list of latent variables (LV) that are driving that cluster. For example, for the hypertension traits, it might find an LV with genes expressed in cardiomyocytes or other potentially related cell types.
#
# It uses the `papermill` API to run the notebook `interpret_cluster.run.ipynb` (which serves as a template) for each cluster. Results are saved in folder `cluster_analyses`.

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
import os
import shutil
from multiprocessing import Pool
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import papermill as pm

import conf

# %% [markdown] tags=[]
# # Load best partitions

# %% tags=[]
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
# I take the top 4 partitions (according to their number of clusters).
# These are the partitions that will be analyzed in the manuscript.
selected_partition_ks = best_partitions[best_partitions["selected"]].index.sort_values(
    ascending=False
)[:4]
display(selected_partition_ks)

# %% [markdown] tags=[]
# # Run interpretation

# %% tags=[]
CLUSTER_ANALYSIS_OUTPUT_DIR = Path(
    conf.RESULTS["CLUSTERING_INTERPRETATION"]["CLUSTERS_STATS"],
    "cluster_analyses",
).resolve()
display(CLUSTER_ANALYSIS_OUTPUT_DIR)

# %% tags=[]
CLUSTER_ANALYSIS_OUTPUT_DIR.mkdir(exist_ok=True)


# %% tags=[]
def run_notebook(input_nb, output_nb, parameters, environment):
    options = []
    for k, v in parameters.items():
        options.append("-p")
        options.append(str(k))
        options.append(str(v))

    cmdlist = (
        ["papermill"]
        + [
            f"'{input_nb}'",
            f"'{output_nb}'",
        ]
        + options
    )
    cmdlist = " ".join(cmdlist)

    res = subprocess.run(
        cmdlist,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=environment,
        text=True,
    )
    return cmdlist, res


# %% tags=[]
tasks = {}

with Pool(conf.GENERAL["N_JOBS"]) as pool:
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
                conf.RESULTS["CLUSTERING_INTERPRETATION"]["CLUSTERS_STATS"],
                "interpret_cluster.run.ipynb",
            ).resolve()

            output_nb = Path(
                output_folder, f"{c_size_idx:02}-part{part_k}_k{c}.ipynb"
            ).resolve()

            parameters = dict(PARTITION_K=part_k, PARTITION_CLUSTER_ID=c)

            res = pool.apply_async(
                run_notebook,
                (
                    input_nb,
                    output_nb,
                    parameters,
                    {k: v for k, v in os.environ.items()},
                ),
            )
            tasks[f"{part_k}_k{c}"] = res

    pool.close()

    # show errors, if any
    for k, t in tasks.items():
        t.wait()

        cmd, out = t.get()
        if out.returncode != 0:
            display(k)
            print(cmd)
            print(out.stdout)

            pool.terminate()
            break

# %% tags=[]
