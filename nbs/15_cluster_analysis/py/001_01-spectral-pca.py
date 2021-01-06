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

# %% [markdown] papermill={"duration": 0.055305, "end_time": "2021-01-05T19:14:21.110948", "exception": false, "start_time": "2021-01-05T19:14:21.055643", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.020152, "end_time": "2021-01-05T19:14:21.152191", "exception": false, "start_time": "2021-01-05T19:14:21.132039", "status": "completed"} tags=[]
# Runs spectral clustering on the pca version of the data.

# %% [markdown] papermill={"duration": 0.020438, "end_time": "2021-01-05T19:14:21.193190", "exception": false, "start_time": "2021-01-05T19:14:21.172752", "status": "completed"} tags=[]
# # Environment variables

# %% papermill={"duration": 0.034587, "end_time": "2021-01-05T19:14:21.248342", "exception": false, "start_time": "2021-01-05T19:14:21.213755", "status": "completed"} tags=[]
from IPython.display import display

import conf

N_JOBS = conf.GENERAL["N_JOBS"]
display(N_JOBS)

# %% papermill={"duration": 0.027844, "end_time": "2021-01-05T19:14:21.297804", "exception": false, "start_time": "2021-01-05T19:14:21.269960", "status": "completed"} tags=[]
# %env MKL_NUM_THREADS=$N_JOBS
# %env OPEN_BLAS_NUM_THREADS=$N_JOBS
# %env NUMEXPR_NUM_THREADS=$N_JOBS
# %env OMP_NUM_THREADS=$N_JOBS

# %% [markdown] papermill={"duration": 0.021358, "end_time": "2021-01-05T19:14:21.341147", "exception": false, "start_time": "2021-01-05T19:14:21.319789", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.031426, "end_time": "2021-01-05T19:14:21.393832", "exception": false, "start_time": "2021-01-05T19:14:21.362406", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 1.61203, "end_time": "2021-01-05T19:14:23.026987", "exception": false, "start_time": "2021-01-05T19:14:21.414957", "status": "completed"} tags=[]
from pathlib import Path

import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns

from utils import generate_result_set_name

# %% [markdown] papermill={"duration": 0.021632, "end_time": "2021-01-05T19:14:23.072159", "exception": false, "start_time": "2021-01-05T19:14:23.050527", "status": "completed"} tags=[]
# # Settings

# %% papermill={"duration": 0.034558, "end_time": "2021-01-05T19:14:23.127547", "exception": false, "start_time": "2021-01-05T19:14:23.092989", "status": "completed"} tags=[]
INITIAL_RANDOM_STATE = 40000

# %% [markdown] papermill={"duration": 0.020844, "end_time": "2021-01-05T19:14:23.170029", "exception": false, "start_time": "2021-01-05T19:14:23.149185", "status": "completed"} tags=[]
# ## Input data

# %% papermill={"duration": 0.03545, "end_time": "2021-01-05T19:14:23.226402", "exception": false, "start_time": "2021-01-05T19:14:23.190952", "status": "completed"} tags=[]
INPUT_SUBSET = "pca"

# %% papermill={"duration": 0.036164, "end_time": "2021-01-05T19:14:23.284249", "exception": false, "start_time": "2021-01-05T19:14:23.248085", "status": "completed"} tags=[]
INPUT_STEM = "z_score_std-projection-smultixcan-efo_partial-mashr-zscores"

# %% papermill={"duration": 0.035679, "end_time": "2021-01-05T19:14:23.341650", "exception": false, "start_time": "2021-01-05T19:14:23.305971", "status": "completed"} tags=[]
DR_OPTIONS = {
    "n_components": 50,
    "svd_solver": "full",
    "random_state": 0,
}

# %% papermill={"duration": 0.03747, "end_time": "2021-01-05T19:14:23.400920", "exception": false, "start_time": "2021-01-05T19:14:23.363450", "status": "completed"} tags=[]
input_filepath = Path(
    conf.RESULTS["DATA_TRANSFORMATIONS_DIR"],
    INPUT_SUBSET,
    generate_result_set_name(
        DR_OPTIONS, prefix=f"{INPUT_SUBSET}-{INPUT_STEM}-", suffix=".pkl"
    ),
).resolve()
display(input_filepath)

assert input_filepath.exists(), "Input file does not exist"

input_filepath_stem = input_filepath.stem
display(input_filepath_stem)

# %% [markdown] papermill={"duration": 0.021638, "end_time": "2021-01-05T19:14:23.444911", "exception": false, "start_time": "2021-01-05T19:14:23.423273", "status": "completed"} tags=[]
# ## Clustering

# %% papermill={"duration": 0.042353, "end_time": "2021-01-05T19:14:23.509002", "exception": false, "start_time": "2021-01-05T19:14:23.466649", "status": "completed"} tags=[]
from sklearn.cluster import SpectralClustering

# %% papermill={"duration": 0.035745, "end_time": "2021-01-05T19:14:23.566729", "exception": false, "start_time": "2021-01-05T19:14:23.530984", "status": "completed"} tags=[]
CLUSTERING_ATTRIBUTES_TO_SAVE = ["n_clusters"]

# %% papermill={"duration": 0.036737, "end_time": "2021-01-05T19:14:23.625781", "exception": false, "start_time": "2021-01-05T19:14:23.589044", "status": "completed"} tags=[]
CLUSTERING_OPTIONS = {}

CLUSTERING_OPTIONS["K_MIN"] = 2
CLUSTERING_OPTIONS["K_MAX"] = 60  # sqrt(3749)
CLUSTERING_OPTIONS["N_REPS_PER_K"] = 5
CLUSTERING_OPTIONS["KMEANS_N_INIT"] = 10
CLUSTERING_OPTIONS["N_NEIGHBORS"] = 10
CLUSTERING_OPTIONS["AFFINITY"] = "nearest_neighbors"

display(CLUSTERING_OPTIONS)

# %% papermill={"duration": 0.037913, "end_time": "2021-01-05T19:14:23.685853", "exception": false, "start_time": "2021-01-05T19:14:23.647940", "status": "completed"} tags=[]
CLUSTERERS = {}

idx = 0
random_state = INITIAL_RANDOM_STATE

for k in range(CLUSTERING_OPTIONS["K_MIN"], CLUSTERING_OPTIONS["K_MAX"] + 1):
    for i in range(CLUSTERING_OPTIONS["N_REPS_PER_K"]):
        clus = SpectralClustering(
            n_clusters=k,
            n_init=CLUSTERING_OPTIONS["KMEANS_N_INIT"],
            affinity=CLUSTERING_OPTIONS["AFFINITY"],
            n_neighbors=CLUSTERING_OPTIONS["N_NEIGHBORS"],
            random_state=random_state,
        )

        method_name = type(clus).__name__
        CLUSTERERS[f"{method_name} #{idx}"] = clus

        random_state = random_state + 1
        idx = idx + 1

# %% papermill={"duration": 0.036754, "end_time": "2021-01-05T19:14:23.746279", "exception": false, "start_time": "2021-01-05T19:14:23.709525", "status": "completed"} tags=[]
display(len(CLUSTERERS))

# %% papermill={"duration": 0.039323, "end_time": "2021-01-05T19:14:23.808842", "exception": false, "start_time": "2021-01-05T19:14:23.769519", "status": "completed"} tags=[]
_iter = iter(CLUSTERERS.items())
display(next(_iter))
display(next(_iter))

# %% papermill={"duration": 0.03777, "end_time": "2021-01-05T19:14:23.869599", "exception": false, "start_time": "2021-01-05T19:14:23.831829", "status": "completed"} tags=[]
clustering_method_name = method_name
display(clustering_method_name)

# %% [markdown] papermill={"duration": 0.022196, "end_time": "2021-01-05T19:14:23.914754", "exception": false, "start_time": "2021-01-05T19:14:23.892558", "status": "completed"} tags=[]
# ## Output directory

# %% papermill={"duration": 0.037829, "end_time": "2021-01-05T19:14:23.975161", "exception": false, "start_time": "2021-01-05T19:14:23.937332", "status": "completed"} tags=[]
# output dir for this notebook
RESULTS_DIR = Path(
    conf.RESULTS["CLUSTERING_RUNS_DIR"],
    f"{INPUT_SUBSET}-{INPUT_STEM}",
).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %% [markdown] papermill={"duration": 0.023061, "end_time": "2021-01-05T19:14:24.022551", "exception": false, "start_time": "2021-01-05T19:14:23.999490", "status": "completed"} tags=[]
# # Load input file

# %% papermill={"duration": 0.038766, "end_time": "2021-01-05T19:14:24.084112", "exception": false, "start_time": "2021-01-05T19:14:24.045346", "status": "completed"} tags=[]
data = pd.read_pickle(input_filepath)

# %% papermill={"duration": 0.039318, "end_time": "2021-01-05T19:14:24.147279", "exception": false, "start_time": "2021-01-05T19:14:24.107961", "status": "completed"} tags=[]
data.shape

# %% papermill={"duration": 0.050631, "end_time": "2021-01-05T19:14:24.221515", "exception": false, "start_time": "2021-01-05T19:14:24.170884", "status": "completed"} tags=[]
data.head()

# %% papermill={"duration": 0.038458, "end_time": "2021-01-05T19:14:24.285010", "exception": false, "start_time": "2021-01-05T19:14:24.246552", "status": "completed"} tags=[]
assert not data.isna().any().any()

# %% [markdown] papermill={"duration": 0.024164, "end_time": "2021-01-05T19:14:24.333705", "exception": false, "start_time": "2021-01-05T19:14:24.309541", "status": "completed"} tags=[]
# # Clustering

# %% [markdown] papermill={"duration": 0.022989, "end_time": "2021-01-05T19:14:24.379842", "exception": false, "start_time": "2021-01-05T19:14:24.356853", "status": "completed"} tags=[]
# ## Generate ensemble

# %% papermill={"duration": 0.039992, "end_time": "2021-01-05T19:14:24.442784", "exception": false, "start_time": "2021-01-05T19:14:24.402792", "status": "completed"} tags=[]
from clustering.ensemble import generate_ensemble

# %% papermill={"duration": 1900.901501, "end_time": "2021-01-05T19:46:05.367990", "exception": false, "start_time": "2021-01-05T19:14:24.466489", "status": "completed"} tags=[]
ensemble = generate_ensemble(
    data,
    CLUSTERERS,
    attributes=CLUSTERING_ATTRIBUTES_TO_SAVE,
)

# %% papermill={"duration": 0.084762, "end_time": "2021-01-05T19:46:05.523223", "exception": false, "start_time": "2021-01-05T19:46:05.438461", "status": "completed"} tags=[]
# the number should be close to 295 (the number of partitions generated by k-means/spectral clustering)
ensemble.shape

# %% papermill={"duration": 0.092754, "end_time": "2021-01-05T19:46:05.687994", "exception": false, "start_time": "2021-01-05T19:46:05.595240", "status": "completed"} tags=[]
ensemble.head()

# %% papermill={"duration": 0.088031, "end_time": "2021-01-05T19:46:05.848564", "exception": false, "start_time": "2021-01-05T19:46:05.760533", "status": "completed"} tags=[]
ensemble["n_clusters"].value_counts().head()

# %% papermill={"duration": 0.090298, "end_time": "2021-01-05T19:46:06.011986", "exception": false, "start_time": "2021-01-05T19:46:05.921688", "status": "completed"} tags=[]
ensemble_stats = ensemble["n_clusters"].describe()
display(ensemble_stats)

# %% [markdown] papermill={"duration": 0.07128, "end_time": "2021-01-05T19:46:06.154723", "exception": false, "start_time": "2021-01-05T19:46:06.083443", "status": "completed"} tags=[]
# ## Testing

# %% papermill={"duration": 0.085747, "end_time": "2021-01-05T19:46:06.311241", "exception": false, "start_time": "2021-01-05T19:46:06.225494", "status": "completed"} tags=[]
assert ensemble_stats["min"] > 1

# %% papermill={"duration": 0.086276, "end_time": "2021-01-05T19:46:06.469983", "exception": false, "start_time": "2021-01-05T19:46:06.383707", "status": "completed"} tags=[]
assert not ensemble["n_clusters"].isna().any()

# %% papermill={"duration": 0.085133, "end_time": "2021-01-05T19:46:06.628464", "exception": false, "start_time": "2021-01-05T19:46:06.543331", "status": "completed"} tags=[]
assert ensemble.shape[0] == len(CLUSTERERS)

# %% papermill={"duration": 0.102909, "end_time": "2021-01-05T19:46:06.802954", "exception": false, "start_time": "2021-01-05T19:46:06.700045", "status": "completed"} tags=[]
# all partitions have the right size
assert np.all(
    [part["partition"].shape[0] == data.shape[0] for idx, part in ensemble.iterrows()]
)

# %% papermill={"duration": 0.109501, "end_time": "2021-01-05T19:46:06.984860", "exception": false, "start_time": "2021-01-05T19:46:06.875359", "status": "completed"} tags=[]
# no partition has negative clusters (noisy points)
assert not np.any([(part["partition"] < 0).any() for idx, part in ensemble.iterrows()])

# %% [markdown] papermill={"duration": 0.072223, "end_time": "2021-01-05T19:46:07.129903", "exception": false, "start_time": "2021-01-05T19:46:07.057680", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.086669, "end_time": "2021-01-05T19:46:07.287504", "exception": false, "start_time": "2021-01-05T19:46:07.200835", "status": "completed"} tags=[]
output_filename = Path(
    RESULTS_DIR,
    generate_result_set_name(
        CLUSTERING_OPTIONS,
        prefix=f"{clustering_method_name}-",
        suffix=".pkl",
    ),
).resolve()
display(output_filename)

# %% papermill={"duration": 0.117024, "end_time": "2021-01-05T19:46:07.480301", "exception": false, "start_time": "2021-01-05T19:46:07.363277", "status": "completed"} tags=[]
ensemble.to_pickle(output_filename)

# %% [markdown] papermill={"duration": 0.070734, "end_time": "2021-01-05T19:46:07.626662", "exception": false, "start_time": "2021-01-05T19:46:07.555928", "status": "completed"} tags=[]
# # Stability

# %% [markdown] papermill={"duration": 0.070777, "end_time": "2021-01-05T19:46:07.768485", "exception": false, "start_time": "2021-01-05T19:46:07.697708", "status": "completed"} tags=[]
# ## Group ensemble by n_clusters

# %% papermill={"duration": 0.100658, "end_time": "2021-01-05T19:46:07.940512", "exception": false, "start_time": "2021-01-05T19:46:07.839854", "status": "completed"} tags=[]
parts = ensemble.groupby("n_clusters").apply(
    lambda x: np.concatenate(x["partition"].apply(lambda x: x.reshape(1, -1)), axis=0)
)

# %% papermill={"duration": 0.097412, "end_time": "2021-01-05T19:46:08.109470", "exception": false, "start_time": "2021-01-05T19:46:08.012058", "status": "completed"} tags=[]
parts.head()

# %% papermill={"duration": 0.087205, "end_time": "2021-01-05T19:46:08.268527", "exception": false, "start_time": "2021-01-05T19:46:08.181322", "status": "completed"} tags=[]
assert np.all(
    [
        parts.loc[k].shape == (CLUSTERING_OPTIONS["N_REPS_PER_K"], data.shape[0])
        for k in parts.index
    ]
)

# %% [markdown] papermill={"duration": 0.07202, "end_time": "2021-01-05T19:46:08.415485", "exception": false, "start_time": "2021-01-05T19:46:08.343465", "status": "completed"} tags=[]
# ## Compute stability

# %% papermill={"duration": 0.085631, "end_time": "2021-01-05T19:46:08.572193", "exception": false, "start_time": "2021-01-05T19:46:08.486562", "status": "completed"} tags=[]
from sklearn.metrics import adjusted_rand_score as ari
from scipy.spatial.distance import squareform, pdist

# %% papermill={"duration": 0.62532, "end_time": "2021-01-05T19:46:09.269184", "exception": false, "start_time": "2021-01-05T19:46:08.643864", "status": "completed"} tags=[]
parts_ari = pd.Series(
    {k: pdist(parts.loc[k], metric=ari) for k in parts.index}, name="k"
)

# %% papermill={"duration": 0.089974, "end_time": "2021-01-05T19:46:09.433041", "exception": false, "start_time": "2021-01-05T19:46:09.343067", "status": "completed"} tags=[]
parts_ari_stability = parts_ari.apply(lambda x: x.mean())
display(parts_ari_stability.sort_values(ascending=False).head(15))

# %% papermill={"duration": 0.087458, "end_time": "2021-01-05T19:46:09.592685", "exception": false, "start_time": "2021-01-05T19:46:09.505227", "status": "completed"} tags=[]
parts_ari_df = pd.DataFrame.from_records(parts_ari.tolist()).set_index(
    parts_ari.index.copy()
)

# %% papermill={"duration": 0.087924, "end_time": "2021-01-05T19:46:09.752406", "exception": false, "start_time": "2021-01-05T19:46:09.664482", "status": "completed"} tags=[]
parts_ari_df.shape

# %% papermill={"duration": 0.088196, "end_time": "2021-01-05T19:46:09.915851", "exception": false, "start_time": "2021-01-05T19:46:09.827655", "status": "completed"} tags=[]
assert (
    int(
        (CLUSTERING_OPTIONS["N_REPS_PER_K"] * (CLUSTERING_OPTIONS["N_REPS_PER_K"] - 1))
        / 2
    )
    == parts_ari_df.shape[1]
)

# %% papermill={"duration": 0.094336, "end_time": "2021-01-05T19:46:10.084230", "exception": false, "start_time": "2021-01-05T19:46:09.989894", "status": "completed"} tags=[]
parts_ari_df.head()

# %% [markdown] papermill={"duration": 0.072453, "end_time": "2021-01-05T19:46:10.229817", "exception": false, "start_time": "2021-01-05T19:46:10.157364", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.090504, "end_time": "2021-01-05T19:46:10.392826", "exception": false, "start_time": "2021-01-05T19:46:10.302322", "status": "completed"} tags=[]
output_filename = Path(
    RESULTS_DIR,
    generate_result_set_name(
        CLUSTERING_OPTIONS,
        prefix=f"{clustering_method_name}-stability-",
        suffix=".pkl",
    ),
).resolve()
display(output_filename)

# %% papermill={"duration": 0.087228, "end_time": "2021-01-05T19:46:10.557325", "exception": false, "start_time": "2021-01-05T19:46:10.470097", "status": "completed"} tags=[]
parts_ari_df.to_pickle(output_filename)

# %% [markdown] papermill={"duration": 0.072617, "end_time": "2021-01-05T19:46:10.702670", "exception": false, "start_time": "2021-01-05T19:46:10.630053", "status": "completed"} tags=[]
# ## Stability plot

# %% papermill={"duration": 0.090058, "end_time": "2021-01-05T19:46:10.865657", "exception": false, "start_time": "2021-01-05T19:46:10.775599", "status": "completed"} tags=[]
parts_ari_df_plot = (
    parts_ari_df.stack()
    .reset_index()
    .rename(columns={"level_0": "k", "level_1": "idx", 0: "ari"})
)

# %% papermill={"duration": 0.089395, "end_time": "2021-01-05T19:46:11.028469", "exception": false, "start_time": "2021-01-05T19:46:10.939074", "status": "completed"} tags=[]
parts_ari_df_plot.dtypes

# %% papermill={"duration": 0.090023, "end_time": "2021-01-05T19:46:11.191677", "exception": false, "start_time": "2021-01-05T19:46:11.101654", "status": "completed"} tags=[]
parts_ari_df_plot.head()

# %% papermill={"duration": 3.014, "end_time": "2021-01-05T19:46:14.281250", "exception": false, "start_time": "2021-01-05T19:46:11.267250", "status": "completed"} tags=[]
# with sns.axes_style('whitegrid', {'grid.linestyle': '--'}):
with sns.plotting_context("talk", font_scale=0.75), sns.axes_style(
    "whitegrid", {"grid.linestyle": "--"}
):
    fig = plt.figure(figsize=(12, 6))
    ax = sns.pointplot(data=parts_ari_df_plot, x="k", y="ari")
    ax.set_ylabel("Averange ARI")
    ax.set_xlabel("Number of clusters ($k$)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    #     ax.set_ylim(0.0, 1.0)
    #     ax.set_xlim(CLUSTERING_OPTIONS['K_MIN'], CLUSTERING_OPTIONS['K_MAX'])
    plt.grid(True)
    plt.tight_layout()

# %% papermill={"duration": 0.074281, "end_time": "2021-01-05T19:46:14.432871", "exception": false, "start_time": "2021-01-05T19:46:14.358590", "status": "completed"} tags=[]
