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

# %% [markdown] papermill={"duration": 0.056374, "end_time": "2021-01-05T16:20:34.628974", "exception": false, "start_time": "2021-01-05T16:20:34.572600", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.020912, "end_time": "2021-01-05T16:20:34.671345", "exception": false, "start_time": "2021-01-05T16:20:34.650433", "status": "completed"} tags=[]
# Runs k-means on the z_score_std version of the data.

# %% [markdown] papermill={"duration": 0.020626, "end_time": "2021-01-05T16:20:34.712969", "exception": false, "start_time": "2021-01-05T16:20:34.692343", "status": "completed"} tags=[]
# # Environment variables

# %% papermill={"duration": 0.035166, "end_time": "2021-01-05T16:20:34.768972", "exception": false, "start_time": "2021-01-05T16:20:34.733806", "status": "completed"} tags=[]
from IPython.display import display

import conf

N_JOBS = conf.GENERAL["N_JOBS"]
display(N_JOBS)

# %% papermill={"duration": 0.027891, "end_time": "2021-01-05T16:20:34.818043", "exception": false, "start_time": "2021-01-05T16:20:34.790152", "status": "completed"} tags=[]
# %env MKL_NUM_THREADS=$N_JOBS
# %env OPEN_BLAS_NUM_THREADS=$N_JOBS
# %env NUMEXPR_NUM_THREADS=$N_JOBS
# %env OMP_NUM_THREADS=$N_JOBS

# %% [markdown] papermill={"duration": 0.021149, "end_time": "2021-01-05T16:20:34.860638", "exception": false, "start_time": "2021-01-05T16:20:34.839489", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.030967, "end_time": "2021-01-05T16:20:34.912664", "exception": false, "start_time": "2021-01-05T16:20:34.881697", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 1.618478, "end_time": "2021-01-05T16:20:36.552699", "exception": false, "start_time": "2021-01-05T16:20:34.934221", "status": "completed"} tags=[]
from pathlib import Path

import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns

from utils import generate_result_set_name

# %% [markdown] papermill={"duration": 0.020614, "end_time": "2021-01-05T16:20:36.596676", "exception": false, "start_time": "2021-01-05T16:20:36.576062", "status": "completed"} tags=[]
# # Settings

# %% papermill={"duration": 0.03568, "end_time": "2021-01-05T16:20:36.653369", "exception": false, "start_time": "2021-01-05T16:20:36.617689", "status": "completed"} tags=[]
INITIAL_RANDOM_STATE = 0

# %% [markdown] papermill={"duration": 0.02095, "end_time": "2021-01-05T16:20:36.695480", "exception": false, "start_time": "2021-01-05T16:20:36.674530", "status": "completed"} tags=[]
# ## Input data

# %% papermill={"duration": 0.034997, "end_time": "2021-01-05T16:20:36.751225", "exception": false, "start_time": "2021-01-05T16:20:36.716228", "status": "completed"} tags=[]
INPUT_SUBSET = "z_score_std"

# %% papermill={"duration": 0.035024, "end_time": "2021-01-05T16:20:36.807808", "exception": false, "start_time": "2021-01-05T16:20:36.772784", "status": "completed"} tags=[]
INPUT_STEM = "projection-smultixcan-efo_partial-mashr-zscores"

# %% papermill={"duration": 0.037851, "end_time": "2021-01-05T16:20:36.866673", "exception": false, "start_time": "2021-01-05T16:20:36.828822", "status": "completed"} tags=[]
input_filepath = Path(
    conf.RESULTS["DATA_TRANSFORMATIONS_DIR"],
    INPUT_SUBSET,
    f"{INPUT_SUBSET}-{INPUT_STEM}.pkl",
).resolve()
display(input_filepath)

assert input_filepath.exists(), "Input file does not exist"

input_filepath_stem = input_filepath.stem
display(input_filepath_stem)

# %% [markdown] papermill={"duration": 0.02135, "end_time": "2021-01-05T16:20:36.910185", "exception": false, "start_time": "2021-01-05T16:20:36.888835", "status": "completed"} tags=[]
# ## Clustering

# %% papermill={"duration": 0.04266, "end_time": "2021-01-05T16:20:36.974127", "exception": false, "start_time": "2021-01-05T16:20:36.931467", "status": "completed"} tags=[]
from sklearn.cluster import KMeans

# %% papermill={"duration": 0.035522, "end_time": "2021-01-05T16:20:37.032064", "exception": false, "start_time": "2021-01-05T16:20:36.996542", "status": "completed"} tags=[]
CLUSTERING_ATTRIBUTES_TO_SAVE = ["n_clusters"]

# %% papermill={"duration": 0.037923, "end_time": "2021-01-05T16:20:37.091732", "exception": false, "start_time": "2021-01-05T16:20:37.053809", "status": "completed"} tags=[]
CLUSTERING_OPTIONS = {}

CLUSTERING_OPTIONS["K_MIN"] = 2
CLUSTERING_OPTIONS["K_MAX"] = 60  # sqrt(3749)
CLUSTERING_OPTIONS["N_REPS_PER_K"] = 5
CLUSTERING_OPTIONS["KMEANS_N_INIT"] = 10

display(CLUSTERING_OPTIONS)

# %% papermill={"duration": 0.038227, "end_time": "2021-01-05T16:20:37.153582", "exception": false, "start_time": "2021-01-05T16:20:37.115355", "status": "completed"} tags=[]
CLUSTERERS = {}

idx = 0
random_state = INITIAL_RANDOM_STATE

for k in range(CLUSTERING_OPTIONS["K_MIN"], CLUSTERING_OPTIONS["K_MAX"] + 1):
    for i in range(CLUSTERING_OPTIONS["N_REPS_PER_K"]):
        clus = KMeans(
            n_clusters=k,
            n_init=CLUSTERING_OPTIONS["KMEANS_N_INIT"],
            random_state=random_state,
        )

        method_name = type(clus).__name__
        CLUSTERERS[f"{method_name} #{idx}"] = clus

        random_state = random_state + 1
        idx = idx + 1

# %% papermill={"duration": 0.037225, "end_time": "2021-01-05T16:20:37.213817", "exception": false, "start_time": "2021-01-05T16:20:37.176592", "status": "completed"} tags=[]
display(len(CLUSTERERS))

# %% papermill={"duration": 0.038378, "end_time": "2021-01-05T16:20:37.275014", "exception": false, "start_time": "2021-01-05T16:20:37.236636", "status": "completed"} tags=[]
_iter = iter(CLUSTERERS.items())
display(next(_iter))
display(next(_iter))

# %% papermill={"duration": 0.037848, "end_time": "2021-01-05T16:20:37.335913", "exception": false, "start_time": "2021-01-05T16:20:37.298065", "status": "completed"} tags=[]
clustering_method_name = method_name
display(clustering_method_name)

# %% [markdown] papermill={"duration": 0.022499, "end_time": "2021-01-05T16:20:37.381459", "exception": false, "start_time": "2021-01-05T16:20:37.358960", "status": "completed"} tags=[]
# ## Output directory

# %% papermill={"duration": 0.038389, "end_time": "2021-01-05T16:20:37.442197", "exception": false, "start_time": "2021-01-05T16:20:37.403808", "status": "completed"} tags=[]
# output dir for this notebook
RESULTS_DIR = Path(
    conf.RESULTS["CLUSTERING_RUNS_DIR"],
    f"{INPUT_SUBSET}-{INPUT_STEM}",
).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %% [markdown] papermill={"duration": 0.022469, "end_time": "2021-01-05T16:20:37.487807", "exception": false, "start_time": "2021-01-05T16:20:37.465338", "status": "completed"} tags=[]
# # Load input file

# %% papermill={"duration": 0.05013, "end_time": "2021-01-05T16:20:37.560596", "exception": false, "start_time": "2021-01-05T16:20:37.510466", "status": "completed"} tags=[]
data = pd.read_pickle(input_filepath)

# %% papermill={"duration": 0.038859, "end_time": "2021-01-05T16:20:37.625801", "exception": false, "start_time": "2021-01-05T16:20:37.586942", "status": "completed"} tags=[]
data.shape

# %% papermill={"duration": 0.052568, "end_time": "2021-01-05T16:20:37.702600", "exception": false, "start_time": "2021-01-05T16:20:37.650032", "status": "completed"} tags=[]
data.head()

# %% papermill={"duration": 0.043809, "end_time": "2021-01-05T16:20:37.771192", "exception": false, "start_time": "2021-01-05T16:20:37.727383", "status": "completed"} tags=[]
assert not data.isna().any().any()

# %% [markdown] papermill={"duration": 0.023273, "end_time": "2021-01-05T16:20:37.818733", "exception": false, "start_time": "2021-01-05T16:20:37.795460", "status": "completed"} tags=[]
# # Clustering

# %% [markdown] papermill={"duration": 0.023099, "end_time": "2021-01-05T16:20:37.865366", "exception": false, "start_time": "2021-01-05T16:20:37.842267", "status": "completed"} tags=[]
# ## Generate ensemble

# %% papermill={"duration": 0.040809, "end_time": "2021-01-05T16:20:37.929680", "exception": false, "start_time": "2021-01-05T16:20:37.888871", "status": "completed"} tags=[]
from clustering.ensemble import generate_ensemble

# %% papermill={"duration": 1301.458246, "end_time": "2021-01-05T16:42:19.411438", "exception": false, "start_time": "2021-01-05T16:20:37.953192", "status": "completed"} tags=[]
ensemble = generate_ensemble(
    data,
    CLUSTERERS,
    attributes=CLUSTERING_ATTRIBUTES_TO_SAVE,
)

# %% papermill={"duration": 0.085355, "end_time": "2021-01-05T16:42:19.568581", "exception": false, "start_time": "2021-01-05T16:42:19.483226", "status": "completed"} tags=[]
# the number should be close to 295 (the number of partitions generated by k-means/spectral clustering)
ensemble.shape

# %% papermill={"duration": 0.092303, "end_time": "2021-01-05T16:42:19.733155", "exception": false, "start_time": "2021-01-05T16:42:19.640852", "status": "completed"} tags=[]
ensemble.head()

# %% papermill={"duration": 0.085965, "end_time": "2021-01-05T16:42:19.890893", "exception": false, "start_time": "2021-01-05T16:42:19.804928", "status": "completed"} tags=[]
ensemble["n_clusters"].value_counts().head()

# %% papermill={"duration": 0.088694, "end_time": "2021-01-05T16:42:20.050892", "exception": false, "start_time": "2021-01-05T16:42:19.962198", "status": "completed"} tags=[]
ensemble_stats = ensemble["n_clusters"].describe()
display(ensemble_stats)

# %% [markdown] papermill={"duration": 0.070719, "end_time": "2021-01-05T16:42:20.193996", "exception": false, "start_time": "2021-01-05T16:42:20.123277", "status": "completed"} tags=[]
# ## Testing

# %% papermill={"duration": 0.089101, "end_time": "2021-01-05T16:42:20.354495", "exception": false, "start_time": "2021-01-05T16:42:20.265394", "status": "completed"} tags=[]
assert ensemble_stats["min"] > 1

# %% papermill={"duration": 0.085281, "end_time": "2021-01-05T16:42:20.511977", "exception": false, "start_time": "2021-01-05T16:42:20.426696", "status": "completed"} tags=[]
assert not ensemble["n_clusters"].isna().any()

# %% papermill={"duration": 0.085608, "end_time": "2021-01-05T16:42:20.670304", "exception": false, "start_time": "2021-01-05T16:42:20.584696", "status": "completed"} tags=[]
assert ensemble.shape[0] == len(CLUSTERERS)

# %% papermill={"duration": 0.103283, "end_time": "2021-01-05T16:42:20.845957", "exception": false, "start_time": "2021-01-05T16:42:20.742674", "status": "completed"} tags=[]
# all partitions have the right size
assert np.all(
    [part["partition"].shape[0] == data.shape[0] for idx, part in ensemble.iterrows()]
)

# %% papermill={"duration": 0.106855, "end_time": "2021-01-05T16:42:21.024442", "exception": false, "start_time": "2021-01-05T16:42:20.917587", "status": "completed"} tags=[]
# no partition has negative clusters (noisy points)
assert not np.any([(part["partition"] < 0).any() for idx, part in ensemble.iterrows()])

# %% [markdown] papermill={"duration": 0.071845, "end_time": "2021-01-05T16:42:21.167894", "exception": false, "start_time": "2021-01-05T16:42:21.096049", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.08855, "end_time": "2021-01-05T16:42:21.326802", "exception": false, "start_time": "2021-01-05T16:42:21.238252", "status": "completed"} tags=[]
output_filename = Path(
    RESULTS_DIR,
    generate_result_set_name(
        CLUSTERING_OPTIONS,
        prefix=f"{clustering_method_name}-",
        suffix=".pkl",
    ),
).resolve()
display(output_filename)

# %% papermill={"duration": 0.093416, "end_time": "2021-01-05T16:42:21.493843", "exception": false, "start_time": "2021-01-05T16:42:21.400427", "status": "completed"} tags=[]
ensemble.to_pickle(output_filename)

# %% [markdown] papermill={"duration": 0.07049, "end_time": "2021-01-05T16:42:21.636125", "exception": false, "start_time": "2021-01-05T16:42:21.565635", "status": "completed"} tags=[]
# # Stability

# %% [markdown] papermill={"duration": 0.070327, "end_time": "2021-01-05T16:42:21.777512", "exception": false, "start_time": "2021-01-05T16:42:21.707185", "status": "completed"} tags=[]
# ## Group ensemble by n_clusters

# %% papermill={"duration": 0.106012, "end_time": "2021-01-05T16:42:21.954556", "exception": false, "start_time": "2021-01-05T16:42:21.848544", "status": "completed"} tags=[]
parts = ensemble.groupby("n_clusters").apply(
    lambda x: np.concatenate(x["partition"].apply(lambda x: x.reshape(1, -1)), axis=0)
)

# %% papermill={"duration": 0.09716, "end_time": "2021-01-05T16:42:22.123525", "exception": false, "start_time": "2021-01-05T16:42:22.026365", "status": "completed"} tags=[]
parts.head()

# %% papermill={"duration": 0.08814, "end_time": "2021-01-05T16:42:22.283679", "exception": false, "start_time": "2021-01-05T16:42:22.195539", "status": "completed"} tags=[]
assert np.all(
    [
        parts.loc[k].shape == (CLUSTERING_OPTIONS["N_REPS_PER_K"], data.shape[0])
        for k in parts.index
    ]
)

# %% [markdown] papermill={"duration": 0.071814, "end_time": "2021-01-05T16:42:22.432432", "exception": false, "start_time": "2021-01-05T16:42:22.360618", "status": "completed"} tags=[]
# ## Compute stability

# %% papermill={"duration": 0.086059, "end_time": "2021-01-05T16:42:22.590025", "exception": false, "start_time": "2021-01-05T16:42:22.503966", "status": "completed"} tags=[]
from sklearn.metrics import adjusted_rand_score as ari
from scipy.spatial.distance import squareform, pdist

# %% papermill={"duration": 0.769453, "end_time": "2021-01-05T16:42:23.431733", "exception": false, "start_time": "2021-01-05T16:42:22.662280", "status": "completed"} tags=[]
parts_ari = pd.Series(
    {k: pdist(parts.loc[k], metric=ari) for k in parts.index}, name="k"
)

# %% papermill={"duration": 0.088318, "end_time": "2021-01-05T16:42:23.591070", "exception": false, "start_time": "2021-01-05T16:42:23.502752", "status": "completed"} tags=[]
parts_ari_stability = parts_ari.apply(lambda x: x.mean())
display(parts_ari_stability.sort_values(ascending=False).head(15))

# %% papermill={"duration": 0.086964, "end_time": "2021-01-05T16:42:23.751060", "exception": false, "start_time": "2021-01-05T16:42:23.664096", "status": "completed"} tags=[]
parts_ari_df = pd.DataFrame.from_records(parts_ari.tolist()).set_index(
    parts_ari.index.copy()
)

# %% papermill={"duration": 0.086952, "end_time": "2021-01-05T16:42:23.910198", "exception": false, "start_time": "2021-01-05T16:42:23.823246", "status": "completed"} tags=[]
parts_ari_df.shape

# %% papermill={"duration": 0.08615, "end_time": "2021-01-05T16:42:24.069581", "exception": false, "start_time": "2021-01-05T16:42:23.983431", "status": "completed"} tags=[]
assert (
    int(
        (CLUSTERING_OPTIONS["N_REPS_PER_K"] * (CLUSTERING_OPTIONS["N_REPS_PER_K"] - 1))
        / 2
    )
    == parts_ari_df.shape[1]
)

# %% papermill={"duration": 0.093592, "end_time": "2021-01-05T16:42:24.237476", "exception": false, "start_time": "2021-01-05T16:42:24.143884", "status": "completed"} tags=[]
parts_ari_df.head()

# %% [markdown] papermill={"duration": 0.074686, "end_time": "2021-01-05T16:42:24.387151", "exception": false, "start_time": "2021-01-05T16:42:24.312465", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.08788, "end_time": "2021-01-05T16:42:24.547684", "exception": false, "start_time": "2021-01-05T16:42:24.459804", "status": "completed"} tags=[]
output_filename = Path(
    RESULTS_DIR,
    generate_result_set_name(
        CLUSTERING_OPTIONS,
        prefix=f"{clustering_method_name}-stability-",
        suffix=".pkl",
    ),
).resolve()
display(output_filename)

# %% papermill={"duration": 0.088711, "end_time": "2021-01-05T16:42:24.710569", "exception": false, "start_time": "2021-01-05T16:42:24.621858", "status": "completed"} tags=[]
parts_ari_df.to_pickle(output_filename)

# %% [markdown] papermill={"duration": 0.072476, "end_time": "2021-01-05T16:42:24.856137", "exception": false, "start_time": "2021-01-05T16:42:24.783661", "status": "completed"} tags=[]
# ## Stability plot

# %% papermill={"duration": 0.088381, "end_time": "2021-01-05T16:42:25.017365", "exception": false, "start_time": "2021-01-05T16:42:24.928984", "status": "completed"} tags=[]
parts_ari_df_plot = (
    parts_ari_df.stack()
    .reset_index()
    .rename(columns={"level_0": "k", "level_1": "idx", 0: "ari"})
)

# %% papermill={"duration": 0.088293, "end_time": "2021-01-05T16:42:25.178491", "exception": false, "start_time": "2021-01-05T16:42:25.090198", "status": "completed"} tags=[]
parts_ari_df_plot.dtypes

# %% papermill={"duration": 0.093011, "end_time": "2021-01-05T16:42:25.345292", "exception": false, "start_time": "2021-01-05T16:42:25.252281", "status": "completed"} tags=[]
parts_ari_df_plot.head()

# %% papermill={"duration": 3.046143, "end_time": "2021-01-05T16:42:28.466836", "exception": false, "start_time": "2021-01-05T16:42:25.420693", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.074239, "end_time": "2021-01-05T16:42:28.615857", "exception": false, "start_time": "2021-01-05T16:42:28.541618", "status": "completed"} tags=[]
