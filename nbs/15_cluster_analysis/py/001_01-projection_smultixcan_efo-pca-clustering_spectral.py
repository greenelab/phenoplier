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

# %% [markdown] papermill={"duration": 0.053394, "end_time": "2020-12-02T21:15:53.150852", "exception": false, "start_time": "2020-12-02T21:15:53.097458", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.017413, "end_time": "2020-12-02T21:15:53.187308", "exception": false, "start_time": "2020-12-02T21:15:53.169895", "status": "completed"} tags=[]
# Runs spectral clustering on the pca version of the data.

# %% [markdown] papermill={"duration": 0.017264, "end_time": "2020-12-02T21:15:53.222074", "exception": false, "start_time": "2020-12-02T21:15:53.204810", "status": "completed"} tags=[]
# # Environment variables

# %% papermill={"duration": 0.031034, "end_time": "2020-12-02T21:15:53.270518", "exception": false, "start_time": "2020-12-02T21:15:53.239484", "status": "completed"} tags=[]
from IPython.display import display

import conf

N_JOBS = conf.GENERAL["N_JOBS"]
display(N_JOBS)

# %% papermill={"duration": 0.024863, "end_time": "2020-12-02T21:15:53.314234", "exception": false, "start_time": "2020-12-02T21:15:53.289371", "status": "completed"} tags=[]
# %env MKL_NUM_THREADS=$N_JOBS
# %env OPEN_BLAS_NUM_THREADS=$N_JOBS
# %env NUMEXPR_NUM_THREADS=$N_JOBS
# %env OMP_NUM_THREADS=$N_JOBS

# %% [markdown] papermill={"duration": 0.018056, "end_time": "2020-12-02T21:15:53.351015", "exception": false, "start_time": "2020-12-02T21:15:53.332959", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.028618, "end_time": "2020-12-02T21:15:53.397565", "exception": false, "start_time": "2020-12-02T21:15:53.368947", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 1.615896, "end_time": "2020-12-02T21:15:55.031991", "exception": false, "start_time": "2020-12-02T21:15:53.416095", "status": "completed"} tags=[]
from pathlib import Path

import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns

from utils import generate_result_set_name

# %% [markdown] papermill={"duration": 0.019147, "end_time": "2020-12-02T21:15:55.072144", "exception": false, "start_time": "2020-12-02T21:15:55.052997", "status": "completed"} tags=[]
# # Settings

# %% papermill={"duration": 0.032505, "end_time": "2020-12-02T21:15:55.123079", "exception": false, "start_time": "2020-12-02T21:15:55.090574", "status": "completed"} tags=[]
INITIAL_RANDOM_STATE = 40000

# %% [markdown] papermill={"duration": 0.018224, "end_time": "2020-12-02T21:15:55.160030", "exception": false, "start_time": "2020-12-02T21:15:55.141806", "status": "completed"} tags=[]
# ## Input data

# %% papermill={"duration": 0.032891, "end_time": "2020-12-02T21:15:55.211276", "exception": false, "start_time": "2020-12-02T21:15:55.178385", "status": "completed"} tags=[]
INPUT_SUBSET = "pca"

# %% papermill={"duration": 0.032551, "end_time": "2020-12-02T21:15:55.262811", "exception": false, "start_time": "2020-12-02T21:15:55.230260", "status": "completed"} tags=[]
INPUT_STEM = "z_score_std-projection-smultixcan-efo_partial-mashr-zscores"

# %% papermill={"duration": 0.032922, "end_time": "2020-12-02T21:15:55.315029", "exception": false, "start_time": "2020-12-02T21:15:55.282107", "status": "completed"} tags=[]
DR_OPTIONS = {
    "n_components": 50,
    "svd_solver": "full",
    "random_state": 0,
}

# %% papermill={"duration": 0.034568, "end_time": "2020-12-02T21:15:55.368271", "exception": false, "start_time": "2020-12-02T21:15:55.333703", "status": "completed"} tags=[]
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

# %% [markdown] papermill={"duration": 0.019151, "end_time": "2020-12-02T21:15:55.407063", "exception": false, "start_time": "2020-12-02T21:15:55.387912", "status": "completed"} tags=[]
# ## Clustering

# %% papermill={"duration": 0.039418, "end_time": "2020-12-02T21:15:55.465531", "exception": false, "start_time": "2020-12-02T21:15:55.426113", "status": "completed"} tags=[]
from sklearn.cluster import SpectralClustering

# %% papermill={"duration": 0.033343, "end_time": "2020-12-02T21:15:55.518167", "exception": false, "start_time": "2020-12-02T21:15:55.484824", "status": "completed"} tags=[]
CLUSTERING_ATTRIBUTES_TO_SAVE = ["n_clusters"]

# %% papermill={"duration": 0.034319, "end_time": "2020-12-02T21:15:55.572159", "exception": false, "start_time": "2020-12-02T21:15:55.537840", "status": "completed"} tags=[]
CLUSTERING_OPTIONS = {}

CLUSTERING_OPTIONS["K_MIN"] = 2
CLUSTERING_OPTIONS["K_MAX"] = 60  # sqrt(3749)
CLUSTERING_OPTIONS["N_REPS_PER_K"] = 5
CLUSTERING_OPTIONS["KMEANS_N_INIT"] = 10
CLUSTERING_OPTIONS["N_NEIGHBORS"] = 10
CLUSTERING_OPTIONS["AFFINITY"] = "nearest_neighbors"

display(CLUSTERING_OPTIONS)

# %% papermill={"duration": 0.035942, "end_time": "2020-12-02T21:15:55.628396", "exception": false, "start_time": "2020-12-02T21:15:55.592454", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.03366, "end_time": "2020-12-02T21:15:55.681352", "exception": false, "start_time": "2020-12-02T21:15:55.647692", "status": "completed"} tags=[]
display(len(CLUSTERERS))

# %% papermill={"duration": 0.036324, "end_time": "2020-12-02T21:15:55.737659", "exception": false, "start_time": "2020-12-02T21:15:55.701335", "status": "completed"} tags=[]
_iter = iter(CLUSTERERS.items())
display(next(_iter))
display(next(_iter))

# %% papermill={"duration": 0.035186, "end_time": "2020-12-02T21:15:55.793292", "exception": false, "start_time": "2020-12-02T21:15:55.758106", "status": "completed"} tags=[]
clustering_method_name = method_name
display(clustering_method_name)

# %% [markdown] papermill={"duration": 0.020245, "end_time": "2020-12-02T21:15:55.835758", "exception": false, "start_time": "2020-12-02T21:15:55.815513", "status": "completed"} tags=[]
# ## Output directory

# %% papermill={"duration": 0.036104, "end_time": "2020-12-02T21:15:55.892321", "exception": false, "start_time": "2020-12-02T21:15:55.856217", "status": "completed"} tags=[]
# output dir for this notebook
RESULTS_DIR = Path(
    conf.RESULTS["CLUSTERING_RUNS_DIR"],
    f"{INPUT_SUBSET}-{INPUT_STEM}",
).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %% [markdown] papermill={"duration": 0.020733, "end_time": "2020-12-02T21:15:55.934518", "exception": false, "start_time": "2020-12-02T21:15:55.913785", "status": "completed"} tags=[]
# # Load input file

# %% papermill={"duration": 0.036581, "end_time": "2020-12-02T21:15:55.991503", "exception": false, "start_time": "2020-12-02T21:15:55.954922", "status": "completed"} tags=[]
data = pd.read_pickle(input_filepath)

# %% papermill={"duration": 0.036283, "end_time": "2020-12-02T21:15:56.049184", "exception": false, "start_time": "2020-12-02T21:15:56.012901", "status": "completed"} tags=[]
data.shape

# %% papermill={"duration": 0.048125, "end_time": "2020-12-02T21:15:56.118491", "exception": false, "start_time": "2020-12-02T21:15:56.070366", "status": "completed"} tags=[]
data.head()

# %% papermill={"duration": 0.036099, "end_time": "2020-12-02T21:15:56.176296", "exception": false, "start_time": "2020-12-02T21:15:56.140197", "status": "completed"} tags=[]
assert not data.isna().any().any()

# %% [markdown] papermill={"duration": 0.021583, "end_time": "2020-12-02T21:15:56.220283", "exception": false, "start_time": "2020-12-02T21:15:56.198700", "status": "completed"} tags=[]
# # Clustering

# %% [markdown] papermill={"duration": 0.020945, "end_time": "2020-12-02T21:15:56.262564", "exception": false, "start_time": "2020-12-02T21:15:56.241619", "status": "completed"} tags=[]
# ## Generate ensemble

# %% papermill={"duration": 0.038524, "end_time": "2020-12-02T21:15:56.322449", "exception": false, "start_time": "2020-12-02T21:15:56.283925", "status": "completed"} tags=[]
from clustering.ensemble import generate_ensemble

# %% papermill={"duration": 1897.410176, "end_time": "2020-12-02T21:47:33.754310", "exception": false, "start_time": "2020-12-02T21:15:56.344134", "status": "completed"} tags=[]
ensemble = generate_ensemble(
    data,
    CLUSTERERS,
    attributes=CLUSTERING_ATTRIBUTES_TO_SAVE,
)

# %% papermill={"duration": 0.083247, "end_time": "2020-12-02T21:47:33.906972", "exception": false, "start_time": "2020-12-02T21:47:33.823725", "status": "completed"} tags=[]
# the number should be close to 295 (the number of partitions generated by k-means/spectral clustering)
ensemble.shape

# %% papermill={"duration": 0.09055, "end_time": "2020-12-02T21:47:34.067003", "exception": false, "start_time": "2020-12-02T21:47:33.976453", "status": "completed"} tags=[]
ensemble.head()

# %% papermill={"duration": 0.084085, "end_time": "2020-12-02T21:47:34.220231", "exception": false, "start_time": "2020-12-02T21:47:34.136146", "status": "completed"} tags=[]
ensemble["n_clusters"].value_counts().head()

# %% papermill={"duration": 0.085022, "end_time": "2020-12-02T21:47:34.373569", "exception": false, "start_time": "2020-12-02T21:47:34.288547", "status": "completed"} tags=[]
ensemble_stats = ensemble["n_clusters"].describe()
display(ensemble_stats)

# %% [markdown] papermill={"duration": 0.070943, "end_time": "2020-12-02T21:47:34.520590", "exception": false, "start_time": "2020-12-02T21:47:34.449647", "status": "completed"} tags=[]
# ## Testing

# %% papermill={"duration": 0.082896, "end_time": "2020-12-02T21:47:34.672374", "exception": false, "start_time": "2020-12-02T21:47:34.589478", "status": "completed"} tags=[]
assert ensemble_stats["min"] > 1

# %% papermill={"duration": 0.083601, "end_time": "2020-12-02T21:47:34.825147", "exception": false, "start_time": "2020-12-02T21:47:34.741546", "status": "completed"} tags=[]
assert not ensemble["n_clusters"].isna().any()

# %% papermill={"duration": 0.084823, "end_time": "2020-12-02T21:47:34.979396", "exception": false, "start_time": "2020-12-02T21:47:34.894573", "status": "completed"} tags=[]
assert ensemble.shape[0] == len(CLUSTERERS)

# %% papermill={"duration": 0.101211, "end_time": "2020-12-02T21:47:35.149484", "exception": false, "start_time": "2020-12-02T21:47:35.048273", "status": "completed"} tags=[]
# all partitions have the right size
assert np.all(
    [part["partition"].shape[0] == data.shape[0] for idx, part in ensemble.iterrows()]
)

# %% papermill={"duration": 0.104723, "end_time": "2020-12-02T21:47:35.324950", "exception": false, "start_time": "2020-12-02T21:47:35.220227", "status": "completed"} tags=[]
# no partition has negative clusters (noisy points)
assert not np.any([(part["partition"] < 0).any() for idx, part in ensemble.iterrows()])

# %% [markdown] papermill={"duration": 0.071837, "end_time": "2020-12-02T21:47:35.466320", "exception": false, "start_time": "2020-12-02T21:47:35.394483", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.085337, "end_time": "2020-12-02T21:47:35.620717", "exception": false, "start_time": "2020-12-02T21:47:35.535380", "status": "completed"} tags=[]
output_filename = Path(
    RESULTS_DIR,
    generate_result_set_name(
        CLUSTERING_OPTIONS,
        prefix=f"{clustering_method_name}-",
        suffix=".pkl",
    ),
).resolve()
display(output_filename)

# %% papermill={"duration": 0.107797, "end_time": "2020-12-02T21:47:35.799202", "exception": false, "start_time": "2020-12-02T21:47:35.691405", "status": "completed"} tags=[]
ensemble.to_pickle(output_filename)

# %% [markdown] papermill={"duration": 0.069897, "end_time": "2020-12-02T21:47:35.940220", "exception": false, "start_time": "2020-12-02T21:47:35.870323", "status": "completed"} tags=[]
# # Stability

# %% [markdown] papermill={"duration": 0.068861, "end_time": "2020-12-02T21:47:36.079534", "exception": false, "start_time": "2020-12-02T21:47:36.010673", "status": "completed"} tags=[]
# ## Group ensemble by n_clusters

# %% papermill={"duration": 0.098662, "end_time": "2020-12-02T21:47:36.248171", "exception": false, "start_time": "2020-12-02T21:47:36.149509", "status": "completed"} tags=[]
parts = ensemble.groupby("n_clusters").apply(
    lambda x: np.concatenate(x["partition"].apply(lambda x: x.reshape(1, -1)), axis=0)
)

# %% papermill={"duration": 0.095699, "end_time": "2020-12-02T21:47:36.414008", "exception": false, "start_time": "2020-12-02T21:47:36.318309", "status": "completed"} tags=[]
parts.head()

# %% papermill={"duration": 0.085796, "end_time": "2020-12-02T21:47:36.572520", "exception": false, "start_time": "2020-12-02T21:47:36.486724", "status": "completed"} tags=[]
assert np.all(
    [
        parts.loc[k].shape == (CLUSTERING_OPTIONS["N_REPS_PER_K"], data.shape[0])
        for k in parts.index
    ]
)

# %% [markdown] papermill={"duration": 0.069384, "end_time": "2020-12-02T21:47:36.712063", "exception": false, "start_time": "2020-12-02T21:47:36.642679", "status": "completed"} tags=[]
# ## Compute stability

# %% papermill={"duration": 0.084138, "end_time": "2020-12-02T21:47:36.865295", "exception": false, "start_time": "2020-12-02T21:47:36.781157", "status": "completed"} tags=[]
from sklearn.metrics import adjusted_rand_score as ari
from scipy.spatial.distance import squareform, pdist

# %% papermill={"duration": 0.633007, "end_time": "2020-12-02T21:47:37.568584", "exception": false, "start_time": "2020-12-02T21:47:36.935577", "status": "completed"} tags=[]
parts_ari = pd.Series(
    {k: pdist(parts.loc[k], metric=ari) for k in parts.index}, name="k"
)

# %% papermill={"duration": 0.085558, "end_time": "2020-12-02T21:47:37.723749", "exception": false, "start_time": "2020-12-02T21:47:37.638191", "status": "completed"} tags=[]
parts_ari_stability = parts_ari.apply(lambda x: x.mean())
display(parts_ari_stability.sort_values(ascending=False).head(15))

# %% papermill={"duration": 0.08689, "end_time": "2020-12-02T21:47:37.881473", "exception": false, "start_time": "2020-12-02T21:47:37.794583", "status": "completed"} tags=[]
parts_ari_df = pd.DataFrame.from_records(parts_ari.tolist()).set_index(
    parts_ari.index.copy()
)

# %% papermill={"duration": 0.086903, "end_time": "2020-12-02T21:47:38.040008", "exception": false, "start_time": "2020-12-02T21:47:37.953105", "status": "completed"} tags=[]
parts_ari_df.shape

# %% papermill={"duration": 0.084811, "end_time": "2020-12-02T21:47:38.196596", "exception": false, "start_time": "2020-12-02T21:47:38.111785", "status": "completed"} tags=[]
assert (
    int(
        (CLUSTERING_OPTIONS["N_REPS_PER_K"] * (CLUSTERING_OPTIONS["N_REPS_PER_K"] - 1))
        / 2
    )
    == parts_ari_df.shape[1]
)

# %% papermill={"duration": 0.092612, "end_time": "2020-12-02T21:47:38.360115", "exception": false, "start_time": "2020-12-02T21:47:38.267503", "status": "completed"} tags=[]
parts_ari_df.head()

# %% [markdown] papermill={"duration": 0.074298, "end_time": "2020-12-02T21:47:38.506658", "exception": false, "start_time": "2020-12-02T21:47:38.432360", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.086457, "end_time": "2020-12-02T21:47:38.664217", "exception": false, "start_time": "2020-12-02T21:47:38.577760", "status": "completed"} tags=[]
output_filename = Path(
    RESULTS_DIR,
    generate_result_set_name(
        CLUSTERING_OPTIONS,
        prefix=f"{clustering_method_name}-stability-",
        suffix=".pkl",
    ),
).resolve()
display(output_filename)

# %% papermill={"duration": 0.08601, "end_time": "2020-12-02T21:47:38.822172", "exception": false, "start_time": "2020-12-02T21:47:38.736162", "status": "completed"} tags=[]
parts_ari_df.to_pickle(output_filename)

# %% [markdown] papermill={"duration": 0.076685, "end_time": "2020-12-02T21:47:38.990992", "exception": false, "start_time": "2020-12-02T21:47:38.914307", "status": "completed"} tags=[]
# ## Stability plot

# %% papermill={"duration": 0.088017, "end_time": "2020-12-02T21:47:39.150339", "exception": false, "start_time": "2020-12-02T21:47:39.062322", "status": "completed"} tags=[]
parts_ari_df_plot = (
    parts_ari_df.stack()
    .reset_index()
    .rename(columns={"level_0": "k", "level_1": "idx", 0: "ari"})
)

# %% papermill={"duration": 0.089092, "end_time": "2020-12-02T21:47:39.313359", "exception": false, "start_time": "2020-12-02T21:47:39.224267", "status": "completed"} tags=[]
parts_ari_df_plot.dtypes

# %% papermill={"duration": 0.092469, "end_time": "2020-12-02T21:47:39.478845", "exception": false, "start_time": "2020-12-02T21:47:39.386376", "status": "completed"} tags=[]
parts_ari_df_plot.head()

# %% papermill={"duration": 2.993668, "end_time": "2020-12-02T21:47:42.546130", "exception": false, "start_time": "2020-12-02T21:47:39.552462", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.073155, "end_time": "2020-12-02T21:47:42.694468", "exception": false, "start_time": "2020-12-02T21:47:42.621313", "status": "completed"} tags=[]
