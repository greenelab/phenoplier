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

# %% [markdown] papermill={"duration": 0.057243, "end_time": "2021-01-05T16:55:10.311987", "exception": false, "start_time": "2021-01-05T16:55:10.254744", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.020594, "end_time": "2021-01-05T16:55:10.353943", "exception": false, "start_time": "2021-01-05T16:55:10.333349", "status": "completed"} tags=[]
# Runs spectral clustering on the z_score_std version of the data.

# %% [markdown] papermill={"duration": 0.020651, "end_time": "2021-01-05T16:55:10.394933", "exception": false, "start_time": "2021-01-05T16:55:10.374282", "status": "completed"} tags=[]
# # Environment variables

# %% papermill={"duration": 0.037623, "end_time": "2021-01-05T16:55:10.453161", "exception": false, "start_time": "2021-01-05T16:55:10.415538", "status": "completed"} tags=[]
from IPython.display import display

import conf

N_JOBS = conf.GENERAL["N_JOBS"]
display(N_JOBS)

# %% papermill={"duration": 0.028398, "end_time": "2021-01-05T16:55:10.503784", "exception": false, "start_time": "2021-01-05T16:55:10.475386", "status": "completed"} tags=[]
# %env MKL_NUM_THREADS=$N_JOBS
# %env OPEN_BLAS_NUM_THREADS=$N_JOBS
# %env NUMEXPR_NUM_THREADS=$N_JOBS
# %env OMP_NUM_THREADS=$N_JOBS

# %% [markdown] papermill={"duration": 0.021406, "end_time": "2021-01-05T16:55:10.546824", "exception": false, "start_time": "2021-01-05T16:55:10.525418", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.031792, "end_time": "2021-01-05T16:55:10.599645", "exception": false, "start_time": "2021-01-05T16:55:10.567853", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 1.612945, "end_time": "2021-01-05T16:55:12.234231", "exception": false, "start_time": "2021-01-05T16:55:10.621286", "status": "completed"} tags=[]
from pathlib import Path

import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns

from utils import generate_result_set_name

# %% [markdown] papermill={"duration": 0.021197, "end_time": "2021-01-05T16:55:12.279038", "exception": false, "start_time": "2021-01-05T16:55:12.257841", "status": "completed"} tags=[]
# # Settings

# %% papermill={"duration": 0.036062, "end_time": "2021-01-05T16:55:12.337127", "exception": false, "start_time": "2021-01-05T16:55:12.301065", "status": "completed"} tags=[]
INITIAL_RANDOM_STATE = 30000

# %% [markdown] papermill={"duration": 0.020806, "end_time": "2021-01-05T16:55:12.379122", "exception": false, "start_time": "2021-01-05T16:55:12.358316", "status": "completed"} tags=[]
# ## Input data

# %% papermill={"duration": 0.035943, "end_time": "2021-01-05T16:55:12.436612", "exception": false, "start_time": "2021-01-05T16:55:12.400669", "status": "completed"} tags=[]
INPUT_SUBSET = "z_score_std"

# %% papermill={"duration": 0.035381, "end_time": "2021-01-05T16:55:12.494214", "exception": false, "start_time": "2021-01-05T16:55:12.458833", "status": "completed"} tags=[]
INPUT_STEM = "projection-smultixcan-efo_partial-mashr-zscores"

# %% papermill={"duration": 0.037394, "end_time": "2021-01-05T16:55:12.553391", "exception": false, "start_time": "2021-01-05T16:55:12.515997", "status": "completed"} tags=[]
input_filepath = Path(
    conf.RESULTS["DATA_TRANSFORMATIONS_DIR"],
    INPUT_SUBSET,
    f"{INPUT_SUBSET}-{INPUT_STEM}.pkl",
).resolve()
display(input_filepath)

assert input_filepath.exists(), "Input file does not exist"

input_filepath_stem = input_filepath.stem
display(input_filepath_stem)

# %% [markdown] papermill={"duration": 0.021837, "end_time": "2021-01-05T16:55:12.596858", "exception": false, "start_time": "2021-01-05T16:55:12.575021", "status": "completed"} tags=[]
# ## Clustering

# %% papermill={"duration": 0.04176, "end_time": "2021-01-05T16:55:12.659804", "exception": false, "start_time": "2021-01-05T16:55:12.618044", "status": "completed"} tags=[]
from sklearn.cluster import SpectralClustering

# %% papermill={"duration": 0.036227, "end_time": "2021-01-05T16:55:12.718737", "exception": false, "start_time": "2021-01-05T16:55:12.682510", "status": "completed"} tags=[]
CLUSTERING_ATTRIBUTES_TO_SAVE = ["n_clusters"]

# %% papermill={"duration": 0.037563, "end_time": "2021-01-05T16:55:12.778188", "exception": false, "start_time": "2021-01-05T16:55:12.740625", "status": "completed"} tags=[]
CLUSTERING_OPTIONS = {}

CLUSTERING_OPTIONS["K_MIN"] = 2
CLUSTERING_OPTIONS["K_MAX"] = 60  # sqrt(3749)
CLUSTERING_OPTIONS["N_REPS_PER_K"] = 5
CLUSTERING_OPTIONS["KMEANS_N_INIT"] = 10
CLUSTERING_OPTIONS["N_NEIGHBORS"] = 10
CLUSTERING_OPTIONS["AFFINITY"] = "nearest_neighbors"

display(CLUSTERING_OPTIONS)

# %% papermill={"duration": 0.038434, "end_time": "2021-01-05T16:55:12.839741", "exception": false, "start_time": "2021-01-05T16:55:12.801307", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.037052, "end_time": "2021-01-05T16:55:12.898988", "exception": false, "start_time": "2021-01-05T16:55:12.861936", "status": "completed"} tags=[]
display(len(CLUSTERERS))

# %% papermill={"duration": 0.039329, "end_time": "2021-01-05T16:55:12.961480", "exception": false, "start_time": "2021-01-05T16:55:12.922151", "status": "completed"} tags=[]
_iter = iter(CLUSTERERS.items())
display(next(_iter))
display(next(_iter))

# %% papermill={"duration": 0.037967, "end_time": "2021-01-05T16:55:13.022175", "exception": false, "start_time": "2021-01-05T16:55:12.984208", "status": "completed"} tags=[]
clustering_method_name = method_name
display(clustering_method_name)

# %% [markdown] papermill={"duration": 0.022345, "end_time": "2021-01-05T16:55:13.067444", "exception": false, "start_time": "2021-01-05T16:55:13.045099", "status": "completed"} tags=[]
# ## Output directory

# %% papermill={"duration": 0.038387, "end_time": "2021-01-05T16:55:13.128638", "exception": false, "start_time": "2021-01-05T16:55:13.090251", "status": "completed"} tags=[]
# output dir for this notebook
RESULTS_DIR = Path(
    conf.RESULTS["CLUSTERING_RUNS_DIR"],
    f"{INPUT_SUBSET}-{INPUT_STEM}",
).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %% [markdown] papermill={"duration": 0.022882, "end_time": "2021-01-05T16:55:13.175489", "exception": false, "start_time": "2021-01-05T16:55:13.152607", "status": "completed"} tags=[]
# # Load input file

# %% papermill={"duration": 0.051146, "end_time": "2021-01-05T16:55:13.249823", "exception": false, "start_time": "2021-01-05T16:55:13.198677", "status": "completed"} tags=[]
data = pd.read_pickle(input_filepath)

# %% papermill={"duration": 0.039185, "end_time": "2021-01-05T16:55:13.314468", "exception": false, "start_time": "2021-01-05T16:55:13.275283", "status": "completed"} tags=[]
data.shape

# %% papermill={"duration": 0.050625, "end_time": "2021-01-05T16:55:13.389252", "exception": false, "start_time": "2021-01-05T16:55:13.338627", "status": "completed"} tags=[]
data.head()

# %% papermill={"duration": 0.045392, "end_time": "2021-01-05T16:55:13.458932", "exception": false, "start_time": "2021-01-05T16:55:13.413540", "status": "completed"} tags=[]
assert not data.isna().any().any()

# %% [markdown] papermill={"duration": 0.023345, "end_time": "2021-01-05T16:55:13.508203", "exception": false, "start_time": "2021-01-05T16:55:13.484858", "status": "completed"} tags=[]
# # Clustering

# %% [markdown] papermill={"duration": 0.02297, "end_time": "2021-01-05T16:55:13.554507", "exception": false, "start_time": "2021-01-05T16:55:13.531537", "status": "completed"} tags=[]
# ## Generate ensemble

# %% papermill={"duration": 0.040664, "end_time": "2021-01-05T16:55:13.618311", "exception": false, "start_time": "2021-01-05T16:55:13.577647", "status": "completed"} tags=[]
from clustering.ensemble import generate_ensemble

# %% papermill={"duration": 8335.83344, "end_time": "2021-01-05T19:14:09.475533", "exception": false, "start_time": "2021-01-05T16:55:13.642093", "status": "completed"} tags=[]
ensemble = generate_ensemble(
    data,
    CLUSTERERS,
    attributes=CLUSTERING_ATTRIBUTES_TO_SAVE,
)

# %% papermill={"duration": 0.086615, "end_time": "2021-01-05T19:14:09.633618", "exception": false, "start_time": "2021-01-05T19:14:09.547003", "status": "completed"} tags=[]
# the number should be close to 295 (the number of partitions generated by k-means/spectral clustering)
ensemble.shape

# %% papermill={"duration": 0.093961, "end_time": "2021-01-05T19:14:09.801058", "exception": false, "start_time": "2021-01-05T19:14:09.707097", "status": "completed"} tags=[]
ensemble.head()

# %% papermill={"duration": 0.088294, "end_time": "2021-01-05T19:14:09.961841", "exception": false, "start_time": "2021-01-05T19:14:09.873547", "status": "completed"} tags=[]
ensemble["n_clusters"].value_counts().head()

# %% papermill={"duration": 0.088975, "end_time": "2021-01-05T19:14:10.122937", "exception": false, "start_time": "2021-01-05T19:14:10.033962", "status": "completed"} tags=[]
ensemble_stats = ensemble["n_clusters"].describe()
display(ensemble_stats)

# %% [markdown] papermill={"duration": 0.071406, "end_time": "2021-01-05T19:14:10.267941", "exception": false, "start_time": "2021-01-05T19:14:10.196535", "status": "completed"} tags=[]
# ## Testing

# %% papermill={"duration": 0.086013, "end_time": "2021-01-05T19:14:10.426033", "exception": false, "start_time": "2021-01-05T19:14:10.340020", "status": "completed"} tags=[]
assert ensemble_stats["min"] > 1

# %% papermill={"duration": 0.086859, "end_time": "2021-01-05T19:14:10.584227", "exception": false, "start_time": "2021-01-05T19:14:10.497368", "status": "completed"} tags=[]
assert not ensemble["n_clusters"].isna().any()

# %% papermill={"duration": 0.092444, "end_time": "2021-01-05T19:14:10.752245", "exception": false, "start_time": "2021-01-05T19:14:10.659801", "status": "completed"} tags=[]
assert ensemble.shape[0] == len(CLUSTERERS)

# %% papermill={"duration": 0.103779, "end_time": "2021-01-05T19:14:10.961522", "exception": false, "start_time": "2021-01-05T19:14:10.857743", "status": "completed"} tags=[]
# all partitions have the right size
assert np.all(
    [part["partition"].shape[0] == data.shape[0] for idx, part in ensemble.iterrows()]
)

# %% papermill={"duration": 0.108258, "end_time": "2021-01-05T19:14:11.143453", "exception": false, "start_time": "2021-01-05T19:14:11.035195", "status": "completed"} tags=[]
# no partition has negative clusters (noisy points)
assert not np.any([(part["partition"] < 0).any() for idx, part in ensemble.iterrows()])

# %% [markdown] papermill={"duration": 0.07194, "end_time": "2021-01-05T19:14:11.287465", "exception": false, "start_time": "2021-01-05T19:14:11.215525", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.088629, "end_time": "2021-01-05T19:14:11.448261", "exception": false, "start_time": "2021-01-05T19:14:11.359632", "status": "completed"} tags=[]
output_filename = Path(
    RESULTS_DIR,
    generate_result_set_name(
        CLUSTERING_OPTIONS,
        prefix=f"{clustering_method_name}-",
        suffix=".pkl",
    ),
).resolve()
display(output_filename)

# %% papermill={"duration": 0.098355, "end_time": "2021-01-05T19:14:11.620384", "exception": false, "start_time": "2021-01-05T19:14:11.522029", "status": "completed"} tags=[]
ensemble.to_pickle(output_filename)

# %% [markdown] papermill={"duration": 0.071427, "end_time": "2021-01-05T19:14:11.766622", "exception": false, "start_time": "2021-01-05T19:14:11.695195", "status": "completed"} tags=[]
# # Stability

# %% [markdown] papermill={"duration": 0.071367, "end_time": "2021-01-05T19:14:11.909484", "exception": false, "start_time": "2021-01-05T19:14:11.838117", "status": "completed"} tags=[]
# ## Group ensemble by n_clusters

# %% papermill={"duration": 0.100933, "end_time": "2021-01-05T19:14:12.082010", "exception": false, "start_time": "2021-01-05T19:14:11.981077", "status": "completed"} tags=[]
parts = ensemble.groupby("n_clusters").apply(
    lambda x: np.concatenate(x["partition"].apply(lambda x: x.reshape(1, -1)), axis=0)
)

# %% papermill={"duration": 0.098129, "end_time": "2021-01-05T19:14:12.252501", "exception": false, "start_time": "2021-01-05T19:14:12.154372", "status": "completed"} tags=[]
parts.head()

# %% papermill={"duration": 0.087658, "end_time": "2021-01-05T19:14:12.413670", "exception": false, "start_time": "2021-01-05T19:14:12.326012", "status": "completed"} tags=[]
assert np.all(
    [
        parts.loc[k].shape == (CLUSTERING_OPTIONS["N_REPS_PER_K"], data.shape[0])
        for k in parts.index
    ]
)

# %% [markdown] papermill={"duration": 0.072146, "end_time": "2021-01-05T19:14:12.557712", "exception": false, "start_time": "2021-01-05T19:14:12.485566", "status": "completed"} tags=[]
# ## Compute stability

# %% papermill={"duration": 0.08936, "end_time": "2021-01-05T19:14:12.718892", "exception": false, "start_time": "2021-01-05T19:14:12.629532", "status": "completed"} tags=[]
from sklearn.metrics import adjusted_rand_score as ari
from scipy.spatial.distance import squareform, pdist

# %% papermill={"duration": 0.597984, "end_time": "2021-01-05T19:14:13.389335", "exception": false, "start_time": "2021-01-05T19:14:12.791351", "status": "completed"} tags=[]
parts_ari = pd.Series(
    {k: pdist(parts.loc[k], metric=ari) for k in parts.index}, name="k"
)

# %% papermill={"duration": 0.088749, "end_time": "2021-01-05T19:14:13.550226", "exception": false, "start_time": "2021-01-05T19:14:13.461477", "status": "completed"} tags=[]
parts_ari_stability = parts_ari.apply(lambda x: x.mean())
display(parts_ari_stability.sort_values(ascending=False).head(15))

# %% papermill={"duration": 0.09064, "end_time": "2021-01-05T19:14:13.713451", "exception": false, "start_time": "2021-01-05T19:14:13.622811", "status": "completed"} tags=[]
parts_ari_df = pd.DataFrame.from_records(parts_ari.tolist()).set_index(
    parts_ari.index.copy()
)

# %% papermill={"duration": 0.086638, "end_time": "2021-01-05T19:14:13.872130", "exception": false, "start_time": "2021-01-05T19:14:13.785492", "status": "completed"} tags=[]
parts_ari_df.shape

# %% papermill={"duration": 0.087657, "end_time": "2021-01-05T19:14:14.033358", "exception": false, "start_time": "2021-01-05T19:14:13.945701", "status": "completed"} tags=[]
assert (
    int(
        (CLUSTERING_OPTIONS["N_REPS_PER_K"] * (CLUSTERING_OPTIONS["N_REPS_PER_K"] - 1))
        / 2
    )
    == parts_ari_df.shape[1]
)

# %% papermill={"duration": 0.093626, "end_time": "2021-01-05T19:14:14.199639", "exception": false, "start_time": "2021-01-05T19:14:14.106013", "status": "completed"} tags=[]
parts_ari_df.head()

# %% [markdown] papermill={"duration": 0.073881, "end_time": "2021-01-05T19:14:14.346764", "exception": false, "start_time": "2021-01-05T19:14:14.272883", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.090043, "end_time": "2021-01-05T19:14:14.509418", "exception": false, "start_time": "2021-01-05T19:14:14.419375", "status": "completed"} tags=[]
output_filename = Path(
    RESULTS_DIR,
    generate_result_set_name(
        CLUSTERING_OPTIONS,
        prefix=f"{clustering_method_name}-stability-",
        suffix=".pkl",
    ),
).resolve()
display(output_filename)

# %% papermill={"duration": 0.089402, "end_time": "2021-01-05T19:14:14.672521", "exception": false, "start_time": "2021-01-05T19:14:14.583119", "status": "completed"} tags=[]
parts_ari_df.to_pickle(output_filename)

# %% [markdown] papermill={"duration": 0.072846, "end_time": "2021-01-05T19:14:14.820865", "exception": false, "start_time": "2021-01-05T19:14:14.748019", "status": "completed"} tags=[]
# ## Stability plot

# %% papermill={"duration": 0.089984, "end_time": "2021-01-05T19:14:14.983329", "exception": false, "start_time": "2021-01-05T19:14:14.893345", "status": "completed"} tags=[]
parts_ari_df_plot = (
    parts_ari_df.stack()
    .reset_index()
    .rename(columns={"level_0": "k", "level_1": "idx", 0: "ari"})
)

# %% papermill={"duration": 0.089743, "end_time": "2021-01-05T19:14:15.146788", "exception": false, "start_time": "2021-01-05T19:14:15.057045", "status": "completed"} tags=[]
parts_ari_df_plot.dtypes

# %% papermill={"duration": 0.090912, "end_time": "2021-01-05T19:14:15.311805", "exception": false, "start_time": "2021-01-05T19:14:15.220893", "status": "completed"} tags=[]
parts_ari_df_plot.head()

# %% papermill={"duration": 3.053521, "end_time": "2021-01-05T19:14:18.439965", "exception": false, "start_time": "2021-01-05T19:14:15.386444", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.074869, "end_time": "2021-01-05T19:14:18.589323", "exception": false, "start_time": "2021-01-05T19:14:18.514454", "status": "completed"} tags=[]
