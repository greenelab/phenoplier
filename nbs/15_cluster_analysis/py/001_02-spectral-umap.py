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

# %% [markdown] papermill={"duration": 0.024902, "end_time": "2021-01-05T19:46:17.826288", "exception": false, "start_time": "2021-01-05T19:46:17.801386", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.020249, "end_time": "2021-01-05T19:46:17.867069", "exception": false, "start_time": "2021-01-05T19:46:17.846820", "status": "completed"} tags=[]
# Runs spectral clustering on the umap version of the data.

# %% [markdown] papermill={"duration": 0.020533, "end_time": "2021-01-05T19:46:17.908163", "exception": false, "start_time": "2021-01-05T19:46:17.887630", "status": "completed"} tags=[]
# # Environment variables

# %% papermill={"duration": 0.034543, "end_time": "2021-01-05T19:46:17.963228", "exception": false, "start_time": "2021-01-05T19:46:17.928685", "status": "completed"} tags=[]
from IPython.display import display

import conf

N_JOBS = conf.GENERAL["N_JOBS"]
display(N_JOBS)

# %% papermill={"duration": 0.027969, "end_time": "2021-01-05T19:46:18.012415", "exception": false, "start_time": "2021-01-05T19:46:17.984446", "status": "completed"} tags=[]
# %env MKL_NUM_THREADS=$N_JOBS
# %env OPEN_BLAS_NUM_THREADS=$N_JOBS
# %env NUMEXPR_NUM_THREADS=$N_JOBS
# %env OMP_NUM_THREADS=$N_JOBS

# %% [markdown] papermill={"duration": 0.021229, "end_time": "2021-01-05T19:46:18.055523", "exception": false, "start_time": "2021-01-05T19:46:18.034294", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.03189, "end_time": "2021-01-05T19:46:18.108466", "exception": false, "start_time": "2021-01-05T19:46:18.076576", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 1.611033, "end_time": "2021-01-05T19:46:19.740668", "exception": false, "start_time": "2021-01-05T19:46:18.129635", "status": "completed"} tags=[]
from pathlib import Path

import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns

from utils import generate_result_set_name

# %% [markdown] papermill={"duration": 0.020946, "end_time": "2021-01-05T19:46:19.785046", "exception": false, "start_time": "2021-01-05T19:46:19.764100", "status": "completed"} tags=[]
# # Settings

# %% papermill={"duration": 0.035331, "end_time": "2021-01-05T19:46:19.841242", "exception": false, "start_time": "2021-01-05T19:46:19.805911", "status": "completed"} tags=[]
INITIAL_RANDOM_STATE = 50000

# %% [markdown] papermill={"duration": 0.02153, "end_time": "2021-01-05T19:46:19.884884", "exception": false, "start_time": "2021-01-05T19:46:19.863354", "status": "completed"} tags=[]
# ## Input data

# %% papermill={"duration": 0.035121, "end_time": "2021-01-05T19:46:19.940907", "exception": false, "start_time": "2021-01-05T19:46:19.905786", "status": "completed"} tags=[]
INPUT_SUBSET = "umap"

# %% papermill={"duration": 0.035148, "end_time": "2021-01-05T19:46:19.997441", "exception": false, "start_time": "2021-01-05T19:46:19.962293", "status": "completed"} tags=[]
INPUT_STEM = "z_score_std-projection-smultixcan-efo_partial-mashr-zscores"

# %% papermill={"duration": 0.035692, "end_time": "2021-01-05T19:46:20.054980", "exception": false, "start_time": "2021-01-05T19:46:20.019288", "status": "completed"} tags=[]
DR_OPTIONS = {
    "n_components": 50,
    "metric": "euclidean",
    "n_neighbors": 15,
    "random_state": 0,
}

# %% papermill={"duration": 0.037473, "end_time": "2021-01-05T19:46:20.114576", "exception": false, "start_time": "2021-01-05T19:46:20.077103", "status": "completed"} tags=[]
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

# %% [markdown] papermill={"duration": 0.021628, "end_time": "2021-01-05T19:46:20.158745", "exception": false, "start_time": "2021-01-05T19:46:20.137117", "status": "completed"} tags=[]
# ## Clustering

# %% papermill={"duration": 0.041937, "end_time": "2021-01-05T19:46:20.222208", "exception": false, "start_time": "2021-01-05T19:46:20.180271", "status": "completed"} tags=[]
from sklearn.cluster import SpectralClustering

# %% papermill={"duration": 0.035761, "end_time": "2021-01-05T19:46:20.280373", "exception": false, "start_time": "2021-01-05T19:46:20.244612", "status": "completed"} tags=[]
CLUSTERING_ATTRIBUTES_TO_SAVE = ["n_clusters"]

# %% papermill={"duration": 0.036975, "end_time": "2021-01-05T19:46:20.338872", "exception": false, "start_time": "2021-01-05T19:46:20.301897", "status": "completed"} tags=[]
CLUSTERING_OPTIONS = {}

CLUSTERING_OPTIONS["K_MIN"] = 2
CLUSTERING_OPTIONS["K_MAX"] = 60  # sqrt(3749)
CLUSTERING_OPTIONS["N_REPS_PER_K"] = 5
CLUSTERING_OPTIONS["KMEANS_N_INIT"] = 10
CLUSTERING_OPTIONS["N_NEIGHBORS"] = None
CLUSTERING_OPTIONS["AFFINITY"] = "rbf"  # nearest neighbors does not work well with umap

display(CLUSTERING_OPTIONS)

# %% papermill={"duration": 0.040114, "end_time": "2021-01-05T19:46:20.402942", "exception": false, "start_time": "2021-01-05T19:46:20.362828", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.036664, "end_time": "2021-01-05T19:46:20.462266", "exception": false, "start_time": "2021-01-05T19:46:20.425602", "status": "completed"} tags=[]
display(len(CLUSTERERS))

# %% papermill={"duration": 0.038756, "end_time": "2021-01-05T19:46:20.523907", "exception": false, "start_time": "2021-01-05T19:46:20.485151", "status": "completed"} tags=[]
_iter = iter(CLUSTERERS.items())
display(next(_iter))
display(next(_iter))

# %% papermill={"duration": 0.03707, "end_time": "2021-01-05T19:46:20.584401", "exception": false, "start_time": "2021-01-05T19:46:20.547331", "status": "completed"} tags=[]
clustering_method_name = method_name
display(clustering_method_name)

# %% [markdown] papermill={"duration": 0.022577, "end_time": "2021-01-05T19:46:20.630603", "exception": false, "start_time": "2021-01-05T19:46:20.608026", "status": "completed"} tags=[]
# ## Output directory

# %% papermill={"duration": 0.037921, "end_time": "2021-01-05T19:46:20.691454", "exception": false, "start_time": "2021-01-05T19:46:20.653533", "status": "completed"} tags=[]
# output dir for this notebook
RESULTS_DIR = Path(
    conf.RESULTS["CLUSTERING_RUNS_DIR"],
    f"{INPUT_SUBSET}-{INPUT_STEM}",
).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %% [markdown] papermill={"duration": 0.022766, "end_time": "2021-01-05T19:46:20.737420", "exception": false, "start_time": "2021-01-05T19:46:20.714654", "status": "completed"} tags=[]
# # Load input file

# %% papermill={"duration": 0.038441, "end_time": "2021-01-05T19:46:20.798534", "exception": false, "start_time": "2021-01-05T19:46:20.760093", "status": "completed"} tags=[]
data = pd.read_pickle(input_filepath)

# %% papermill={"duration": 0.038568, "end_time": "2021-01-05T19:46:20.860518", "exception": false, "start_time": "2021-01-05T19:46:20.821950", "status": "completed"} tags=[]
data.shape

# %% papermill={"duration": 0.05042, "end_time": "2021-01-05T19:46:20.934600", "exception": false, "start_time": "2021-01-05T19:46:20.884180", "status": "completed"} tags=[]
data.head()

# %% papermill={"duration": 0.038129, "end_time": "2021-01-05T19:46:20.997011", "exception": false, "start_time": "2021-01-05T19:46:20.958882", "status": "completed"} tags=[]
assert not data.isna().any().any()

# %% [markdown] papermill={"duration": 0.024772, "end_time": "2021-01-05T19:46:21.046231", "exception": false, "start_time": "2021-01-05T19:46:21.021459", "status": "completed"} tags=[]
# # Clustering

# %% [markdown] papermill={"duration": 0.023259, "end_time": "2021-01-05T19:46:21.092873", "exception": false, "start_time": "2021-01-05T19:46:21.069614", "status": "completed"} tags=[]
# ## Generate ensemble

# %% papermill={"duration": 0.040046, "end_time": "2021-01-05T19:46:21.156024", "exception": false, "start_time": "2021-01-05T19:46:21.115978", "status": "completed"} tags=[]
from clustering.ensemble import generate_ensemble

# %% papermill={"duration": 788.824459, "end_time": "2021-01-05T19:59:30.004092", "exception": false, "start_time": "2021-01-05T19:46:21.179633", "status": "completed"} tags=[]
ensemble = generate_ensemble(
    data,
    CLUSTERERS,
    attributes=CLUSTERING_ATTRIBUTES_TO_SAVE,
)

# %% papermill={"duration": 0.084539, "end_time": "2021-01-05T19:59:30.157904", "exception": false, "start_time": "2021-01-05T19:59:30.073365", "status": "completed"} tags=[]
# the number should be close to 295 (the number of partitions generated by k-means/spectral clustering)
ensemble.shape

# %% papermill={"duration": 0.092329, "end_time": "2021-01-05T19:59:30.321189", "exception": false, "start_time": "2021-01-05T19:59:30.228860", "status": "completed"} tags=[]
ensemble.head()

# %% papermill={"duration": 0.086728, "end_time": "2021-01-05T19:59:30.479633", "exception": false, "start_time": "2021-01-05T19:59:30.392905", "status": "completed"} tags=[]
ensemble["n_clusters"].value_counts().head()

# %% papermill={"duration": 0.086712, "end_time": "2021-01-05T19:59:30.639490", "exception": false, "start_time": "2021-01-05T19:59:30.552778", "status": "completed"} tags=[]
ensemble_stats = ensemble["n_clusters"].describe()
display(ensemble_stats)

# %% [markdown] papermill={"duration": 0.070312, "end_time": "2021-01-05T19:59:30.781832", "exception": false, "start_time": "2021-01-05T19:59:30.711520", "status": "completed"} tags=[]
# ## Testing

# %% papermill={"duration": 0.084486, "end_time": "2021-01-05T19:59:30.937366", "exception": false, "start_time": "2021-01-05T19:59:30.852880", "status": "completed"} tags=[]
assert ensemble_stats["min"] > 1

# %% papermill={"duration": 0.084521, "end_time": "2021-01-05T19:59:31.092311", "exception": false, "start_time": "2021-01-05T19:59:31.007790", "status": "completed"} tags=[]
assert not ensemble["n_clusters"].isna().any()

# %% papermill={"duration": 0.085013, "end_time": "2021-01-05T19:59:31.247949", "exception": false, "start_time": "2021-01-05T19:59:31.162936", "status": "completed"} tags=[]
assert ensemble.shape[0] == len(CLUSTERERS)

# %% papermill={"duration": 0.102575, "end_time": "2021-01-05T19:59:31.421706", "exception": false, "start_time": "2021-01-05T19:59:31.319131", "status": "completed"} tags=[]
# all partitions have the right size
assert np.all(
    [part["partition"].shape[0] == data.shape[0] for idx, part in ensemble.iterrows()]
)

# %% papermill={"duration": 0.107798, "end_time": "2021-01-05T19:59:31.601275", "exception": false, "start_time": "2021-01-05T19:59:31.493477", "status": "completed"} tags=[]
# no partition has negative clusters (noisy points)
assert not np.any([(part["partition"] < 0).any() for idx, part in ensemble.iterrows()])

# %% [markdown] papermill={"duration": 0.071079, "end_time": "2021-01-05T19:59:31.742747", "exception": false, "start_time": "2021-01-05T19:59:31.671668", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.087113, "end_time": "2021-01-05T19:59:31.900567", "exception": false, "start_time": "2021-01-05T19:59:31.813454", "status": "completed"} tags=[]
output_filename = Path(
    RESULTS_DIR,
    generate_result_set_name(
        CLUSTERING_OPTIONS,
        prefix=f"{clustering_method_name}-",
        suffix=".pkl",
    ),
).resolve()
display(output_filename)

# %% papermill={"duration": 0.092706, "end_time": "2021-01-05T19:59:32.065073", "exception": false, "start_time": "2021-01-05T19:59:31.972367", "status": "completed"} tags=[]
ensemble.to_pickle(output_filename)

# %% [markdown] papermill={"duration": 0.070873, "end_time": "2021-01-05T19:59:32.209987", "exception": false, "start_time": "2021-01-05T19:59:32.139114", "status": "completed"} tags=[]
# # Stability

# %% [markdown] papermill={"duration": 0.070484, "end_time": "2021-01-05T19:59:32.351154", "exception": false, "start_time": "2021-01-05T19:59:32.280670", "status": "completed"} tags=[]
# ## Group ensemble by n_clusters

# %% papermill={"duration": 0.100434, "end_time": "2021-01-05T19:59:32.522690", "exception": false, "start_time": "2021-01-05T19:59:32.422256", "status": "completed"} tags=[]
parts = ensemble.groupby("n_clusters").apply(
    lambda x: np.concatenate(x["partition"].apply(lambda x: x.reshape(1, -1)), axis=0)
)

# %% papermill={"duration": 0.097215, "end_time": "2021-01-05T19:59:32.692141", "exception": false, "start_time": "2021-01-05T19:59:32.594926", "status": "completed"} tags=[]
parts.head()

# %% papermill={"duration": 0.085657, "end_time": "2021-01-05T19:59:32.849711", "exception": false, "start_time": "2021-01-05T19:59:32.764054", "status": "completed"} tags=[]
assert np.all(
    [
        parts.loc[k].shape == (CLUSTERING_OPTIONS["N_REPS_PER_K"], data.shape[0])
        for k in parts.index
    ]
)

# %% [markdown] papermill={"duration": 0.071568, "end_time": "2021-01-05T19:59:32.992555", "exception": false, "start_time": "2021-01-05T19:59:32.920987", "status": "completed"} tags=[]
# ## Compute stability

# %% papermill={"duration": 0.086213, "end_time": "2021-01-05T19:59:33.149687", "exception": false, "start_time": "2021-01-05T19:59:33.063474", "status": "completed"} tags=[]
from sklearn.metrics import adjusted_rand_score as ari
from scipy.spatial.distance import squareform, pdist

# %% papermill={"duration": 0.676318, "end_time": "2021-01-05T19:59:33.897266", "exception": false, "start_time": "2021-01-05T19:59:33.220948", "status": "completed"} tags=[]
parts_ari = pd.Series(
    {k: pdist(parts.loc[k], metric=ari) for k in parts.index}, name="k"
)

# %% papermill={"duration": 0.087254, "end_time": "2021-01-05T19:59:34.055451", "exception": false, "start_time": "2021-01-05T19:59:33.968197", "status": "completed"} tags=[]
parts_ari_stability = parts_ari.apply(lambda x: x.mean())
display(parts_ari_stability.sort_values(ascending=False).head(15))

# %% papermill={"duration": 0.087548, "end_time": "2021-01-05T19:59:34.215061", "exception": false, "start_time": "2021-01-05T19:59:34.127513", "status": "completed"} tags=[]
parts_ari_df = pd.DataFrame.from_records(parts_ari.tolist()).set_index(
    parts_ari.index.copy()
)

# %% papermill={"duration": 0.085878, "end_time": "2021-01-05T19:59:34.372367", "exception": false, "start_time": "2021-01-05T19:59:34.286489", "status": "completed"} tags=[]
parts_ari_df.shape

# %% papermill={"duration": 0.08696, "end_time": "2021-01-05T19:59:34.531957", "exception": false, "start_time": "2021-01-05T19:59:34.444997", "status": "completed"} tags=[]
assert (
    int(
        (CLUSTERING_OPTIONS["N_REPS_PER_K"] * (CLUSTERING_OPTIONS["N_REPS_PER_K"] - 1))
        / 2
    )
    == parts_ari_df.shape[1]
)

# %% papermill={"duration": 0.091562, "end_time": "2021-01-05T19:59:34.696641", "exception": false, "start_time": "2021-01-05T19:59:34.605079", "status": "completed"} tags=[]
parts_ari_df.head()

# %% [markdown] papermill={"duration": 0.071406, "end_time": "2021-01-05T19:59:34.844904", "exception": false, "start_time": "2021-01-05T19:59:34.773498", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.086779, "end_time": "2021-01-05T19:59:35.002898", "exception": false, "start_time": "2021-01-05T19:59:34.916119", "status": "completed"} tags=[]
output_filename = Path(
    RESULTS_DIR,
    generate_result_set_name(
        CLUSTERING_OPTIONS,
        prefix=f"{clustering_method_name}-stability-",
        suffix=".pkl",
    ),
).resolve()
display(output_filename)

# %% papermill={"duration": 0.086579, "end_time": "2021-01-05T19:59:35.161727", "exception": false, "start_time": "2021-01-05T19:59:35.075148", "status": "completed"} tags=[]
parts_ari_df.to_pickle(output_filename)

# %% [markdown] papermill={"duration": 0.071567, "end_time": "2021-01-05T19:59:35.306138", "exception": false, "start_time": "2021-01-05T19:59:35.234571", "status": "completed"} tags=[]
# ## Stability plot

# %% papermill={"duration": 0.088931, "end_time": "2021-01-05T19:59:35.467117", "exception": false, "start_time": "2021-01-05T19:59:35.378186", "status": "completed"} tags=[]
parts_ari_df_plot = (
    parts_ari_df.stack()
    .reset_index()
    .rename(columns={"level_0": "k", "level_1": "idx", 0: "ari"})
)

# %% papermill={"duration": 0.089725, "end_time": "2021-01-05T19:59:35.631010", "exception": false, "start_time": "2021-01-05T19:59:35.541285", "status": "completed"} tags=[]
parts_ari_df_plot.dtypes

# %% papermill={"duration": 0.0896, "end_time": "2021-01-05T19:59:35.793753", "exception": false, "start_time": "2021-01-05T19:59:35.704153", "status": "completed"} tags=[]
parts_ari_df_plot.head()

# %% papermill={"duration": 2.953401, "end_time": "2021-01-05T19:59:38.821200", "exception": false, "start_time": "2021-01-05T19:59:35.867799", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.074501, "end_time": "2021-01-05T19:59:38.970344", "exception": false, "start_time": "2021-01-05T19:59:38.895843", "status": "completed"} tags=[]
