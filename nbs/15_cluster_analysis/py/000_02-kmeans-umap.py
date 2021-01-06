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

# %% [markdown] papermill={"duration": 0.054831, "end_time": "2021-01-05T16:48:39.949331", "exception": false, "start_time": "2021-01-05T16:48:39.894500", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.021079, "end_time": "2021-01-05T16:48:39.991757", "exception": false, "start_time": "2021-01-05T16:48:39.970678", "status": "completed"} tags=[]
# Runs k-means on the umap version of the data.

# %% [markdown] papermill={"duration": 0.020877, "end_time": "2021-01-05T16:48:40.033458", "exception": false, "start_time": "2021-01-05T16:48:40.012581", "status": "completed"} tags=[]
# # Environment variables

# %% papermill={"duration": 0.035005, "end_time": "2021-01-05T16:48:40.089465", "exception": false, "start_time": "2021-01-05T16:48:40.054460", "status": "completed"} tags=[]
from IPython.display import display

import conf

N_JOBS = conf.GENERAL["N_JOBS"]
display(N_JOBS)

# %% papermill={"duration": 0.028803, "end_time": "2021-01-05T16:48:40.140939", "exception": false, "start_time": "2021-01-05T16:48:40.112136", "status": "completed"} tags=[]
# %env MKL_NUM_THREADS=$N_JOBS
# %env OPEN_BLAS_NUM_THREADS=$N_JOBS
# %env NUMEXPR_NUM_THREADS=$N_JOBS
# %env OMP_NUM_THREADS=$N_JOBS

# %% [markdown] papermill={"duration": 0.021951, "end_time": "2021-01-05T16:48:40.185522", "exception": false, "start_time": "2021-01-05T16:48:40.163571", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.032724, "end_time": "2021-01-05T16:48:40.239503", "exception": false, "start_time": "2021-01-05T16:48:40.206779", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 1.599849, "end_time": "2021-01-05T16:48:41.862035", "exception": false, "start_time": "2021-01-05T16:48:40.262186", "status": "completed"} tags=[]
from pathlib import Path

import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns

from utils import generate_result_set_name

# %% [markdown] papermill={"duration": 0.022021, "end_time": "2021-01-05T16:48:41.909449", "exception": false, "start_time": "2021-01-05T16:48:41.887428", "status": "completed"} tags=[]
# # Settings

# %% papermill={"duration": 0.035517, "end_time": "2021-01-05T16:48:41.966517", "exception": false, "start_time": "2021-01-05T16:48:41.931000", "status": "completed"} tags=[]
INITIAL_RANDOM_STATE = 20000

# %% [markdown] papermill={"duration": 0.021068, "end_time": "2021-01-05T16:48:42.009479", "exception": false, "start_time": "2021-01-05T16:48:41.988411", "status": "completed"} tags=[]
# ## Input data

# %% papermill={"duration": 0.035605, "end_time": "2021-01-05T16:48:42.066289", "exception": false, "start_time": "2021-01-05T16:48:42.030684", "status": "completed"} tags=[]
INPUT_SUBSET = "umap"

# %% papermill={"duration": 0.036344, "end_time": "2021-01-05T16:48:42.125029", "exception": false, "start_time": "2021-01-05T16:48:42.088685", "status": "completed"} tags=[]
INPUT_STEM = "z_score_std-projection-smultixcan-efo_partial-mashr-zscores"

# %% papermill={"duration": 0.036001, "end_time": "2021-01-05T16:48:42.182924", "exception": false, "start_time": "2021-01-05T16:48:42.146923", "status": "completed"} tags=[]
DR_OPTIONS = {
    "n_components": 50,
    "metric": "euclidean",
    "n_neighbors": 15,
    "random_state": 0,
}

# %% papermill={"duration": 0.038139, "end_time": "2021-01-05T16:48:42.243540", "exception": false, "start_time": "2021-01-05T16:48:42.205401", "status": "completed"} tags=[]
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

# %% [markdown] papermill={"duration": 0.021882, "end_time": "2021-01-05T16:48:42.287962", "exception": false, "start_time": "2021-01-05T16:48:42.266080", "status": "completed"} tags=[]
# ## Clustering

# %% papermill={"duration": 0.04201, "end_time": "2021-01-05T16:48:42.352082", "exception": false, "start_time": "2021-01-05T16:48:42.310072", "status": "completed"} tags=[]
from sklearn.cluster import KMeans

# %% papermill={"duration": 0.036334, "end_time": "2021-01-05T16:48:42.411369", "exception": false, "start_time": "2021-01-05T16:48:42.375035", "status": "completed"} tags=[]
CLUSTERING_ATTRIBUTES_TO_SAVE = ["n_clusters"]

# %% papermill={"duration": 0.037325, "end_time": "2021-01-05T16:48:42.471168", "exception": false, "start_time": "2021-01-05T16:48:42.433843", "status": "completed"} tags=[]
CLUSTERING_OPTIONS = {}

CLUSTERING_OPTIONS["K_MIN"] = 2
CLUSTERING_OPTIONS["K_MAX"] = 60  # sqrt(3749)
CLUSTERING_OPTIONS["N_REPS_PER_K"] = 5
CLUSTERING_OPTIONS["KMEANS_N_INIT"] = 10

display(CLUSTERING_OPTIONS)

# %% papermill={"duration": 0.038136, "end_time": "2021-01-05T16:48:42.531862", "exception": false, "start_time": "2021-01-05T16:48:42.493726", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.037154, "end_time": "2021-01-05T16:48:42.591266", "exception": false, "start_time": "2021-01-05T16:48:42.554112", "status": "completed"} tags=[]
display(len(CLUSTERERS))

# %% papermill={"duration": 0.038882, "end_time": "2021-01-05T16:48:42.653357", "exception": false, "start_time": "2021-01-05T16:48:42.614475", "status": "completed"} tags=[]
_iter = iter(CLUSTERERS.items())
display(next(_iter))
display(next(_iter))

# %% papermill={"duration": 0.037721, "end_time": "2021-01-05T16:48:42.715012", "exception": false, "start_time": "2021-01-05T16:48:42.677291", "status": "completed"} tags=[]
clustering_method_name = method_name
display(clustering_method_name)

# %% [markdown] papermill={"duration": 0.023111, "end_time": "2021-01-05T16:48:42.761863", "exception": false, "start_time": "2021-01-05T16:48:42.738752", "status": "completed"} tags=[]
# ## Output directory

# %% papermill={"duration": 0.038615, "end_time": "2021-01-05T16:48:42.823842", "exception": false, "start_time": "2021-01-05T16:48:42.785227", "status": "completed"} tags=[]
# output dir for this notebook
RESULTS_DIR = Path(
    conf.RESULTS["CLUSTERING_RUNS_DIR"],
    f"{INPUT_SUBSET}-{INPUT_STEM}",
).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %% [markdown] papermill={"duration": 0.025006, "end_time": "2021-01-05T16:48:42.872456", "exception": false, "start_time": "2021-01-05T16:48:42.847450", "status": "completed"} tags=[]
# # Load input file

# %% papermill={"duration": 0.039234, "end_time": "2021-01-05T16:48:42.936023", "exception": false, "start_time": "2021-01-05T16:48:42.896789", "status": "completed"} tags=[]
data = pd.read_pickle(input_filepath)

# %% papermill={"duration": 0.038771, "end_time": "2021-01-05T16:48:42.999447", "exception": false, "start_time": "2021-01-05T16:48:42.960676", "status": "completed"} tags=[]
data.shape

# %% papermill={"duration": 0.051497, "end_time": "2021-01-05T16:48:43.074820", "exception": false, "start_time": "2021-01-05T16:48:43.023323", "status": "completed"} tags=[]
data.head()

# %% papermill={"duration": 0.038951, "end_time": "2021-01-05T16:48:43.137935", "exception": false, "start_time": "2021-01-05T16:48:43.098984", "status": "completed"} tags=[]
assert not data.isna().any().any()

# %% [markdown] papermill={"duration": 0.023855, "end_time": "2021-01-05T16:48:43.186896", "exception": false, "start_time": "2021-01-05T16:48:43.163041", "status": "completed"} tags=[]
# # Clustering

# %% [markdown] papermill={"duration": 0.023668, "end_time": "2021-01-05T16:48:43.234705", "exception": false, "start_time": "2021-01-05T16:48:43.211037", "status": "completed"} tags=[]
# ## Generate ensemble

# %% papermill={"duration": 0.040925, "end_time": "2021-01-05T16:48:43.299318", "exception": false, "start_time": "2021-01-05T16:48:43.258393", "status": "completed"} tags=[]
from clustering.ensemble import generate_ensemble

# %% papermill={"duration": 374.521416, "end_time": "2021-01-05T16:54:57.845276", "exception": false, "start_time": "2021-01-05T16:48:43.323860", "status": "completed"} tags=[]
ensemble = generate_ensemble(
    data,
    CLUSTERERS,
    attributes=CLUSTERING_ATTRIBUTES_TO_SAVE,
)

# %% papermill={"duration": 0.085225, "end_time": "2021-01-05T16:54:58.000861", "exception": false, "start_time": "2021-01-05T16:54:57.915636", "status": "completed"} tags=[]
# the number should be close to 295 (the number of partitions generated by k-means/spectral clustering)
ensemble.shape

# %% papermill={"duration": 0.092062, "end_time": "2021-01-05T16:54:58.164832", "exception": false, "start_time": "2021-01-05T16:54:58.072770", "status": "completed"} tags=[]
ensemble.head()

# %% papermill={"duration": 0.087414, "end_time": "2021-01-05T16:54:58.324417", "exception": false, "start_time": "2021-01-05T16:54:58.237003", "status": "completed"} tags=[]
ensemble["n_clusters"].value_counts().head()

# %% papermill={"duration": 0.087704, "end_time": "2021-01-05T16:54:58.487515", "exception": false, "start_time": "2021-01-05T16:54:58.399811", "status": "completed"} tags=[]
ensemble_stats = ensemble["n_clusters"].describe()
display(ensemble_stats)

# %% [markdown] papermill={"duration": 0.069858, "end_time": "2021-01-05T16:54:58.629304", "exception": false, "start_time": "2021-01-05T16:54:58.559446", "status": "completed"} tags=[]
# ## Testing

# %% papermill={"duration": 0.085465, "end_time": "2021-01-05T16:54:58.785028", "exception": false, "start_time": "2021-01-05T16:54:58.699563", "status": "completed"} tags=[]
assert ensemble_stats["min"] > 1

# %% papermill={"duration": 0.084717, "end_time": "2021-01-05T16:54:58.940365", "exception": false, "start_time": "2021-01-05T16:54:58.855648", "status": "completed"} tags=[]
assert not ensemble["n_clusters"].isna().any()

# %% papermill={"duration": 0.084872, "end_time": "2021-01-05T16:54:59.096576", "exception": false, "start_time": "2021-01-05T16:54:59.011704", "status": "completed"} tags=[]
assert ensemble.shape[0] == len(CLUSTERERS)

# %% papermill={"duration": 0.103813, "end_time": "2021-01-05T16:54:59.270892", "exception": false, "start_time": "2021-01-05T16:54:59.167079", "status": "completed"} tags=[]
# all partitions have the right size
assert np.all(
    [part["partition"].shape[0] == data.shape[0] for idx, part in ensemble.iterrows()]
)

# %% papermill={"duration": 0.110394, "end_time": "2021-01-05T16:54:59.451952", "exception": false, "start_time": "2021-01-05T16:54:59.341558", "status": "completed"} tags=[]
# no partition has negative clusters (noisy points)
assert not np.any([(part["partition"] < 0).any() for idx, part in ensemble.iterrows()])

# %% [markdown] papermill={"duration": 0.070086, "end_time": "2021-01-05T16:54:59.593068", "exception": false, "start_time": "2021-01-05T16:54:59.522982", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.087394, "end_time": "2021-01-05T16:54:59.751766", "exception": false, "start_time": "2021-01-05T16:54:59.664372", "status": "completed"} tags=[]
output_filename = Path(
    RESULTS_DIR,
    generate_result_set_name(
        CLUSTERING_OPTIONS,
        prefix=f"{clustering_method_name}-",
        suffix=".pkl",
    ),
).resolve()
display(output_filename)

# %% papermill={"duration": 0.098363, "end_time": "2021-01-05T16:54:59.921908", "exception": false, "start_time": "2021-01-05T16:54:59.823545", "status": "completed"} tags=[]
ensemble.to_pickle(output_filename)

# %% [markdown] papermill={"duration": 0.070536, "end_time": "2021-01-05T16:55:00.066634", "exception": false, "start_time": "2021-01-05T16:54:59.996098", "status": "completed"} tags=[]
# # Stability

# %% [markdown] papermill={"duration": 0.071332, "end_time": "2021-01-05T16:55:00.209074", "exception": false, "start_time": "2021-01-05T16:55:00.137742", "status": "completed"} tags=[]
# ## Group ensemble by n_clusters

# %% papermill={"duration": 0.102165, "end_time": "2021-01-05T16:55:00.382691", "exception": false, "start_time": "2021-01-05T16:55:00.280526", "status": "completed"} tags=[]
parts = ensemble.groupby("n_clusters").apply(
    lambda x: np.concatenate(x["partition"].apply(lambda x: x.reshape(1, -1)), axis=0)
)

# %% papermill={"duration": 0.096543, "end_time": "2021-01-05T16:55:00.553437", "exception": false, "start_time": "2021-01-05T16:55:00.456894", "status": "completed"} tags=[]
parts.head()

# %% papermill={"duration": 0.087168, "end_time": "2021-01-05T16:55:00.712457", "exception": false, "start_time": "2021-01-05T16:55:00.625289", "status": "completed"} tags=[]
assert np.all(
    [
        parts.loc[k].shape == (CLUSTERING_OPTIONS["N_REPS_PER_K"], data.shape[0])
        for k in parts.index
    ]
)

# %% [markdown] papermill={"duration": 0.070564, "end_time": "2021-01-05T16:55:00.854295", "exception": false, "start_time": "2021-01-05T16:55:00.783731", "status": "completed"} tags=[]
# ## Compute stability

# %% papermill={"duration": 0.085869, "end_time": "2021-01-05T16:55:01.012022", "exception": false, "start_time": "2021-01-05T16:55:00.926153", "status": "completed"} tags=[]
from sklearn.metrics import adjusted_rand_score as ari
from scipy.spatial.distance import squareform, pdist

# %% papermill={"duration": 0.774003, "end_time": "2021-01-05T16:55:01.857514", "exception": false, "start_time": "2021-01-05T16:55:01.083511", "status": "completed"} tags=[]
parts_ari = pd.Series(
    {k: pdist(parts.loc[k], metric=ari) for k in parts.index}, name="k"
)

# %% papermill={"duration": 0.089054, "end_time": "2021-01-05T16:55:02.018349", "exception": false, "start_time": "2021-01-05T16:55:01.929295", "status": "completed"} tags=[]
parts_ari_stability = parts_ari.apply(lambda x: x.mean())
display(parts_ari_stability.sort_values(ascending=False).head(15))

# %% papermill={"duration": 0.087117, "end_time": "2021-01-05T16:55:02.178093", "exception": false, "start_time": "2021-01-05T16:55:02.090976", "status": "completed"} tags=[]
parts_ari_df = pd.DataFrame.from_records(parts_ari.tolist()).set_index(
    parts_ari.index.copy()
)

# %% papermill={"duration": 0.087196, "end_time": "2021-01-05T16:55:02.336974", "exception": false, "start_time": "2021-01-05T16:55:02.249778", "status": "completed"} tags=[]
parts_ari_df.shape

# %% papermill={"duration": 0.086738, "end_time": "2021-01-05T16:55:02.498212", "exception": false, "start_time": "2021-01-05T16:55:02.411474", "status": "completed"} tags=[]
assert (
    int(
        (CLUSTERING_OPTIONS["N_REPS_PER_K"] * (CLUSTERING_OPTIONS["N_REPS_PER_K"] - 1))
        / 2
    )
    == parts_ari_df.shape[1]
)

# %% papermill={"duration": 0.091914, "end_time": "2021-01-05T16:55:02.661522", "exception": false, "start_time": "2021-01-05T16:55:02.569608", "status": "completed"} tags=[]
parts_ari_df.head()

# %% [markdown] papermill={"duration": 0.07076, "end_time": "2021-01-05T16:55:02.804616", "exception": false, "start_time": "2021-01-05T16:55:02.733856", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.087062, "end_time": "2021-01-05T16:55:02.962645", "exception": false, "start_time": "2021-01-05T16:55:02.875583", "status": "completed"} tags=[]
output_filename = Path(
    RESULTS_DIR,
    generate_result_set_name(
        CLUSTERING_OPTIONS,
        prefix=f"{clustering_method_name}-stability-",
        suffix=".pkl",
    ),
).resolve()
display(output_filename)

# %% papermill={"duration": 0.085798, "end_time": "2021-01-05T16:55:03.122182", "exception": false, "start_time": "2021-01-05T16:55:03.036384", "status": "completed"} tags=[]
parts_ari_df.to_pickle(output_filename)

# %% [markdown] papermill={"duration": 0.072366, "end_time": "2021-01-05T16:55:03.266946", "exception": false, "start_time": "2021-01-05T16:55:03.194580", "status": "completed"} tags=[]
# ## Stability plot

# %% papermill={"duration": 0.092085, "end_time": "2021-01-05T16:55:03.431062", "exception": false, "start_time": "2021-01-05T16:55:03.338977", "status": "completed"} tags=[]
parts_ari_df_plot = (
    parts_ari_df.stack()
    .reset_index()
    .rename(columns={"level_0": "k", "level_1": "idx", 0: "ari"})
)

# %% papermill={"duration": 0.088284, "end_time": "2021-01-05T16:55:03.592625", "exception": false, "start_time": "2021-01-05T16:55:03.504341", "status": "completed"} tags=[]
parts_ari_df_plot.dtypes

# %% papermill={"duration": 0.09094, "end_time": "2021-01-05T16:55:03.757003", "exception": false, "start_time": "2021-01-05T16:55:03.666063", "status": "completed"} tags=[]
parts_ari_df_plot.head()

# %% papermill={"duration": 2.998659, "end_time": "2021-01-05T16:55:06.828686", "exception": false, "start_time": "2021-01-05T16:55:03.830027", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.074406, "end_time": "2021-01-05T16:55:06.977905", "exception": false, "start_time": "2021-01-05T16:55:06.903499", "status": "completed"} tags=[]
