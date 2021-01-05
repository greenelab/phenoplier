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

# %% [markdown] papermill={"duration": 0.056714, "end_time": "2020-12-02T18:50:06.623779", "exception": false, "start_time": "2020-12-02T18:50:06.567065", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.020258, "end_time": "2020-12-02T18:50:06.664733", "exception": false, "start_time": "2020-12-02T18:50:06.644475", "status": "completed"} tags=[]
# Runs k-means on the umap version of the data.

# %% [markdown] papermill={"duration": 0.019621, "end_time": "2020-12-02T18:50:06.704498", "exception": false, "start_time": "2020-12-02T18:50:06.684877", "status": "completed"} tags=[]
# # Environment variables

# %% papermill={"duration": 0.033616, "end_time": "2020-12-02T18:50:06.758120", "exception": false, "start_time": "2020-12-02T18:50:06.724504", "status": "completed"} tags=[]
from IPython.display import display

import conf

N_JOBS = conf.GENERAL["N_JOBS"]
display(N_JOBS)

# %% papermill={"duration": 0.02707, "end_time": "2020-12-02T18:50:06.806150", "exception": false, "start_time": "2020-12-02T18:50:06.779080", "status": "completed"} tags=[]
# %env MKL_NUM_THREADS=$N_JOBS
# %env OPEN_BLAS_NUM_THREADS=$N_JOBS
# %env NUMEXPR_NUM_THREADS=$N_JOBS
# %env OMP_NUM_THREADS=$N_JOBS

# %% [markdown] papermill={"duration": 0.020438, "end_time": "2020-12-02T18:50:06.847573", "exception": false, "start_time": "2020-12-02T18:50:06.827135", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.030859, "end_time": "2020-12-02T18:50:06.898514", "exception": false, "start_time": "2020-12-02T18:50:06.867655", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 1.594128, "end_time": "2020-12-02T18:50:08.513434", "exception": false, "start_time": "2020-12-02T18:50:06.919306", "status": "completed"} tags=[]
from pathlib import Path

import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns

from utils import generate_result_set_name

# %% [markdown] papermill={"duration": 0.020105, "end_time": "2020-12-02T18:50:08.555767", "exception": false, "start_time": "2020-12-02T18:50:08.535662", "status": "completed"} tags=[]
# # Settings

# %% papermill={"duration": 0.034365, "end_time": "2020-12-02T18:50:08.610087", "exception": false, "start_time": "2020-12-02T18:50:08.575722", "status": "completed"} tags=[]
INITIAL_RANDOM_STATE = 20000

# %% [markdown] papermill={"duration": 0.020198, "end_time": "2020-12-02T18:50:08.650880", "exception": false, "start_time": "2020-12-02T18:50:08.630682", "status": "completed"} tags=[]
# ## Input data

# %% papermill={"duration": 0.03417, "end_time": "2020-12-02T18:50:08.704929", "exception": false, "start_time": "2020-12-02T18:50:08.670759", "status": "completed"} tags=[]
INPUT_SUBSET = "umap"

# %% papermill={"duration": 0.034018, "end_time": "2020-12-02T18:50:08.758956", "exception": false, "start_time": "2020-12-02T18:50:08.724938", "status": "completed"} tags=[]
INPUT_STEM = "z_score_std-projection-smultixcan-efo_partial-mashr-zscores"

# %% papermill={"duration": 0.033975, "end_time": "2020-12-02T18:50:08.813348", "exception": false, "start_time": "2020-12-02T18:50:08.779373", "status": "completed"} tags=[]
DR_OPTIONS = {
    "n_components": 50,
    "metric": "euclidean",
    "n_neighbors": 15,
    "random_state": 0,
}

# %% papermill={"duration": 0.036937, "end_time": "2020-12-02T18:50:08.871242", "exception": false, "start_time": "2020-12-02T18:50:08.834305", "status": "completed"} tags=[]
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

# %% [markdown] papermill={"duration": 0.020712, "end_time": "2020-12-02T18:50:08.913277", "exception": false, "start_time": "2020-12-02T18:50:08.892565", "status": "completed"} tags=[]
# ## Clustering

# %% papermill={"duration": 0.042447, "end_time": "2020-12-02T18:50:08.976054", "exception": false, "start_time": "2020-12-02T18:50:08.933607", "status": "completed"} tags=[]
from sklearn.cluster import KMeans

# %% papermill={"duration": 0.035328, "end_time": "2020-12-02T18:50:09.033551", "exception": false, "start_time": "2020-12-02T18:50:08.998223", "status": "completed"} tags=[]
CLUSTERING_ATTRIBUTES_TO_SAVE = ["n_clusters"]

# %% papermill={"duration": 0.036041, "end_time": "2020-12-02T18:50:09.091263", "exception": false, "start_time": "2020-12-02T18:50:09.055222", "status": "completed"} tags=[]
CLUSTERING_OPTIONS = {}

CLUSTERING_OPTIONS["K_MIN"] = 2
CLUSTERING_OPTIONS["K_MAX"] = 60  # sqrt(3749)
CLUSTERING_OPTIONS["N_REPS_PER_K"] = 5
CLUSTERING_OPTIONS["KMEANS_N_INIT"] = 10

display(CLUSTERING_OPTIONS)

# %% papermill={"duration": 0.036947, "end_time": "2020-12-02T18:50:09.149420", "exception": false, "start_time": "2020-12-02T18:50:09.112473", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.035849, "end_time": "2020-12-02T18:50:09.206660", "exception": false, "start_time": "2020-12-02T18:50:09.170811", "status": "completed"} tags=[]
display(len(CLUSTERERS))

# %% papermill={"duration": 0.03753, "end_time": "2020-12-02T18:50:09.265615", "exception": false, "start_time": "2020-12-02T18:50:09.228085", "status": "completed"} tags=[]
_iter = iter(CLUSTERERS.items())
display(next(_iter))
display(next(_iter))

# %% papermill={"duration": 0.036575, "end_time": "2020-12-02T18:50:09.324955", "exception": false, "start_time": "2020-12-02T18:50:09.288380", "status": "completed"} tags=[]
clustering_method_name = method_name
display(clustering_method_name)

# %% [markdown] papermill={"duration": 0.02212, "end_time": "2020-12-02T18:50:09.368851", "exception": false, "start_time": "2020-12-02T18:50:09.346731", "status": "completed"} tags=[]
# ## Output directory

# %% papermill={"duration": 0.038234, "end_time": "2020-12-02T18:50:09.428792", "exception": false, "start_time": "2020-12-02T18:50:09.390558", "status": "completed"} tags=[]
# output dir for this notebook
RESULTS_DIR = Path(
    conf.RESULTS["CLUSTERING_RUNS_DIR"],
    f"{INPUT_SUBSET}-{INPUT_STEM}",
).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %% [markdown] papermill={"duration": 0.021784, "end_time": "2020-12-02T18:50:09.474069", "exception": false, "start_time": "2020-12-02T18:50:09.452285", "status": "completed"} tags=[]
# # Load input file

# %% papermill={"duration": 0.037442, "end_time": "2020-12-02T18:50:09.533424", "exception": false, "start_time": "2020-12-02T18:50:09.495982", "status": "completed"} tags=[]
data = pd.read_pickle(input_filepath)

# %% papermill={"duration": 0.036423, "end_time": "2020-12-02T18:50:09.592256", "exception": false, "start_time": "2020-12-02T18:50:09.555833", "status": "completed"} tags=[]
data.shape

# %% papermill={"duration": 0.04946, "end_time": "2020-12-02T18:50:09.664275", "exception": false, "start_time": "2020-12-02T18:50:09.614815", "status": "completed"} tags=[]
data.head()

# %% papermill={"duration": 0.037074, "end_time": "2020-12-02T18:50:09.724134", "exception": false, "start_time": "2020-12-02T18:50:09.687060", "status": "completed"} tags=[]
assert not data.isna().any().any()

# %% [markdown] papermill={"duration": 0.022234, "end_time": "2020-12-02T18:50:09.768856", "exception": false, "start_time": "2020-12-02T18:50:09.746622", "status": "completed"} tags=[]
# # Clustering

# %% [markdown] papermill={"duration": 0.021974, "end_time": "2020-12-02T18:50:09.812884", "exception": false, "start_time": "2020-12-02T18:50:09.790910", "status": "completed"} tags=[]
# ## Generate ensemble

# %% papermill={"duration": 0.040006, "end_time": "2020-12-02T18:50:09.874911", "exception": false, "start_time": "2020-12-02T18:50:09.834905", "status": "completed"} tags=[]
from clustering.ensemble import generate_ensemble

# %% papermill={"duration": 372.109662, "end_time": "2020-12-02T18:56:22.009342", "exception": false, "start_time": "2020-12-02T18:50:09.899680", "status": "completed"} tags=[]
ensemble = generate_ensemble(
    data,
    CLUSTERERS,
    attributes=CLUSTERING_ATTRIBUTES_TO_SAVE,
)

# %% papermill={"duration": 0.081664, "end_time": "2020-12-02T18:56:22.159058", "exception": false, "start_time": "2020-12-02T18:56:22.077394", "status": "completed"} tags=[]
# the number should be close to 295 (the number of partitions generated by k-means/spectral clustering)
ensemble.shape

# %% papermill={"duration": 0.089733, "end_time": "2020-12-02T18:56:22.317322", "exception": false, "start_time": "2020-12-02T18:56:22.227589", "status": "completed"} tags=[]
ensemble.head()

# %% papermill={"duration": 0.085079, "end_time": "2020-12-02T18:56:22.471654", "exception": false, "start_time": "2020-12-02T18:56:22.386575", "status": "completed"} tags=[]
ensemble["n_clusters"].value_counts().head()

# %% papermill={"duration": 0.084878, "end_time": "2020-12-02T18:56:22.626588", "exception": false, "start_time": "2020-12-02T18:56:22.541710", "status": "completed"} tags=[]
ensemble_stats = ensemble["n_clusters"].describe()
display(ensemble_stats)

# %% [markdown] papermill={"duration": 0.067602, "end_time": "2020-12-02T18:56:22.762736", "exception": false, "start_time": "2020-12-02T18:56:22.695134", "status": "completed"} tags=[]
# ## Testing

# %% papermill={"duration": 0.082741, "end_time": "2020-12-02T18:56:22.913598", "exception": false, "start_time": "2020-12-02T18:56:22.830857", "status": "completed"} tags=[]
assert ensemble_stats["min"] > 1

# %% papermill={"duration": 0.082681, "end_time": "2020-12-02T18:56:23.065213", "exception": false, "start_time": "2020-12-02T18:56:22.982532", "status": "completed"} tags=[]
assert not ensemble["n_clusters"].isna().any()

# %% papermill={"duration": 0.08285, "end_time": "2020-12-02T18:56:23.218128", "exception": false, "start_time": "2020-12-02T18:56:23.135278", "status": "completed"} tags=[]
assert ensemble.shape[0] == len(CLUSTERERS)

# %% papermill={"duration": 0.10096, "end_time": "2020-12-02T18:56:23.387804", "exception": false, "start_time": "2020-12-02T18:56:23.286844", "status": "completed"} tags=[]
# all partitions have the right size
assert np.all(
    [part["partition"].shape[0] == data.shape[0] for idx, part in ensemble.iterrows()]
)

# %% papermill={"duration": 0.107181, "end_time": "2020-12-02T18:56:23.564106", "exception": false, "start_time": "2020-12-02T18:56:23.456925", "status": "completed"} tags=[]
# no partition has negative clusters (noisy points)
assert not np.any([(part["partition"] < 0).any() for idx, part in ensemble.iterrows()])

# %% [markdown] papermill={"duration": 0.06929, "end_time": "2020-12-02T18:56:23.703396", "exception": false, "start_time": "2020-12-02T18:56:23.634106", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.084516, "end_time": "2020-12-02T18:56:23.856127", "exception": false, "start_time": "2020-12-02T18:56:23.771611", "status": "completed"} tags=[]
output_filename = Path(
    RESULTS_DIR,
    generate_result_set_name(
        CLUSTERING_OPTIONS,
        prefix=f"{clustering_method_name}-",
        suffix=".pkl",
    ),
).resolve()
display(output_filename)

# %% papermill={"duration": 0.094654, "end_time": "2020-12-02T18:56:24.020908", "exception": false, "start_time": "2020-12-02T18:56:23.926254", "status": "completed"} tags=[]
ensemble.to_pickle(output_filename)

# %% [markdown] papermill={"duration": 0.06903, "end_time": "2020-12-02T18:56:24.160863", "exception": false, "start_time": "2020-12-02T18:56:24.091833", "status": "completed"} tags=[]
# # Stability

# %% [markdown] papermill={"duration": 0.068939, "end_time": "2020-12-02T18:56:24.299714", "exception": false, "start_time": "2020-12-02T18:56:24.230775", "status": "completed"} tags=[]
# ## Group ensemble by n_clusters

# %% papermill={"duration": 0.099924, "end_time": "2020-12-02T18:56:24.469107", "exception": false, "start_time": "2020-12-02T18:56:24.369183", "status": "completed"} tags=[]
parts = ensemble.groupby("n_clusters").apply(
    lambda x: np.concatenate(x["partition"].apply(lambda x: x.reshape(1, -1)), axis=0)
)

# %% papermill={"duration": 0.095843, "end_time": "2020-12-02T18:56:24.636935", "exception": false, "start_time": "2020-12-02T18:56:24.541092", "status": "completed"} tags=[]
parts.head()

# %% papermill={"duration": 0.083895, "end_time": "2020-12-02T18:56:24.791118", "exception": false, "start_time": "2020-12-02T18:56:24.707223", "status": "completed"} tags=[]
assert np.all(
    [
        parts.loc[k].shape == (CLUSTERING_OPTIONS["N_REPS_PER_K"], data.shape[0])
        for k in parts.index
    ]
)

# %% [markdown] papermill={"duration": 0.069711, "end_time": "2020-12-02T18:56:24.930841", "exception": false, "start_time": "2020-12-02T18:56:24.861130", "status": "completed"} tags=[]
# ## Compute stability

# %% papermill={"duration": 0.083734, "end_time": "2020-12-02T18:56:25.083922", "exception": false, "start_time": "2020-12-02T18:56:25.000188", "status": "completed"} tags=[]
from sklearn.metrics import adjusted_rand_score as ari
from scipy.spatial.distance import squareform, pdist

# %% papermill={"duration": 0.772839, "end_time": "2020-12-02T18:56:25.926542", "exception": false, "start_time": "2020-12-02T18:56:25.153703", "status": "completed"} tags=[]
parts_ari = pd.Series(
    {k: pdist(parts.loc[k], metric=ari) for k in parts.index}, name="k"
)

# %% papermill={"duration": 0.085482, "end_time": "2020-12-02T18:56:26.082740", "exception": false, "start_time": "2020-12-02T18:56:25.997258", "status": "completed"} tags=[]
parts_ari_stability = parts_ari.apply(lambda x: x.mean())
display(parts_ari_stability.sort_values(ascending=False).head(15))

# %% papermill={"duration": 0.087331, "end_time": "2020-12-02T18:56:26.241110", "exception": false, "start_time": "2020-12-02T18:56:26.153779", "status": "completed"} tags=[]
parts_ari_df = pd.DataFrame.from_records(parts_ari.tolist()).set_index(
    parts_ari.index.copy()
)

# %% papermill={"duration": 0.085678, "end_time": "2020-12-02T18:56:26.399680", "exception": false, "start_time": "2020-12-02T18:56:26.314002", "status": "completed"} tags=[]
parts_ari_df.shape

# %% papermill={"duration": 0.086206, "end_time": "2020-12-02T18:56:26.557306", "exception": false, "start_time": "2020-12-02T18:56:26.471100", "status": "completed"} tags=[]
assert (
    int(
        (CLUSTERING_OPTIONS["N_REPS_PER_K"] * (CLUSTERING_OPTIONS["N_REPS_PER_K"] - 1))
        / 2
    )
    == parts_ari_df.shape[1]
)

# %% papermill={"duration": 0.090734, "end_time": "2020-12-02T18:56:26.718160", "exception": false, "start_time": "2020-12-02T18:56:26.627426", "status": "completed"} tags=[]
parts_ari_df.head()

# %% [markdown] papermill={"duration": 0.069441, "end_time": "2020-12-02T18:56:26.858660", "exception": false, "start_time": "2020-12-02T18:56:26.789219", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.085017, "end_time": "2020-12-02T18:56:27.013943", "exception": false, "start_time": "2020-12-02T18:56:26.928926", "status": "completed"} tags=[]
output_filename = Path(
    RESULTS_DIR,
    generate_result_set_name(
        CLUSTERING_OPTIONS,
        prefix=f"{clustering_method_name}-stability-",
        suffix=".pkl",
    ),
).resolve()
display(output_filename)

# %% papermill={"duration": 0.085149, "end_time": "2020-12-02T18:56:27.170961", "exception": false, "start_time": "2020-12-02T18:56:27.085812", "status": "completed"} tags=[]
parts_ari_df.to_pickle(output_filename)

# %% [markdown] papermill={"duration": 0.07034, "end_time": "2020-12-02T18:56:27.311990", "exception": false, "start_time": "2020-12-02T18:56:27.241650", "status": "completed"} tags=[]
# ## Stability plot

# %% papermill={"duration": 0.086939, "end_time": "2020-12-02T18:56:27.468686", "exception": false, "start_time": "2020-12-02T18:56:27.381747", "status": "completed"} tags=[]
parts_ari_df_plot = (
    parts_ari_df.stack()
    .reset_index()
    .rename(columns={"level_0": "k", "level_1": "idx", 0: "ari"})
)

# %% papermill={"duration": 0.085429, "end_time": "2020-12-02T18:56:27.627699", "exception": false, "start_time": "2020-12-02T18:56:27.542270", "status": "completed"} tags=[]
parts_ari_df_plot.dtypes

# %% papermill={"duration": 0.087352, "end_time": "2020-12-02T18:56:27.785614", "exception": false, "start_time": "2020-12-02T18:56:27.698262", "status": "completed"} tags=[]
parts_ari_df_plot.head()

# %% papermill={"duration": 2.961661, "end_time": "2020-12-02T18:56:30.818746", "exception": false, "start_time": "2020-12-02T18:56:27.857085", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.071919, "end_time": "2020-12-02T18:56:30.962663", "exception": false, "start_time": "2020-12-02T18:56:30.890744", "status": "completed"} tags=[]
