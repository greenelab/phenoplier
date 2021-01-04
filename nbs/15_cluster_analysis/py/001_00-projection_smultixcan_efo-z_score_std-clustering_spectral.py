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

# %% [markdown] papermill={"duration": 0.056657, "end_time": "2020-12-02T18:56:32.875092", "exception": false, "start_time": "2020-12-02T18:56:32.818435", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.019378, "end_time": "2020-12-02T18:56:32.915019", "exception": false, "start_time": "2020-12-02T18:56:32.895641", "status": "completed"} tags=[]
# Runs spectral clustering on the z_score_std version of the data.

# %% [markdown] papermill={"duration": 0.019439, "end_time": "2020-12-02T18:56:32.954066", "exception": false, "start_time": "2020-12-02T18:56:32.934627", "status": "completed"} tags=[]
# # Environment variables

# %% papermill={"duration": 0.033341, "end_time": "2020-12-02T18:56:33.006860", "exception": false, "start_time": "2020-12-02T18:56:32.973519", "status": "completed"} tags=[]
from IPython.display import display

import conf

N_JOBS = conf.GENERAL["N_JOBS"]
display(N_JOBS)

# %% papermill={"duration": 0.026988, "end_time": "2020-12-02T18:56:33.054115", "exception": false, "start_time": "2020-12-02T18:56:33.027127", "status": "completed"} tags=[]
# %env MKL_NUM_THREADS=$N_JOBS
# %env OPEN_BLAS_NUM_THREADS=$N_JOBS
# %env NUMEXPR_NUM_THREADS=$N_JOBS
# %env OMP_NUM_THREADS=$N_JOBS

# %% [markdown] papermill={"duration": 0.019958, "end_time": "2020-12-02T18:56:33.094388", "exception": false, "start_time": "2020-12-02T18:56:33.074430", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.030897, "end_time": "2020-12-02T18:56:33.145465", "exception": false, "start_time": "2020-12-02T18:56:33.114568", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 1.625233, "end_time": "2020-12-02T18:56:34.791127", "exception": false, "start_time": "2020-12-02T18:56:33.165894", "status": "completed"} tags=[]
from pathlib import Path

import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns

from utils import generate_result_set_name

# %% [markdown] papermill={"duration": 0.020541, "end_time": "2020-12-02T18:56:34.834353", "exception": false, "start_time": "2020-12-02T18:56:34.813812", "status": "completed"} tags=[]
# # Settings

# %% papermill={"duration": 0.034437, "end_time": "2020-12-02T18:56:34.889084", "exception": false, "start_time": "2020-12-02T18:56:34.854647", "status": "completed"} tags=[]
INITIAL_RANDOM_STATE = 30000

# %% [markdown] papermill={"duration": 0.0198, "end_time": "2020-12-02T18:56:34.929655", "exception": false, "start_time": "2020-12-02T18:56:34.909855", "status": "completed"} tags=[]
# ## Input data

# %% papermill={"duration": 0.034492, "end_time": "2020-12-02T18:56:34.984258", "exception": false, "start_time": "2020-12-02T18:56:34.949766", "status": "completed"} tags=[]
INPUT_SUBSET = "z_score_std"

# %% papermill={"duration": 0.034083, "end_time": "2020-12-02T18:56:35.038619", "exception": false, "start_time": "2020-12-02T18:56:35.004536", "status": "completed"} tags=[]
INPUT_STEM = "projection-smultixcan-efo_partial-mashr-zscores"

# %% papermill={"duration": 0.037262, "end_time": "2020-12-02T18:56:35.097414", "exception": false, "start_time": "2020-12-02T18:56:35.060152", "status": "completed"} tags=[]
input_filepath = Path(
    conf.RESULTS["DATA_TRANSFORMATIONS_DIR"],
    INPUT_SUBSET,
    f"{INPUT_SUBSET}-{INPUT_STEM}.pkl",
).resolve()
display(input_filepath)

assert input_filepath.exists(), "Input file does not exist"

input_filepath_stem = input_filepath.stem
display(input_filepath_stem)

# %% [markdown] papermill={"duration": 0.020581, "end_time": "2020-12-02T18:56:35.139806", "exception": false, "start_time": "2020-12-02T18:56:35.119225", "status": "completed"} tags=[]
# ## Clustering

# %% papermill={"duration": 0.040915, "end_time": "2020-12-02T18:56:35.201312", "exception": false, "start_time": "2020-12-02T18:56:35.160397", "status": "completed"} tags=[]
from sklearn.cluster import SpectralClustering

# %% papermill={"duration": 0.035633, "end_time": "2020-12-02T18:56:35.258455", "exception": false, "start_time": "2020-12-02T18:56:35.222822", "status": "completed"} tags=[]
CLUSTERING_ATTRIBUTES_TO_SAVE = ["n_clusters"]

# %% papermill={"duration": 0.036217, "end_time": "2020-12-02T18:56:35.315951", "exception": false, "start_time": "2020-12-02T18:56:35.279734", "status": "completed"} tags=[]
CLUSTERING_OPTIONS = {}

CLUSTERING_OPTIONS["K_MIN"] = 2
CLUSTERING_OPTIONS["K_MAX"] = 60  # sqrt(3749)
CLUSTERING_OPTIONS["N_REPS_PER_K"] = 5
CLUSTERING_OPTIONS["KMEANS_N_INIT"] = 10
CLUSTERING_OPTIONS["N_NEIGHBORS"] = 10
CLUSTERING_OPTIONS["AFFINITY"] = "nearest_neighbors"

display(CLUSTERING_OPTIONS)

# %% papermill={"duration": 0.037917, "end_time": "2020-12-02T18:56:35.375164", "exception": false, "start_time": "2020-12-02T18:56:35.337247", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.035289, "end_time": "2020-12-02T18:56:35.431257", "exception": false, "start_time": "2020-12-02T18:56:35.395968", "status": "completed"} tags=[]
display(len(CLUSTERERS))

# %% papermill={"duration": 0.038178, "end_time": "2020-12-02T18:56:35.490971", "exception": false, "start_time": "2020-12-02T18:56:35.452793", "status": "completed"} tags=[]
_iter = iter(CLUSTERERS.items())
display(next(_iter))
display(next(_iter))

# %% papermill={"duration": 0.038482, "end_time": "2020-12-02T18:56:35.552539", "exception": false, "start_time": "2020-12-02T18:56:35.514057", "status": "completed"} tags=[]
clustering_method_name = method_name
display(clustering_method_name)

# %% [markdown] papermill={"duration": 0.021954, "end_time": "2020-12-02T18:56:35.598393", "exception": false, "start_time": "2020-12-02T18:56:35.576439", "status": "completed"} tags=[]
# ## Output directory

# %% papermill={"duration": 0.037373, "end_time": "2020-12-02T18:56:35.657548", "exception": false, "start_time": "2020-12-02T18:56:35.620175", "status": "completed"} tags=[]
# output dir for this notebook
RESULTS_DIR = Path(
    conf.RESULTS["CLUSTERING_RUNS_DIR"],
    f"{INPUT_SUBSET}-{INPUT_STEM}",
).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %% [markdown] papermill={"duration": 0.021925, "end_time": "2020-12-02T18:56:35.701847", "exception": false, "start_time": "2020-12-02T18:56:35.679922", "status": "completed"} tags=[]
# # Load input file

# %% papermill={"duration": 0.048388, "end_time": "2020-12-02T18:56:35.771954", "exception": false, "start_time": "2020-12-02T18:56:35.723566", "status": "completed"} tags=[]
data = pd.read_pickle(input_filepath)

# %% papermill={"duration": 0.037414, "end_time": "2020-12-02T18:56:35.833413", "exception": false, "start_time": "2020-12-02T18:56:35.795999", "status": "completed"} tags=[]
data.shape

# %% papermill={"duration": 0.050003, "end_time": "2020-12-02T18:56:35.906992", "exception": false, "start_time": "2020-12-02T18:56:35.856989", "status": "completed"} tags=[]
data.head()

# %% papermill={"duration": 0.04232, "end_time": "2020-12-02T18:56:35.972513", "exception": false, "start_time": "2020-12-02T18:56:35.930193", "status": "completed"} tags=[]
assert not data.isna().any().any()

# %% [markdown] papermill={"duration": 0.022953, "end_time": "2020-12-02T18:56:36.020280", "exception": false, "start_time": "2020-12-02T18:56:35.997327", "status": "completed"} tags=[]
# # Clustering

# %% [markdown] papermill={"duration": 0.022279, "end_time": "2020-12-02T18:56:36.064949", "exception": false, "start_time": "2020-12-02T18:56:36.042670", "status": "completed"} tags=[]
# ## Generate ensemble

# %% papermill={"duration": 0.039289, "end_time": "2020-12-02T18:56:36.126663", "exception": false, "start_time": "2020-12-02T18:56:36.087374", "status": "completed"} tags=[]
from clustering.ensemble import generate_ensemble

# %% papermill={"duration": 8345.210119, "end_time": "2020-12-02T21:15:41.359537", "exception": false, "start_time": "2020-12-02T18:56:36.149418", "status": "completed"} tags=[]
ensemble = generate_ensemble(
    data,
    CLUSTERERS,
    attributes=CLUSTERING_ATTRIBUTES_TO_SAVE,
)

# %% papermill={"duration": 0.08423, "end_time": "2020-12-02T21:15:41.513117", "exception": false, "start_time": "2020-12-02T21:15:41.428887", "status": "completed"} tags=[]
# the number should be close to 295 (the number of partitions generated by k-means/spectral clustering)
ensemble.shape

# %% papermill={"duration": 0.092134, "end_time": "2020-12-02T21:15:41.674362", "exception": false, "start_time": "2020-12-02T21:15:41.582228", "status": "completed"} tags=[]
ensemble.head()

# %% papermill={"duration": 0.08595, "end_time": "2020-12-02T21:15:41.832573", "exception": false, "start_time": "2020-12-02T21:15:41.746623", "status": "completed"} tags=[]
ensemble["n_clusters"].value_counts().head()

# %% papermill={"duration": 0.08677, "end_time": "2020-12-02T21:15:41.990672", "exception": false, "start_time": "2020-12-02T21:15:41.903902", "status": "completed"} tags=[]
ensemble_stats = ensemble["n_clusters"].describe()
display(ensemble_stats)

# %% [markdown] papermill={"duration": 0.06938, "end_time": "2020-12-02T21:15:42.129436", "exception": false, "start_time": "2020-12-02T21:15:42.060056", "status": "completed"} tags=[]
# ## Testing

# %% papermill={"duration": 0.083795, "end_time": "2020-12-02T21:15:42.282610", "exception": false, "start_time": "2020-12-02T21:15:42.198815", "status": "completed"} tags=[]
assert ensemble_stats["min"] > 1

# %% papermill={"duration": 0.083625, "end_time": "2020-12-02T21:15:42.435866", "exception": false, "start_time": "2020-12-02T21:15:42.352241", "status": "completed"} tags=[]
assert not ensemble["n_clusters"].isna().any()

# %% papermill={"duration": 0.083812, "end_time": "2020-12-02T21:15:42.589471", "exception": false, "start_time": "2020-12-02T21:15:42.505659", "status": "completed"} tags=[]
assert ensemble.shape[0] == len(CLUSTERERS)

# %% papermill={"duration": 0.104327, "end_time": "2020-12-02T21:15:42.763575", "exception": false, "start_time": "2020-12-02T21:15:42.659248", "status": "completed"} tags=[]
# all partitions have the right size
assert np.all(
    [part["partition"].shape[0] == data.shape[0] for idx, part in ensemble.iterrows()]
)

# %% papermill={"duration": 0.106328, "end_time": "2020-12-02T21:15:42.941229", "exception": false, "start_time": "2020-12-02T21:15:42.834901", "status": "completed"} tags=[]
# no partition has negative clusters (noisy points)
assert not np.any([(part["partition"] < 0).any() for idx, part in ensemble.iterrows()])

# %% [markdown] papermill={"duration": 0.069369, "end_time": "2020-12-02T21:15:43.081325", "exception": false, "start_time": "2020-12-02T21:15:43.011956", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.085589, "end_time": "2020-12-02T21:15:43.236864", "exception": false, "start_time": "2020-12-02T21:15:43.151275", "status": "completed"} tags=[]
output_filename = Path(
    RESULTS_DIR,
    generate_result_set_name(
        CLUSTERING_OPTIONS,
        prefix=f"{clustering_method_name}-",
        suffix=".pkl",
    ),
).resolve()
display(output_filename)

# %% papermill={"duration": 0.095978, "end_time": "2020-12-02T21:15:43.404015", "exception": false, "start_time": "2020-12-02T21:15:43.308037", "status": "completed"} tags=[]
ensemble.to_pickle(output_filename)

# %% [markdown] papermill={"duration": 0.06952, "end_time": "2020-12-02T21:15:43.545035", "exception": false, "start_time": "2020-12-02T21:15:43.475515", "status": "completed"} tags=[]
# # Stability

# %% [markdown] papermill={"duration": 0.0689, "end_time": "2020-12-02T21:15:43.684488", "exception": false, "start_time": "2020-12-02T21:15:43.615588", "status": "completed"} tags=[]
# ## Group ensemble by n_clusters

# %% papermill={"duration": 0.100389, "end_time": "2020-12-02T21:15:43.857972", "exception": false, "start_time": "2020-12-02T21:15:43.757583", "status": "completed"} tags=[]
parts = ensemble.groupby("n_clusters").apply(
    lambda x: np.concatenate(x["partition"].apply(lambda x: x.reshape(1, -1)), axis=0)
)

# %% papermill={"duration": 0.095484, "end_time": "2020-12-02T21:15:44.024362", "exception": false, "start_time": "2020-12-02T21:15:43.928878", "status": "completed"} tags=[]
parts.head()

# %% papermill={"duration": 0.086393, "end_time": "2020-12-02T21:15:44.181296", "exception": false, "start_time": "2020-12-02T21:15:44.094903", "status": "completed"} tags=[]
assert np.all(
    [
        parts.loc[k].shape == (CLUSTERING_OPTIONS["N_REPS_PER_K"], data.shape[0])
        for k in parts.index
    ]
)

# %% [markdown] papermill={"duration": 0.06923, "end_time": "2020-12-02T21:15:44.319585", "exception": false, "start_time": "2020-12-02T21:15:44.250355", "status": "completed"} tags=[]
# ## Compute stability

# %% papermill={"duration": 0.084534, "end_time": "2020-12-02T21:15:44.473509", "exception": false, "start_time": "2020-12-02T21:15:44.388975", "status": "completed"} tags=[]
from sklearn.metrics import adjusted_rand_score as ari
from scipy.spatial.distance import squareform, pdist

# %% papermill={"duration": 0.614006, "end_time": "2020-12-02T21:15:45.159385", "exception": false, "start_time": "2020-12-02T21:15:44.545379", "status": "completed"} tags=[]
parts_ari = pd.Series(
    {k: pdist(parts.loc[k], metric=ari) for k in parts.index}, name="k"
)

# %% papermill={"duration": 0.08689, "end_time": "2020-12-02T21:15:45.317037", "exception": false, "start_time": "2020-12-02T21:15:45.230147", "status": "completed"} tags=[]
parts_ari_stability = parts_ari.apply(lambda x: x.mean())
display(parts_ari_stability.sort_values(ascending=False).head(15))

# %% papermill={"duration": 0.085969, "end_time": "2020-12-02T21:15:45.474270", "exception": false, "start_time": "2020-12-02T21:15:45.388301", "status": "completed"} tags=[]
parts_ari_df = pd.DataFrame.from_records(parts_ari.tolist()).set_index(
    parts_ari.index.copy()
)

# %% papermill={"duration": 0.086489, "end_time": "2020-12-02T21:15:45.631698", "exception": false, "start_time": "2020-12-02T21:15:45.545209", "status": "completed"} tags=[]
parts_ari_df.shape

# %% papermill={"duration": 0.088262, "end_time": "2020-12-02T21:15:45.790773", "exception": false, "start_time": "2020-12-02T21:15:45.702511", "status": "completed"} tags=[]
assert (
    int(
        (CLUSTERING_OPTIONS["N_REPS_PER_K"] * (CLUSTERING_OPTIONS["N_REPS_PER_K"] - 1))
        / 2
    )
    == parts_ari_df.shape[1]
)

# %% papermill={"duration": 0.092257, "end_time": "2020-12-02T21:15:45.954784", "exception": false, "start_time": "2020-12-02T21:15:45.862527", "status": "completed"} tags=[]
parts_ari_df.head()

# %% [markdown] papermill={"duration": 0.070535, "end_time": "2020-12-02T21:15:46.097185", "exception": false, "start_time": "2020-12-02T21:15:46.026650", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.086844, "end_time": "2020-12-02T21:15:46.254727", "exception": false, "start_time": "2020-12-02T21:15:46.167883", "status": "completed"} tags=[]
output_filename = Path(
    RESULTS_DIR,
    generate_result_set_name(
        CLUSTERING_OPTIONS,
        prefix=f"{clustering_method_name}-stability-",
        suffix=".pkl",
    ),
).resolve()
display(output_filename)

# %% papermill={"duration": 0.086859, "end_time": "2020-12-02T21:15:46.413298", "exception": false, "start_time": "2020-12-02T21:15:46.326439", "status": "completed"} tags=[]
parts_ari_df.to_pickle(output_filename)

# %% [markdown] papermill={"duration": 0.070496, "end_time": "2020-12-02T21:15:46.554151", "exception": false, "start_time": "2020-12-02T21:15:46.483655", "status": "completed"} tags=[]
# ## Stability plot

# %% papermill={"duration": 0.088038, "end_time": "2020-12-02T21:15:46.713710", "exception": false, "start_time": "2020-12-02T21:15:46.625672", "status": "completed"} tags=[]
parts_ari_df_plot = (
    parts_ari_df.stack()
    .reset_index()
    .rename(columns={"level_0": "k", "level_1": "idx", 0: "ari"})
)

# %% papermill={"duration": 0.089326, "end_time": "2020-12-02T21:15:46.877170", "exception": false, "start_time": "2020-12-02T21:15:46.787844", "status": "completed"} tags=[]
parts_ari_df_plot.dtypes

# %% papermill={"duration": 0.089177, "end_time": "2020-12-02T21:15:47.039076", "exception": false, "start_time": "2020-12-02T21:15:46.949899", "status": "completed"} tags=[]
parts_ari_df_plot.head()

# %% papermill={"duration": 3.057362, "end_time": "2020-12-02T21:15:50.169021", "exception": false, "start_time": "2020-12-02T21:15:47.111659", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.072467, "end_time": "2020-12-02T21:15:50.314333", "exception": false, "start_time": "2020-12-02T21:15:50.241866", "status": "completed"} tags=[]
