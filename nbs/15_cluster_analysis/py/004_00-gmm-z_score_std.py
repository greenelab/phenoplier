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

# %% [markdown] papermill={"duration": 0.057183, "end_time": "2021-01-05T22:03:08.974110", "exception": false, "start_time": "2021-01-05T22:03:08.916927", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.020457, "end_time": "2021-01-05T22:03:09.015800", "exception": false, "start_time": "2021-01-05T22:03:08.995343", "status": "completed"} tags=[]
# Runs gaussian mixture model on the z_score_std version of the data.

# %% [markdown] papermill={"duration": 0.022292, "end_time": "2021-01-05T22:03:09.058759", "exception": false, "start_time": "2021-01-05T22:03:09.036467", "status": "completed"} tags=[]
# # Environment variables

# %% papermill={"duration": 0.034622, "end_time": "2021-01-05T22:03:09.114439", "exception": false, "start_time": "2021-01-05T22:03:09.079817", "status": "completed"} tags=[]
from IPython.display import display

import conf

N_JOBS = conf.GENERAL["N_JOBS"]
display(N_JOBS)

# %% papermill={"duration": 0.02771, "end_time": "2021-01-05T22:03:09.163300", "exception": false, "start_time": "2021-01-05T22:03:09.135590", "status": "completed"} tags=[]
# %env MKL_NUM_THREADS=$N_JOBS
# %env OPEN_BLAS_NUM_THREADS=$N_JOBS
# %env NUMEXPR_NUM_THREADS=$N_JOBS
# %env OMP_NUM_THREADS=$N_JOBS

# %% [markdown] papermill={"duration": 0.02101, "end_time": "2021-01-05T22:03:09.205873", "exception": false, "start_time": "2021-01-05T22:03:09.184863", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.031389, "end_time": "2021-01-05T22:03:09.258220", "exception": false, "start_time": "2021-01-05T22:03:09.226831", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 1.600495, "end_time": "2021-01-05T22:03:10.880366", "exception": false, "start_time": "2021-01-05T22:03:09.279871", "status": "completed"} tags=[]
from pathlib import Path

import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns

from utils import generate_result_set_name

# %% [markdown] papermill={"duration": 0.020787, "end_time": "2021-01-05T22:03:10.924173", "exception": false, "start_time": "2021-01-05T22:03:10.903386", "status": "completed"} tags=[]
# # Settings

# %% papermill={"duration": 0.038191, "end_time": "2021-01-05T22:03:10.983170", "exception": false, "start_time": "2021-01-05T22:03:10.944979", "status": "completed"} tags=[]
INITIAL_RANDOM_STATE = 60000

# %% [markdown] papermill={"duration": 0.022204, "end_time": "2021-01-05T22:03:11.042176", "exception": false, "start_time": "2021-01-05T22:03:11.019972", "status": "completed"} tags=[]
# ## Input data

# %% papermill={"duration": 0.035782, "end_time": "2021-01-05T22:03:11.101172", "exception": false, "start_time": "2021-01-05T22:03:11.065390", "status": "completed"} tags=[]
INPUT_SUBSET = "z_score_std"

# %% papermill={"duration": 0.034498, "end_time": "2021-01-05T22:03:11.157424", "exception": false, "start_time": "2021-01-05T22:03:11.122926", "status": "completed"} tags=[]
INPUT_STEM = "projection-smultixcan-efo_partial-mashr-zscores"

# %% papermill={"duration": 0.037613, "end_time": "2021-01-05T22:03:11.216848", "exception": false, "start_time": "2021-01-05T22:03:11.179235", "status": "completed"} tags=[]
input_filepath = Path(
    conf.RESULTS["DATA_TRANSFORMATIONS_DIR"],
    INPUT_SUBSET,
    f"{INPUT_SUBSET}-{INPUT_STEM}.pkl",
).resolve()
display(input_filepath)

assert input_filepath.exists(), "Input file does not exist"

input_filepath_stem = input_filepath.stem
display(input_filepath_stem)

# %% [markdown] papermill={"duration": 0.021134, "end_time": "2021-01-05T22:03:11.259508", "exception": false, "start_time": "2021-01-05T22:03:11.238374", "status": "completed"} tags=[]
# ## Clustering

# %% papermill={"duration": 0.050317, "end_time": "2021-01-05T22:03:11.331004", "exception": false, "start_time": "2021-01-05T22:03:11.280687", "status": "completed"} tags=[]
from sklearn.mixture import GaussianMixture

# %% papermill={"duration": 0.035807, "end_time": "2021-01-05T22:03:11.388880", "exception": false, "start_time": "2021-01-05T22:03:11.353073", "status": "completed"} tags=[]
CLUSTERING_ATTRIBUTES_TO_SAVE = ["n_clusters"]

# %% papermill={"duration": 0.037229, "end_time": "2021-01-05T22:03:11.448131", "exception": false, "start_time": "2021-01-05T22:03:11.410902", "status": "completed"} tags=[]
CLUSTERING_OPTIONS = {}

CLUSTERING_OPTIONS["K_MIN"] = 2
CLUSTERING_OPTIONS["K_MAX"] = 60  # sqrt(3749)
CLUSTERING_OPTIONS["N_REPS_PER_K"] = 5
CLUSTERING_OPTIONS["N_INIT"] = 10
CLUSTERING_OPTIONS["COVARIANCE_TYPE"] = "full"

display(CLUSTERING_OPTIONS)

# %% papermill={"duration": 0.037756, "end_time": "2021-01-05T22:03:11.508478", "exception": false, "start_time": "2021-01-05T22:03:11.470722", "status": "completed"} tags=[]
CLUSTERERS = {}

idx = 0
random_state = INITIAL_RANDOM_STATE

for k in range(CLUSTERING_OPTIONS["K_MIN"], CLUSTERING_OPTIONS["K_MAX"] + 1):
    for i in range(CLUSTERING_OPTIONS["N_REPS_PER_K"]):
        clus = GaussianMixture(
            n_components=k,
            n_init=CLUSTERING_OPTIONS["N_INIT"],
            covariance_type=CLUSTERING_OPTIONS["COVARIANCE_TYPE"],
            random_state=random_state,
        )

        method_name = type(clus).__name__
        CLUSTERERS[f"{method_name} #{idx}"] = clus

        random_state = random_state + 1
        idx = idx + 1

# %% papermill={"duration": 0.037475, "end_time": "2021-01-05T22:03:11.568808", "exception": false, "start_time": "2021-01-05T22:03:11.531333", "status": "completed"} tags=[]
display(len(CLUSTERERS))

# %% papermill={"duration": 0.038819, "end_time": "2021-01-05T22:03:11.629807", "exception": false, "start_time": "2021-01-05T22:03:11.590988", "status": "completed"} tags=[]
_iter = iter(CLUSTERERS.items())
display(next(_iter))
display(next(_iter))

# %% papermill={"duration": 0.037293, "end_time": "2021-01-05T22:03:11.690518", "exception": false, "start_time": "2021-01-05T22:03:11.653225", "status": "completed"} tags=[]
clustering_method_name = method_name
display(clustering_method_name)

# %% [markdown] papermill={"duration": 0.022785, "end_time": "2021-01-05T22:03:11.737064", "exception": false, "start_time": "2021-01-05T22:03:11.714279", "status": "completed"} tags=[]
# ## Output directory

# %% papermill={"duration": 0.038391, "end_time": "2021-01-05T22:03:11.797951", "exception": false, "start_time": "2021-01-05T22:03:11.759560", "status": "completed"} tags=[]
# output dir for this notebook
RESULTS_DIR = Path(
    conf.RESULTS["CLUSTERING_RUNS_DIR"],
    f"{INPUT_SUBSET}-{INPUT_STEM}",
).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %% [markdown] papermill={"duration": 0.022808, "end_time": "2021-01-05T22:03:11.844684", "exception": false, "start_time": "2021-01-05T22:03:11.821876", "status": "completed"} tags=[]
# # Load input file

# %% papermill={"duration": 0.050307, "end_time": "2021-01-05T22:03:11.918102", "exception": false, "start_time": "2021-01-05T22:03:11.867795", "status": "completed"} tags=[]
data = pd.read_pickle(input_filepath)

# %% papermill={"duration": 0.037222, "end_time": "2021-01-05T22:03:11.980013", "exception": false, "start_time": "2021-01-05T22:03:11.942791", "status": "completed"} tags=[]
data.shape

# %% papermill={"duration": 0.051039, "end_time": "2021-01-05T22:03:12.054671", "exception": false, "start_time": "2021-01-05T22:03:12.003632", "status": "completed"} tags=[]
data.head()

# %% papermill={"duration": 0.043169, "end_time": "2021-01-05T22:03:12.123371", "exception": false, "start_time": "2021-01-05T22:03:12.080202", "status": "completed"} tags=[]
assert not data.isna().any().any()

# %% [markdown] papermill={"duration": 0.023563, "end_time": "2021-01-05T22:03:12.173021", "exception": false, "start_time": "2021-01-05T22:03:12.149458", "status": "completed"} tags=[]
# # Clustering

# %% [markdown] papermill={"duration": 0.023357, "end_time": "2021-01-05T22:03:12.219362", "exception": false, "start_time": "2021-01-05T22:03:12.196005", "status": "completed"} tags=[]
# ## Generate ensemble

# %% papermill={"duration": 0.040204, "end_time": "2021-01-05T22:03:12.282717", "exception": false, "start_time": "2021-01-05T22:03:12.242513", "status": "completed"} tags=[]
from clustering.ensemble import generate_ensemble

# %% papermill={"duration": 41282.067934, "end_time": "2021-01-06T09:31:14.375095", "exception": false, "start_time": "2021-01-05T22:03:12.307161", "status": "completed"} tags=[]
ensemble = generate_ensemble(
    data,
    CLUSTERERS,
    attributes=CLUSTERING_ATTRIBUTES_TO_SAVE,
)

# %% papermill={"duration": 0.086748, "end_time": "2021-01-06T09:31:15.366928", "exception": false, "start_time": "2021-01-06T09:31:15.280180", "status": "completed"} tags=[]
# the number should be close to 295 (the number of partitions generated by k-means/spectral clustering)
ensemble.shape

# %% papermill={"duration": 0.093504, "end_time": "2021-01-06T09:31:15.532912", "exception": false, "start_time": "2021-01-06T09:31:15.439408", "status": "completed"} tags=[]
ensemble.head()

# %% papermill={"duration": 0.086733, "end_time": "2021-01-06T09:31:15.691378", "exception": false, "start_time": "2021-01-06T09:31:15.604645", "status": "completed"} tags=[]
ensemble["n_clusters"].value_counts().head()

# %% papermill={"duration": 0.087328, "end_time": "2021-01-06T09:31:15.850246", "exception": false, "start_time": "2021-01-06T09:31:15.762918", "status": "completed"} tags=[]
ensemble_stats = ensemble["n_clusters"].describe()
display(ensemble_stats)

# %% [markdown] papermill={"duration": 0.071989, "end_time": "2021-01-06T09:31:15.995280", "exception": false, "start_time": "2021-01-06T09:31:15.923291", "status": "completed"} tags=[]
# ## Testing

# %% papermill={"duration": 0.085448, "end_time": "2021-01-06T09:31:16.151583", "exception": false, "start_time": "2021-01-06T09:31:16.066135", "status": "completed"} tags=[]
assert ensemble_stats["min"] > 1

# %% papermill={"duration": 0.088665, "end_time": "2021-01-06T09:31:16.312039", "exception": false, "start_time": "2021-01-06T09:31:16.223374", "status": "completed"} tags=[]
assert not ensemble["n_clusters"].isna().any()

# %% papermill={"duration": 0.085698, "end_time": "2021-01-06T09:31:16.470859", "exception": false, "start_time": "2021-01-06T09:31:16.385161", "status": "completed"} tags=[]
assert ensemble.shape[0] == len(CLUSTERERS)

# %% papermill={"duration": 0.105792, "end_time": "2021-01-06T09:31:16.649305", "exception": false, "start_time": "2021-01-06T09:31:16.543513", "status": "completed"} tags=[]
# all partitions have the right size
assert np.all(
    [part["partition"].shape[0] == data.shape[0] for idx, part in ensemble.iterrows()]
)

# %% papermill={"duration": 0.107449, "end_time": "2021-01-06T09:31:16.830088", "exception": false, "start_time": "2021-01-06T09:31:16.722639", "status": "completed"} tags=[]
# no partition has negative clusters (noisy points)
assert not np.any([(part["partition"] < 0).any() for idx, part in ensemble.iterrows()])

# %% [markdown] papermill={"duration": 0.072449, "end_time": "2021-01-06T09:31:16.975220", "exception": false, "start_time": "2021-01-06T09:31:16.902771", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.087021, "end_time": "2021-01-06T09:31:17.132789", "exception": false, "start_time": "2021-01-06T09:31:17.045768", "status": "completed"} tags=[]
output_filename = Path(
    RESULTS_DIR,
    generate_result_set_name(
        CLUSTERING_OPTIONS,
        prefix=f"{clustering_method_name}-",
        suffix=".pkl",
    ),
).resolve()
display(output_filename)

# %% papermill={"duration": 0.097965, "end_time": "2021-01-06T09:31:17.306433", "exception": false, "start_time": "2021-01-06T09:31:17.208468", "status": "completed"} tags=[]
ensemble.to_pickle(output_filename)

# %% [markdown] papermill={"duration": 0.071046, "end_time": "2021-01-06T09:31:17.451751", "exception": false, "start_time": "2021-01-06T09:31:17.380705", "status": "completed"} tags=[]
# # Stability

# %% [markdown] papermill={"duration": 0.071456, "end_time": "2021-01-06T09:31:17.593952", "exception": false, "start_time": "2021-01-06T09:31:17.522496", "status": "completed"} tags=[]
# ## Group ensemble by n_clusters

# %% papermill={"duration": 0.099756, "end_time": "2021-01-06T09:31:17.765036", "exception": false, "start_time": "2021-01-06T09:31:17.665280", "status": "completed"} tags=[]
parts = ensemble.groupby("n_clusters").apply(
    lambda x: np.concatenate(x["partition"].apply(lambda x: x.reshape(1, -1)), axis=0)
)

# %% papermill={"duration": 0.097357, "end_time": "2021-01-06T09:31:17.933242", "exception": false, "start_time": "2021-01-06T09:31:17.835885", "status": "completed"} tags=[]
parts.head()

# %% papermill={"duration": 0.085896, "end_time": "2021-01-06T09:31:18.090880", "exception": false, "start_time": "2021-01-06T09:31:18.004984", "status": "completed"} tags=[]
assert np.all(
    [
        parts.loc[k].shape == (CLUSTERING_OPTIONS["N_REPS_PER_K"], data.shape[0])
        for k in parts.index
    ]
)

# %% [markdown] papermill={"duration": 0.072165, "end_time": "2021-01-06T09:31:18.236232", "exception": false, "start_time": "2021-01-06T09:31:18.164067", "status": "completed"} tags=[]
# ## Compute stability

# %% papermill={"duration": 0.086434, "end_time": "2021-01-06T09:31:18.397495", "exception": false, "start_time": "2021-01-06T09:31:18.311061", "status": "completed"} tags=[]
from sklearn.metrics import adjusted_rand_score as ari
from scipy.spatial.distance import squareform, pdist

# %% papermill={"duration": 0.810572, "end_time": "2021-01-06T09:31:19.281094", "exception": false, "start_time": "2021-01-06T09:31:18.470522", "status": "completed"} tags=[]
parts_ari = pd.Series(
    {k: pdist(parts.loc[k], metric=ari) for k in parts.index}, name="k"
)

# %% papermill={"duration": 0.088828, "end_time": "2021-01-06T09:31:19.443582", "exception": false, "start_time": "2021-01-06T09:31:19.354754", "status": "completed"} tags=[]
parts_ari_stability = parts_ari.apply(lambda x: x.mean())
display(parts_ari_stability.sort_values(ascending=False).head(15))

# %% papermill={"duration": 0.087648, "end_time": "2021-01-06T09:31:19.604598", "exception": false, "start_time": "2021-01-06T09:31:19.516950", "status": "completed"} tags=[]
parts_ari_df = pd.DataFrame.from_records(parts_ari.tolist()).set_index(
    parts_ari.index.copy()
)

# %% papermill={"duration": 0.086964, "end_time": "2021-01-06T09:31:19.765057", "exception": false, "start_time": "2021-01-06T09:31:19.678093", "status": "completed"} tags=[]
parts_ari_df.shape

# %% papermill={"duration": 0.086904, "end_time": "2021-01-06T09:31:19.926422", "exception": false, "start_time": "2021-01-06T09:31:19.839518", "status": "completed"} tags=[]
assert (
    int(
        (CLUSTERING_OPTIONS["N_REPS_PER_K"] * (CLUSTERING_OPTIONS["N_REPS_PER_K"] - 1))
        / 2
    )
    == parts_ari_df.shape[1]
)

# %% papermill={"duration": 0.092258, "end_time": "2021-01-06T09:31:20.090931", "exception": false, "start_time": "2021-01-06T09:31:19.998673", "status": "completed"} tags=[]
parts_ari_df.head()

# %% [markdown] papermill={"duration": 0.07118, "end_time": "2021-01-06T09:31:20.234589", "exception": false, "start_time": "2021-01-06T09:31:20.163409", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.088677, "end_time": "2021-01-06T09:31:20.397668", "exception": false, "start_time": "2021-01-06T09:31:20.308991", "status": "completed"} tags=[]
output_filename = Path(
    RESULTS_DIR,
    generate_result_set_name(
        CLUSTERING_OPTIONS,
        prefix=f"{clustering_method_name}-stability-",
        suffix=".pkl",
    ),
).resolve()
display(output_filename)

# %% papermill={"duration": 0.088684, "end_time": "2021-01-06T09:31:20.560478", "exception": false, "start_time": "2021-01-06T09:31:20.471794", "status": "completed"} tags=[]
parts_ari_df.to_pickle(output_filename)

# %% [markdown] papermill={"duration": 0.072832, "end_time": "2021-01-06T09:31:20.707107", "exception": false, "start_time": "2021-01-06T09:31:20.634275", "status": "completed"} tags=[]
# ## Stability plot

# %% papermill={"duration": 0.088935, "end_time": "2021-01-06T09:31:20.868569", "exception": false, "start_time": "2021-01-06T09:31:20.779634", "status": "completed"} tags=[]
parts_ari_df_plot = (
    parts_ari_df.stack()
    .reset_index()
    .rename(columns={"level_0": "k", "level_1": "idx", 0: "ari"})
)

# %% papermill={"duration": 0.089732, "end_time": "2021-01-06T09:31:21.033877", "exception": false, "start_time": "2021-01-06T09:31:20.944145", "status": "completed"} tags=[]
parts_ari_df_plot.dtypes

# %% papermill={"duration": 0.090277, "end_time": "2021-01-06T09:31:21.198847", "exception": false, "start_time": "2021-01-06T09:31:21.108570", "status": "completed"} tags=[]
parts_ari_df_plot.head()

# %% papermill={"duration": 2.967032, "end_time": "2021-01-06T09:31:24.239785", "exception": false, "start_time": "2021-01-06T09:31:21.272753", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.075805, "end_time": "2021-01-06T09:31:24.393025", "exception": false, "start_time": "2021-01-06T09:31:24.317220", "status": "completed"} tags=[]
