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

# %% [markdown] papermill={"duration": 0.053307, "end_time": "2020-12-02T02:27:47.607550", "exception": false, "start_time": "2020-12-02T02:27:47.554243", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.017665, "end_time": "2020-12-02T02:27:47.644276", "exception": false, "start_time": "2020-12-02T02:27:47.626611", "status": "completed"} tags=[]
# Runs gaussian mixture model on the z_score_std version of the data.

# %% [markdown] papermill={"duration": 0.017428, "end_time": "2020-12-02T02:27:47.679496", "exception": false, "start_time": "2020-12-02T02:27:47.662068", "status": "completed"} tags=[]
# # Environment variables

# %% papermill={"duration": 0.03147, "end_time": "2020-12-02T02:27:47.728584", "exception": false, "start_time": "2020-12-02T02:27:47.697114", "status": "completed"} tags=[]
from IPython.display import display

import conf

N_JOBS = conf.GENERAL["N_JOBS"]
display(N_JOBS)

# %% papermill={"duration": 0.024959, "end_time": "2020-12-02T02:27:47.772360", "exception": false, "start_time": "2020-12-02T02:27:47.747401", "status": "completed"} tags=[]
# %env MKL_NUM_THREADS=$N_JOBS
# %env OPEN_BLAS_NUM_THREADS=$N_JOBS
# %env NUMEXPR_NUM_THREADS=$N_JOBS
# %env OMP_NUM_THREADS=$N_JOBS

# %% [markdown] papermill={"duration": 0.018136, "end_time": "2020-12-02T02:27:47.809364", "exception": false, "start_time": "2020-12-02T02:27:47.791228", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.028833, "end_time": "2020-12-02T02:27:47.855956", "exception": false, "start_time": "2020-12-02T02:27:47.827123", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 1.610963, "end_time": "2020-12-02T02:27:49.485359", "exception": false, "start_time": "2020-12-02T02:27:47.874396", "status": "completed"} tags=[]
from pathlib import Path

import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns

from utils import generate_result_set_name

# %% [markdown] papermill={"duration": 0.01801, "end_time": "2020-12-02T02:27:49.523815", "exception": false, "start_time": "2020-12-02T02:27:49.505805", "status": "completed"} tags=[]
# # Settings

# %% papermill={"duration": 0.032012, "end_time": "2020-12-02T02:27:49.573632", "exception": false, "start_time": "2020-12-02T02:27:49.541620", "status": "completed"} tags=[]
INITIAL_RANDOM_STATE = 60000

# %% [markdown] papermill={"duration": 0.018201, "end_time": "2020-12-02T02:27:49.610340", "exception": false, "start_time": "2020-12-02T02:27:49.592139", "status": "completed"} tags=[]
# ## Input data

# %% papermill={"duration": 0.033005, "end_time": "2020-12-02T02:27:49.661562", "exception": false, "start_time": "2020-12-02T02:27:49.628557", "status": "completed"} tags=[]
INPUT_SUBSET = "z_score_std"

# %% papermill={"duration": 0.033359, "end_time": "2020-12-02T02:27:49.714006", "exception": false, "start_time": "2020-12-02T02:27:49.680647", "status": "completed"} tags=[]
INPUT_STEM = "projection-smultixcan-efo_partial-mashr-zscores"

# %% papermill={"duration": 0.034466, "end_time": "2020-12-02T02:27:49.767394", "exception": false, "start_time": "2020-12-02T02:27:49.732928", "status": "completed"} tags=[]
input_filepath = Path(
    conf.RESULTS["DATA_TRANSFORMATIONS_DIR"],
    INPUT_SUBSET,
    f"{INPUT_SUBSET}-{INPUT_STEM}.pkl",
).resolve()
display(input_filepath)

assert input_filepath.exists(), "Input file does not exist"

input_filepath_stem = input_filepath.stem
display(input_filepath_stem)

# %% [markdown] papermill={"duration": 0.019173, "end_time": "2020-12-02T02:27:49.806333", "exception": false, "start_time": "2020-12-02T02:27:49.787160", "status": "completed"} tags=[]
# ## Clustering

# %% papermill={"duration": 0.040477, "end_time": "2020-12-02T02:27:49.865520", "exception": false, "start_time": "2020-12-02T02:27:49.825043", "status": "completed"} tags=[]
from sklearn.mixture import GaussianMixture

# %% papermill={"duration": 0.032796, "end_time": "2020-12-02T02:27:49.917601", "exception": false, "start_time": "2020-12-02T02:27:49.884805", "status": "completed"} tags=[]
CLUSTERING_ATTRIBUTES_TO_SAVE = ["n_clusters"]

# %% papermill={"duration": 0.035853, "end_time": "2020-12-02T02:27:49.973015", "exception": false, "start_time": "2020-12-02T02:27:49.937162", "status": "completed"} tags=[]
CLUSTERING_OPTIONS = {}

CLUSTERING_OPTIONS["K_MIN"] = 2
CLUSTERING_OPTIONS["K_MAX"] = 60  # sqrt(3749)
CLUSTERING_OPTIONS["N_REPS_PER_K"] = 5
CLUSTERING_OPTIONS["N_INIT"] = 10
CLUSTERING_OPTIONS["COVARIANCE_TYPE"] = "full"

display(CLUSTERING_OPTIONS)

# %% papermill={"duration": 0.036604, "end_time": "2020-12-02T02:27:50.029936", "exception": false, "start_time": "2020-12-02T02:27:49.993332", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.034574, "end_time": "2020-12-02T02:27:50.084844", "exception": false, "start_time": "2020-12-02T02:27:50.050270", "status": "completed"} tags=[]
display(len(CLUSTERERS))

# %% papermill={"duration": 0.03647, "end_time": "2020-12-02T02:27:50.141578", "exception": false, "start_time": "2020-12-02T02:27:50.105108", "status": "completed"} tags=[]
_iter = iter(CLUSTERERS.items())
display(next(_iter))
display(next(_iter))

# %% papermill={"duration": 0.035352, "end_time": "2020-12-02T02:27:50.198116", "exception": false, "start_time": "2020-12-02T02:27:50.162764", "status": "completed"} tags=[]
clustering_method_name = method_name
display(clustering_method_name)

# %% [markdown] papermill={"duration": 0.020068, "end_time": "2020-12-02T02:27:50.238642", "exception": false, "start_time": "2020-12-02T02:27:50.218574", "status": "completed"} tags=[]
# ## Output directory

# %% papermill={"duration": 0.035526, "end_time": "2020-12-02T02:27:50.294503", "exception": false, "start_time": "2020-12-02T02:27:50.258977", "status": "completed"} tags=[]
# output dir for this notebook
RESULTS_DIR = Path(
    conf.RESULTS["CLUSTERING_RUNS_DIR"],
    f"{INPUT_SUBSET}-{INPUT_STEM}",
).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %% [markdown] papermill={"duration": 0.020204, "end_time": "2020-12-02T02:27:50.335547", "exception": false, "start_time": "2020-12-02T02:27:50.315343", "status": "completed"} tags=[]
# # Load input file

# %% papermill={"duration": 0.046387, "end_time": "2020-12-02T02:27:50.402599", "exception": false, "start_time": "2020-12-02T02:27:50.356212", "status": "completed"} tags=[]
data = pd.read_pickle(input_filepath)

# %% papermill={"duration": 0.037154, "end_time": "2020-12-02T02:27:50.462604", "exception": false, "start_time": "2020-12-02T02:27:50.425450", "status": "completed"} tags=[]
data.shape

# %% papermill={"duration": 0.047893, "end_time": "2020-12-02T02:27:50.532384", "exception": false, "start_time": "2020-12-02T02:27:50.484491", "status": "completed"} tags=[]
data.head()

# %% papermill={"duration": 0.040685, "end_time": "2020-12-02T02:27:50.595816", "exception": false, "start_time": "2020-12-02T02:27:50.555131", "status": "completed"} tags=[]
assert not data.isna().any().any()

# %% [markdown] papermill={"duration": 0.021002, "end_time": "2020-12-02T02:27:50.639033", "exception": false, "start_time": "2020-12-02T02:27:50.618031", "status": "completed"} tags=[]
# # Clustering

# %% [markdown] papermill={"duration": 0.021179, "end_time": "2020-12-02T02:27:50.681718", "exception": false, "start_time": "2020-12-02T02:27:50.660539", "status": "completed"} tags=[]
# ## Generate ensemble

# %% papermill={"duration": 0.038034, "end_time": "2020-12-02T02:27:50.741007", "exception": false, "start_time": "2020-12-02T02:27:50.702973", "status": "completed"} tags=[]
from clustering.ensemble import generate_ensemble

# %% papermill={"duration": 41270.177501, "end_time": "2020-12-02T13:55:40.940480", "exception": false, "start_time": "2020-12-02T02:27:50.762979", "status": "completed"} tags=[]
ensemble = generate_ensemble(
    data,
    CLUSTERERS,
    attributes=CLUSTERING_ATTRIBUTES_TO_SAVE,
)

# %% papermill={"duration": 0.087439, "end_time": "2020-12-02T13:55:41.939287", "exception": false, "start_time": "2020-12-02T13:55:41.851848", "status": "completed"} tags=[]
# the number should be close to 295 (the number of partitions generated by k-means/spectral clustering)
ensemble.shape

# %% papermill={"duration": 0.091427, "end_time": "2020-12-02T13:55:42.100943", "exception": false, "start_time": "2020-12-02T13:55:42.009516", "status": "completed"} tags=[]
ensemble.head()

# %% papermill={"duration": 0.08535, "end_time": "2020-12-02T13:55:42.256126", "exception": false, "start_time": "2020-12-02T13:55:42.170776", "status": "completed"} tags=[]
ensemble["n_clusters"].value_counts().head()

# %% papermill={"duration": 0.090212, "end_time": "2020-12-02T13:55:42.420584", "exception": false, "start_time": "2020-12-02T13:55:42.330372", "status": "completed"} tags=[]
ensemble_stats = ensemble["n_clusters"].describe()
display(ensemble_stats)

# %% [markdown] papermill={"duration": 0.070265, "end_time": "2020-12-02T13:55:42.562859", "exception": false, "start_time": "2020-12-02T13:55:42.492594", "status": "completed"} tags=[]
# ## Testing

# %% papermill={"duration": 0.084092, "end_time": "2020-12-02T13:55:42.716283", "exception": false, "start_time": "2020-12-02T13:55:42.632191", "status": "completed"} tags=[]
assert ensemble_stats["min"] > 1

# %% papermill={"duration": 0.083787, "end_time": "2020-12-02T13:55:42.870643", "exception": false, "start_time": "2020-12-02T13:55:42.786856", "status": "completed"} tags=[]
assert not ensemble["n_clusters"].isna().any()

# %% papermill={"duration": 0.084027, "end_time": "2020-12-02T13:55:43.025014", "exception": false, "start_time": "2020-12-02T13:55:42.940987", "status": "completed"} tags=[]
assert ensemble.shape[0] == len(CLUSTERERS)

# %% papermill={"duration": 0.102245, "end_time": "2020-12-02T13:55:43.197943", "exception": false, "start_time": "2020-12-02T13:55:43.095698", "status": "completed"} tags=[]
# all partitions have the right size
assert np.all(
    [part["partition"].shape[0] == data.shape[0] for idx, part in ensemble.iterrows()]
)

# %% papermill={"duration": 0.108861, "end_time": "2020-12-02T13:55:43.378381", "exception": false, "start_time": "2020-12-02T13:55:43.269520", "status": "completed"} tags=[]
# no partition has negative clusters (noisy points)
assert not np.any([(part["partition"] < 0).any() for idx, part in ensemble.iterrows()])

# %% [markdown] papermill={"duration": 0.070396, "end_time": "2020-12-02T13:55:43.520155", "exception": false, "start_time": "2020-12-02T13:55:43.449759", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.085942, "end_time": "2020-12-02T13:55:43.676563", "exception": false, "start_time": "2020-12-02T13:55:43.590621", "status": "completed"} tags=[]
output_filename = Path(
    RESULTS_DIR,
    generate_result_set_name(
        CLUSTERING_OPTIONS,
        prefix=f"{clustering_method_name}-",
        suffix=".pkl",
    ),
).resolve()
display(output_filename)

# %% papermill={"duration": 0.09361, "end_time": "2020-12-02T13:55:43.840922", "exception": false, "start_time": "2020-12-02T13:55:43.747312", "status": "completed"} tags=[]
ensemble.to_pickle(output_filename)

# %% [markdown] papermill={"duration": 0.070596, "end_time": "2020-12-02T13:55:43.982795", "exception": false, "start_time": "2020-12-02T13:55:43.912199", "status": "completed"} tags=[]
# # Stability

# %% [markdown] papermill={"duration": 0.06996, "end_time": "2020-12-02T13:55:44.124169", "exception": false, "start_time": "2020-12-02T13:55:44.054209", "status": "completed"} tags=[]
# ## Group ensemble by n_clusters

# %% papermill={"duration": 0.100185, "end_time": "2020-12-02T13:55:44.294620", "exception": false, "start_time": "2020-12-02T13:55:44.194435", "status": "completed"} tags=[]
parts = ensemble.groupby("n_clusters").apply(
    lambda x: np.concatenate(x["partition"].apply(lambda x: x.reshape(1, -1)), axis=0)
)

# %% papermill={"duration": 0.098205, "end_time": "2020-12-02T13:55:44.466683", "exception": false, "start_time": "2020-12-02T13:55:44.368478", "status": "completed"} tags=[]
parts.head()

# %% papermill={"duration": 0.086819, "end_time": "2020-12-02T13:55:44.625919", "exception": false, "start_time": "2020-12-02T13:55:44.539100", "status": "completed"} tags=[]
assert np.all(
    [
        parts.loc[k].shape == (CLUSTERING_OPTIONS["N_REPS_PER_K"], data.shape[0])
        for k in parts.index
    ]
)

# %% [markdown] papermill={"duration": 0.070877, "end_time": "2020-12-02T13:55:44.769137", "exception": false, "start_time": "2020-12-02T13:55:44.698260", "status": "completed"} tags=[]
# ## Compute stability

# %% papermill={"duration": 0.084557, "end_time": "2020-12-02T13:55:44.924778", "exception": false, "start_time": "2020-12-02T13:55:44.840221", "status": "completed"} tags=[]
from sklearn.metrics import adjusted_rand_score as ari
from scipy.spatial.distance import squareform, pdist

# %% papermill={"duration": 0.805153, "end_time": "2020-12-02T13:55:45.801512", "exception": false, "start_time": "2020-12-02T13:55:44.996359", "status": "completed"} tags=[]
parts_ari = pd.Series(
    {k: pdist(parts.loc[k], metric=ari) for k in parts.index}, name="k"
)

# %% papermill={"duration": 0.088438, "end_time": "2020-12-02T13:55:45.961454", "exception": false, "start_time": "2020-12-02T13:55:45.873016", "status": "completed"} tags=[]
parts_ari_stability = parts_ari.apply(lambda x: x.mean())
display(parts_ari_stability.sort_values(ascending=False).head(15))

# %% papermill={"duration": 0.087466, "end_time": "2020-12-02T13:55:46.121111", "exception": false, "start_time": "2020-12-02T13:55:46.033645", "status": "completed"} tags=[]
parts_ari_df = pd.DataFrame.from_records(parts_ari.tolist()).set_index(
    parts_ari.index.copy()
)

# %% papermill={"duration": 0.087254, "end_time": "2020-12-02T13:55:46.282002", "exception": false, "start_time": "2020-12-02T13:55:46.194748", "status": "completed"} tags=[]
parts_ari_df.shape

# %% papermill={"duration": 0.087864, "end_time": "2020-12-02T13:55:46.443874", "exception": false, "start_time": "2020-12-02T13:55:46.356010", "status": "completed"} tags=[]
assert (
    int(
        (CLUSTERING_OPTIONS["N_REPS_PER_K"] * (CLUSTERING_OPTIONS["N_REPS_PER_K"] - 1))
        / 2
    )
    == parts_ari_df.shape[1]
)

# %% papermill={"duration": 0.092687, "end_time": "2020-12-02T13:55:46.610154", "exception": false, "start_time": "2020-12-02T13:55:46.517467", "status": "completed"} tags=[]
parts_ari_df.head()

# %% [markdown] papermill={"duration": 0.072072, "end_time": "2020-12-02T13:55:46.754900", "exception": false, "start_time": "2020-12-02T13:55:46.682828", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.087445, "end_time": "2020-12-02T13:55:46.914019", "exception": false, "start_time": "2020-12-02T13:55:46.826574", "status": "completed"} tags=[]
output_filename = Path(
    RESULTS_DIR,
    generate_result_set_name(
        CLUSTERING_OPTIONS,
        prefix=f"{clustering_method_name}-stability-",
        suffix=".pkl",
    ),
).resolve()
display(output_filename)

# %% papermill={"duration": 0.086956, "end_time": "2020-12-02T13:55:47.073713", "exception": false, "start_time": "2020-12-02T13:55:46.986757", "status": "completed"} tags=[]
parts_ari_df.to_pickle(output_filename)

# %% [markdown] papermill={"duration": 0.072293, "end_time": "2020-12-02T13:55:47.218958", "exception": false, "start_time": "2020-12-02T13:55:47.146665", "status": "completed"} tags=[]
# ## Stability plot

# %% papermill={"duration": 0.091349, "end_time": "2020-12-02T13:55:47.382883", "exception": false, "start_time": "2020-12-02T13:55:47.291534", "status": "completed"} tags=[]
parts_ari_df_plot = (
    parts_ari_df.stack()
    .reset_index()
    .rename(columns={"level_0": "k", "level_1": "idx", 0: "ari"})
)

# %% papermill={"duration": 0.09034, "end_time": "2020-12-02T13:55:47.547767", "exception": false, "start_time": "2020-12-02T13:55:47.457427", "status": "completed"} tags=[]
parts_ari_df_plot.dtypes

# %% papermill={"duration": 0.091469, "end_time": "2020-12-02T13:55:47.713848", "exception": false, "start_time": "2020-12-02T13:55:47.622379", "status": "completed"} tags=[]
parts_ari_df_plot.head()

# %% papermill={"duration": 2.967621, "end_time": "2020-12-02T13:55:50.758028", "exception": false, "start_time": "2020-12-02T13:55:47.790407", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.074592, "end_time": "2020-12-02T13:55:50.908389", "exception": false, "start_time": "2020-12-02T13:55:50.833797", "status": "completed"} tags=[]
