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

# %% [markdown] papermill={"duration": 0.025246, "end_time": "2021-01-06T09:31:26.892002", "exception": false, "start_time": "2021-01-06T09:31:26.866756", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.020704, "end_time": "2021-01-06T09:31:26.933548", "exception": false, "start_time": "2021-01-06T09:31:26.912844", "status": "completed"} tags=[]
# Runs gaussian mixture model on the pca version of the data.

# %% [markdown] papermill={"duration": 0.020947, "end_time": "2021-01-06T09:31:26.975623", "exception": false, "start_time": "2021-01-06T09:31:26.954676", "status": "completed"} tags=[]
# # Environment variables

# %% papermill={"duration": 0.034553, "end_time": "2021-01-06T09:31:27.031282", "exception": false, "start_time": "2021-01-06T09:31:26.996729", "status": "completed"} tags=[]
from IPython.display import display

import conf

N_JOBS = conf.GENERAL["N_JOBS"]
display(N_JOBS)

# %% papermill={"duration": 0.029533, "end_time": "2021-01-06T09:31:27.082966", "exception": false, "start_time": "2021-01-06T09:31:27.053433", "status": "completed"} tags=[]
# %env MKL_NUM_THREADS=$N_JOBS
# %env OPEN_BLAS_NUM_THREADS=$N_JOBS
# %env NUMEXPR_NUM_THREADS=$N_JOBS
# %env OMP_NUM_THREADS=$N_JOBS

# %% [markdown] papermill={"duration": 0.022598, "end_time": "2021-01-06T09:31:27.128480", "exception": false, "start_time": "2021-01-06T09:31:27.105882", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.033212, "end_time": "2021-01-06T09:31:27.183521", "exception": false, "start_time": "2021-01-06T09:31:27.150309", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 1.642174, "end_time": "2021-01-06T09:31:28.847895", "exception": false, "start_time": "2021-01-06T09:31:27.205721", "status": "completed"} tags=[]
from pathlib import Path

import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns

from utils import generate_result_set_name

# %% [markdown] papermill={"duration": 0.021354, "end_time": "2021-01-06T09:31:28.893364", "exception": false, "start_time": "2021-01-06T09:31:28.872010", "status": "completed"} tags=[]
# # Settings

# %% papermill={"duration": 0.03574, "end_time": "2021-01-06T09:31:28.950430", "exception": false, "start_time": "2021-01-06T09:31:28.914690", "status": "completed"} tags=[]
INITIAL_RANDOM_STATE = 70000

# %% [markdown] papermill={"duration": 0.021522, "end_time": "2021-01-06T09:31:28.993777", "exception": false, "start_time": "2021-01-06T09:31:28.972255", "status": "completed"} tags=[]
# ## Input data

# %% papermill={"duration": 0.035321, "end_time": "2021-01-06T09:31:29.050130", "exception": false, "start_time": "2021-01-06T09:31:29.014809", "status": "completed"} tags=[]
INPUT_SUBSET = "pca"

# %% papermill={"duration": 0.035507, "end_time": "2021-01-06T09:31:29.107532", "exception": false, "start_time": "2021-01-06T09:31:29.072025", "status": "completed"} tags=[]
INPUT_STEM = "z_score_std-projection-smultixcan-efo_partial-mashr-zscores"

# %% papermill={"duration": 0.034845, "end_time": "2021-01-06T09:31:29.164190", "exception": false, "start_time": "2021-01-06T09:31:29.129345", "status": "completed"} tags=[]
DR_OPTIONS = {
    "n_components": 50,
    "svd_solver": "full",
    "random_state": 0,
}

# %% papermill={"duration": 0.038455, "end_time": "2021-01-06T09:31:29.224780", "exception": false, "start_time": "2021-01-06T09:31:29.186325", "status": "completed"} tags=[]
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

# %% [markdown] papermill={"duration": 0.022005, "end_time": "2021-01-06T09:31:29.269599", "exception": false, "start_time": "2021-01-06T09:31:29.247594", "status": "completed"} tags=[]
# ## Clustering

# %% papermill={"duration": 0.044819, "end_time": "2021-01-06T09:31:29.336246", "exception": false, "start_time": "2021-01-06T09:31:29.291427", "status": "completed"} tags=[]
from sklearn.mixture import GaussianMixture

# %% papermill={"duration": 0.036417, "end_time": "2021-01-06T09:31:29.396778", "exception": false, "start_time": "2021-01-06T09:31:29.360361", "status": "completed"} tags=[]
CLUSTERING_ATTRIBUTES_TO_SAVE = ["n_clusters"]

# %% papermill={"duration": 0.037713, "end_time": "2021-01-06T09:31:29.457259", "exception": false, "start_time": "2021-01-06T09:31:29.419546", "status": "completed"} tags=[]
CLUSTERING_OPTIONS = {}

CLUSTERING_OPTIONS["K_MIN"] = 2
CLUSTERING_OPTIONS["K_MAX"] = 60  # sqrt(3749)
CLUSTERING_OPTIONS["N_REPS_PER_K"] = 5
CLUSTERING_OPTIONS["N_INIT"] = 10
CLUSTERING_OPTIONS["COVARIANCE_TYPE"] = "full"

display(CLUSTERING_OPTIONS)

# %% papermill={"duration": 0.038805, "end_time": "2021-01-06T09:31:29.519962", "exception": false, "start_time": "2021-01-06T09:31:29.481157", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.037006, "end_time": "2021-01-06T09:31:29.579439", "exception": false, "start_time": "2021-01-06T09:31:29.542433", "status": "completed"} tags=[]
display(len(CLUSTERERS))

# %% papermill={"duration": 0.039611, "end_time": "2021-01-06T09:31:29.643017", "exception": false, "start_time": "2021-01-06T09:31:29.603406", "status": "completed"} tags=[]
_iter = iter(CLUSTERERS.items())
display(next(_iter))
display(next(_iter))

# %% papermill={"duration": 0.03762, "end_time": "2021-01-06T09:31:29.704190", "exception": false, "start_time": "2021-01-06T09:31:29.666570", "status": "completed"} tags=[]
clustering_method_name = method_name
display(clustering_method_name)

# %% [markdown] papermill={"duration": 0.02283, "end_time": "2021-01-06T09:31:29.750360", "exception": false, "start_time": "2021-01-06T09:31:29.727530", "status": "completed"} tags=[]
# ## Output directory

# %% papermill={"duration": 0.037892, "end_time": "2021-01-06T09:31:29.811352", "exception": false, "start_time": "2021-01-06T09:31:29.773460", "status": "completed"} tags=[]
# output dir for this notebook
RESULTS_DIR = Path(
    conf.RESULTS["CLUSTERING_RUNS_DIR"],
    f"{INPUT_SUBSET}-{INPUT_STEM}",
).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %% [markdown] papermill={"duration": 0.023695, "end_time": "2021-01-06T09:31:29.859296", "exception": false, "start_time": "2021-01-06T09:31:29.835601", "status": "completed"} tags=[]
# # Load input file

# %% papermill={"duration": 0.03891, "end_time": "2021-01-06T09:31:29.921593", "exception": false, "start_time": "2021-01-06T09:31:29.882683", "status": "completed"} tags=[]
data = pd.read_pickle(input_filepath)

# %% papermill={"duration": 0.038229, "end_time": "2021-01-06T09:31:29.984023", "exception": false, "start_time": "2021-01-06T09:31:29.945794", "status": "completed"} tags=[]
data.shape

# %% papermill={"duration": 0.050481, "end_time": "2021-01-06T09:31:30.057918", "exception": false, "start_time": "2021-01-06T09:31:30.007437", "status": "completed"} tags=[]
data.head()

# %% papermill={"duration": 0.038975, "end_time": "2021-01-06T09:31:30.121798", "exception": false, "start_time": "2021-01-06T09:31:30.082823", "status": "completed"} tags=[]
assert not data.isna().any().any()

# %% [markdown] papermill={"duration": 0.023451, "end_time": "2021-01-06T09:31:30.169787", "exception": false, "start_time": "2021-01-06T09:31:30.146336", "status": "completed"} tags=[]
# # Clustering

# %% [markdown] papermill={"duration": 0.023252, "end_time": "2021-01-06T09:31:30.216372", "exception": false, "start_time": "2021-01-06T09:31:30.193120", "status": "completed"} tags=[]
# ## Generate ensemble

# %% papermill={"duration": 0.040136, "end_time": "2021-01-06T09:31:30.280344", "exception": false, "start_time": "2021-01-06T09:31:30.240208", "status": "completed"} tags=[]
from clustering.ensemble import generate_ensemble

# %% papermill={"duration": 10573.57241, "end_time": "2021-01-06T12:27:43.877596", "exception": false, "start_time": "2021-01-06T09:31:30.305186", "status": "completed"} tags=[]
ensemble = generate_ensemble(
    data,
    CLUSTERERS,
    attributes=CLUSTERING_ATTRIBUTES_TO_SAVE,
)

# %% papermill={"duration": 0.086559, "end_time": "2021-01-06T12:27:44.034594", "exception": false, "start_time": "2021-01-06T12:27:43.948035", "status": "completed"} tags=[]
# the number should be close to 295 (the number of partitions generated by k-means/spectral clustering)
ensemble.shape

# %% papermill={"duration": 0.092459, "end_time": "2021-01-06T12:27:44.198555", "exception": false, "start_time": "2021-01-06T12:27:44.106096", "status": "completed"} tags=[]
ensemble.head()

# %% papermill={"duration": 0.087183, "end_time": "2021-01-06T12:27:44.357575", "exception": false, "start_time": "2021-01-06T12:27:44.270392", "status": "completed"} tags=[]
ensemble["n_clusters"].value_counts().head()

# %% papermill={"duration": 0.088407, "end_time": "2021-01-06T12:27:44.517805", "exception": false, "start_time": "2021-01-06T12:27:44.429398", "status": "completed"} tags=[]
ensemble_stats = ensemble["n_clusters"].describe()
display(ensemble_stats)

# %% [markdown] papermill={"duration": 0.071606, "end_time": "2021-01-06T12:27:44.660802", "exception": false, "start_time": "2021-01-06T12:27:44.589196", "status": "completed"} tags=[]
# ## Testing

# %% papermill={"duration": 0.089032, "end_time": "2021-01-06T12:27:44.821040", "exception": false, "start_time": "2021-01-06T12:27:44.732008", "status": "completed"} tags=[]
assert ensemble_stats["min"] > 1

# %% papermill={"duration": 0.085613, "end_time": "2021-01-06T12:27:44.978865", "exception": false, "start_time": "2021-01-06T12:27:44.893252", "status": "completed"} tags=[]
assert not ensemble["n_clusters"].isna().any()

# %% papermill={"duration": 0.086459, "end_time": "2021-01-06T12:27:45.136773", "exception": false, "start_time": "2021-01-06T12:27:45.050314", "status": "completed"} tags=[]
assert ensemble.shape[0] == len(CLUSTERERS)

# %% papermill={"duration": 0.103422, "end_time": "2021-01-06T12:27:45.312728", "exception": false, "start_time": "2021-01-06T12:27:45.209306", "status": "completed"} tags=[]
# all partitions have the right size
assert np.all(
    [part["partition"].shape[0] == data.shape[0] for idx, part in ensemble.iterrows()]
)

# %% papermill={"duration": 0.106217, "end_time": "2021-01-06T12:27:45.490544", "exception": false, "start_time": "2021-01-06T12:27:45.384327", "status": "completed"} tags=[]
# no partition has negative clusters (noisy points)
assert not np.any([(part["partition"] < 0).any() for idx, part in ensemble.iterrows()])

# %% [markdown] papermill={"duration": 0.071915, "end_time": "2021-01-06T12:27:45.635422", "exception": false, "start_time": "2021-01-06T12:27:45.563507", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.0894, "end_time": "2021-01-06T12:27:45.796065", "exception": false, "start_time": "2021-01-06T12:27:45.706665", "status": "completed"} tags=[]
output_filename = Path(
    RESULTS_DIR,
    generate_result_set_name(
        CLUSTERING_OPTIONS,
        prefix=f"{clustering_method_name}-",
        suffix=".pkl",
    ),
).resolve()
display(output_filename)

# %% papermill={"duration": 0.096736, "end_time": "2021-01-06T12:27:45.966675", "exception": false, "start_time": "2021-01-06T12:27:45.869939", "status": "completed"} tags=[]
ensemble.to_pickle(output_filename)

# %% [markdown] papermill={"duration": 0.073241, "end_time": "2021-01-06T12:27:46.112296", "exception": false, "start_time": "2021-01-06T12:27:46.039055", "status": "completed"} tags=[]
# # Stability

# %% [markdown] papermill={"duration": 0.072467, "end_time": "2021-01-06T12:27:46.255757", "exception": false, "start_time": "2021-01-06T12:27:46.183290", "status": "completed"} tags=[]
# ## Group ensemble by n_clusters

# %% papermill={"duration": 0.099882, "end_time": "2021-01-06T12:27:46.427215", "exception": false, "start_time": "2021-01-06T12:27:46.327333", "status": "completed"} tags=[]
parts = ensemble.groupby("n_clusters").apply(
    lambda x: np.concatenate(x["partition"].apply(lambda x: x.reshape(1, -1)), axis=0)
)

# %% papermill={"duration": 0.098294, "end_time": "2021-01-06T12:27:46.597906", "exception": false, "start_time": "2021-01-06T12:27:46.499612", "status": "completed"} tags=[]
parts.head()

# %% papermill={"duration": 0.086397, "end_time": "2021-01-06T12:27:46.756464", "exception": false, "start_time": "2021-01-06T12:27:46.670067", "status": "completed"} tags=[]
assert np.all(
    [
        parts.loc[k].shape == (CLUSTERING_OPTIONS["N_REPS_PER_K"], data.shape[0])
        for k in parts.index
    ]
)

# %% [markdown] papermill={"duration": 0.071515, "end_time": "2021-01-06T12:27:46.902112", "exception": false, "start_time": "2021-01-06T12:27:46.830597", "status": "completed"} tags=[]
# ## Compute stability

# %% papermill={"duration": 0.084929, "end_time": "2021-01-06T12:27:47.059067", "exception": false, "start_time": "2021-01-06T12:27:46.974138", "status": "completed"} tags=[]
from sklearn.metrics import adjusted_rand_score as ari
from scipy.spatial.distance import squareform, pdist

# %% papermill={"duration": 0.709435, "end_time": "2021-01-06T12:27:47.840534", "exception": false, "start_time": "2021-01-06T12:27:47.131099", "status": "completed"} tags=[]
parts_ari = pd.Series(
    {k: pdist(parts.loc[k], metric=ari) for k in parts.index}, name="k"
)

# %% papermill={"duration": 0.088306, "end_time": "2021-01-06T12:27:48.000313", "exception": false, "start_time": "2021-01-06T12:27:47.912007", "status": "completed"} tags=[]
parts_ari_stability = parts_ari.apply(lambda x: x.mean())
display(parts_ari_stability.sort_values(ascending=False).head(15))

# %% papermill={"duration": 0.08858, "end_time": "2021-01-06T12:27:48.161297", "exception": false, "start_time": "2021-01-06T12:27:48.072717", "status": "completed"} tags=[]
parts_ari_df = pd.DataFrame.from_records(parts_ari.tolist()).set_index(
    parts_ari.index.copy()
)

# %% papermill={"duration": 0.087636, "end_time": "2021-01-06T12:27:48.321609", "exception": false, "start_time": "2021-01-06T12:27:48.233973", "status": "completed"} tags=[]
parts_ari_df.shape

# %% papermill={"duration": 0.086225, "end_time": "2021-01-06T12:27:48.479966", "exception": false, "start_time": "2021-01-06T12:27:48.393741", "status": "completed"} tags=[]
assert (
    int(
        (CLUSTERING_OPTIONS["N_REPS_PER_K"] * (CLUSTERING_OPTIONS["N_REPS_PER_K"] - 1))
        / 2
    )
    == parts_ari_df.shape[1]
)

# %% papermill={"duration": 0.092455, "end_time": "2021-01-06T12:27:48.645599", "exception": false, "start_time": "2021-01-06T12:27:48.553144", "status": "completed"} tags=[]
parts_ari_df.head()

# %% [markdown] papermill={"duration": 0.074961, "end_time": "2021-01-06T12:27:48.793636", "exception": false, "start_time": "2021-01-06T12:27:48.718675", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.087988, "end_time": "2021-01-06T12:27:48.954599", "exception": false, "start_time": "2021-01-06T12:27:48.866611", "status": "completed"} tags=[]
output_filename = Path(
    RESULTS_DIR,
    generate_result_set_name(
        CLUSTERING_OPTIONS,
        prefix=f"{clustering_method_name}-stability-",
        suffix=".pkl",
    ),
).resolve()
display(output_filename)

# %% papermill={"duration": 0.08808, "end_time": "2021-01-06T12:27:49.116952", "exception": false, "start_time": "2021-01-06T12:27:49.028872", "status": "completed"} tags=[]
parts_ari_df.to_pickle(output_filename)

# %% [markdown] papermill={"duration": 0.072696, "end_time": "2021-01-06T12:27:49.263436", "exception": false, "start_time": "2021-01-06T12:27:49.190740", "status": "completed"} tags=[]
# ## Stability plot

# %% papermill={"duration": 0.089051, "end_time": "2021-01-06T12:27:49.426176", "exception": false, "start_time": "2021-01-06T12:27:49.337125", "status": "completed"} tags=[]
parts_ari_df_plot = (
    parts_ari_df.stack()
    .reset_index()
    .rename(columns={"level_0": "k", "level_1": "idx", 0: "ari"})
)

# %% papermill={"duration": 0.088475, "end_time": "2021-01-06T12:27:49.590053", "exception": false, "start_time": "2021-01-06T12:27:49.501578", "status": "completed"} tags=[]
parts_ari_df_plot.dtypes

# %% papermill={"duration": 0.090427, "end_time": "2021-01-06T12:27:49.755436", "exception": false, "start_time": "2021-01-06T12:27:49.665009", "status": "completed"} tags=[]
parts_ari_df_plot.head()

# %% papermill={"duration": 2.958004, "end_time": "2021-01-06T12:27:52.790413", "exception": false, "start_time": "2021-01-06T12:27:49.832409", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.074939, "end_time": "2021-01-06T12:27:52.942652", "exception": false, "start_time": "2021-01-06T12:27:52.867713", "status": "completed"} tags=[]
