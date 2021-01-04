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

# %% [markdown] papermill={"duration": 0.05426, "end_time": "2020-12-02T18:22:01.689838", "exception": false, "start_time": "2020-12-02T18:22:01.635578", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.019601, "end_time": "2020-12-02T18:22:01.729822", "exception": false, "start_time": "2020-12-02T18:22:01.710221", "status": "completed"} tags=[]
# Runs k-means on the z_score_std version of the data.

# %% [markdown] papermill={"duration": 0.019576, "end_time": "2020-12-02T18:22:01.769044", "exception": false, "start_time": "2020-12-02T18:22:01.749468", "status": "completed"} tags=[]
# # Environment variables

# %% papermill={"duration": 0.034031, "end_time": "2020-12-02T18:22:01.822660", "exception": false, "start_time": "2020-12-02T18:22:01.788629", "status": "completed"} tags=[]
from IPython.display import display

import conf

N_JOBS = conf.GENERAL["N_JOBS"]
display(N_JOBS)

# %% papermill={"duration": 0.026826, "end_time": "2020-12-02T18:22:01.870278", "exception": false, "start_time": "2020-12-02T18:22:01.843452", "status": "completed"} tags=[]
# %env MKL_NUM_THREADS=$N_JOBS
# %env OPEN_BLAS_NUM_THREADS=$N_JOBS
# %env NUMEXPR_NUM_THREADS=$N_JOBS
# %env OMP_NUM_THREADS=$N_JOBS

# %% [markdown] papermill={"duration": 0.020494, "end_time": "2020-12-02T18:22:01.911488", "exception": false, "start_time": "2020-12-02T18:22:01.890994", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.031244, "end_time": "2020-12-02T18:22:01.962757", "exception": false, "start_time": "2020-12-02T18:22:01.931513", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 1.59627, "end_time": "2020-12-02T18:22:03.580325", "exception": false, "start_time": "2020-12-02T18:22:01.984055", "status": "completed"} tags=[]
from pathlib import Path

import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns

from utils import generate_result_set_name

# %% [markdown] papermill={"duration": 0.020613, "end_time": "2020-12-02T18:22:03.623273", "exception": false, "start_time": "2020-12-02T18:22:03.602660", "status": "completed"} tags=[]
# # Settings

# %% papermill={"duration": 0.034119, "end_time": "2020-12-02T18:22:03.678037", "exception": false, "start_time": "2020-12-02T18:22:03.643918", "status": "completed"} tags=[]
INITIAL_RANDOM_STATE = 0

# %% [markdown] papermill={"duration": 0.019944, "end_time": "2020-12-02T18:22:03.718423", "exception": false, "start_time": "2020-12-02T18:22:03.698479", "status": "completed"} tags=[]
# ## Input data

# %% papermill={"duration": 0.033993, "end_time": "2020-12-02T18:22:03.772878", "exception": false, "start_time": "2020-12-02T18:22:03.738885", "status": "completed"} tags=[]
INPUT_SUBSET = "z_score_std"

# %% papermill={"duration": 0.034256, "end_time": "2020-12-02T18:22:03.827176", "exception": false, "start_time": "2020-12-02T18:22:03.792920", "status": "completed"} tags=[]
INPUT_STEM = "projection-smultixcan-efo_partial-mashr-zscores"

# %% papermill={"duration": 0.036583, "end_time": "2020-12-02T18:22:03.884665", "exception": false, "start_time": "2020-12-02T18:22:03.848082", "status": "completed"} tags=[]
input_filepath = Path(
    conf.RESULTS["DATA_TRANSFORMATIONS_DIR"],
    INPUT_SUBSET,
    f"{INPUT_SUBSET}-{INPUT_STEM}.pkl",
).resolve()
display(input_filepath)

assert input_filepath.exists(), "Input file does not exist"

input_filepath_stem = input_filepath.stem
display(input_filepath_stem)

# %% [markdown] papermill={"duration": 0.020548, "end_time": "2020-12-02T18:22:03.925546", "exception": false, "start_time": "2020-12-02T18:22:03.904998", "status": "completed"} tags=[]
# ## Clustering

# %% papermill={"duration": 0.040655, "end_time": "2020-12-02T18:22:03.986832", "exception": false, "start_time": "2020-12-02T18:22:03.946177", "status": "completed"} tags=[]
from sklearn.cluster import KMeans

# %% papermill={"duration": 0.034694, "end_time": "2020-12-02T18:22:04.042159", "exception": false, "start_time": "2020-12-02T18:22:04.007465", "status": "completed"} tags=[]
CLUSTERING_ATTRIBUTES_TO_SAVE = ["n_clusters"]

# %% papermill={"duration": 0.035716, "end_time": "2020-12-02T18:22:04.099556", "exception": false, "start_time": "2020-12-02T18:22:04.063840", "status": "completed"} tags=[]
CLUSTERING_OPTIONS = {}

CLUSTERING_OPTIONS["K_MIN"] = 2
CLUSTERING_OPTIONS["K_MAX"] = 60  # sqrt(3749)
CLUSTERING_OPTIONS["N_REPS_PER_K"] = 5
CLUSTERING_OPTIONS["KMEANS_N_INIT"] = 10

display(CLUSTERING_OPTIONS)

# %% papermill={"duration": 0.037202, "end_time": "2020-12-02T18:22:04.157543", "exception": false, "start_time": "2020-12-02T18:22:04.120341", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.035634, "end_time": "2020-12-02T18:22:04.214550", "exception": false, "start_time": "2020-12-02T18:22:04.178916", "status": "completed"} tags=[]
display(len(CLUSTERERS))

# %% papermill={"duration": 0.038804, "end_time": "2020-12-02T18:22:04.276291", "exception": false, "start_time": "2020-12-02T18:22:04.237487", "status": "completed"} tags=[]
_iter = iter(CLUSTERERS.items())
display(next(_iter))
display(next(_iter))

# %% papermill={"duration": 0.036776, "end_time": "2020-12-02T18:22:04.335973", "exception": false, "start_time": "2020-12-02T18:22:04.299197", "status": "completed"} tags=[]
clustering_method_name = method_name
display(clustering_method_name)

# %% [markdown] papermill={"duration": 0.021348, "end_time": "2020-12-02T18:22:04.379083", "exception": false, "start_time": "2020-12-02T18:22:04.357735", "status": "completed"} tags=[]
# ## Output directory

# %% papermill={"duration": 0.03682, "end_time": "2020-12-02T18:22:04.437376", "exception": false, "start_time": "2020-12-02T18:22:04.400556", "status": "completed"} tags=[]
# output dir for this notebook
RESULTS_DIR = Path(
    conf.RESULTS["CLUSTERING_RUNS_DIR"],
    f"{INPUT_SUBSET}-{INPUT_STEM}",
).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %% [markdown] papermill={"duration": 0.022133, "end_time": "2020-12-02T18:22:04.482743", "exception": false, "start_time": "2020-12-02T18:22:04.460610", "status": "completed"} tags=[]
# # Load input file

# %% papermill={"duration": 0.047976, "end_time": "2020-12-02T18:22:04.552607", "exception": false, "start_time": "2020-12-02T18:22:04.504631", "status": "completed"} tags=[]
data = pd.read_pickle(input_filepath)

# %% papermill={"duration": 0.036934, "end_time": "2020-12-02T18:22:04.613671", "exception": false, "start_time": "2020-12-02T18:22:04.576737", "status": "completed"} tags=[]
data.shape

# %% papermill={"duration": 0.049043, "end_time": "2020-12-02T18:22:04.685862", "exception": false, "start_time": "2020-12-02T18:22:04.636819", "status": "completed"} tags=[]
data.head()

# %% papermill={"duration": 0.042498, "end_time": "2020-12-02T18:22:04.751908", "exception": false, "start_time": "2020-12-02T18:22:04.709410", "status": "completed"} tags=[]
assert not data.isna().any().any()

# %% [markdown] papermill={"duration": 0.021811, "end_time": "2020-12-02T18:22:04.796639", "exception": false, "start_time": "2020-12-02T18:22:04.774828", "status": "completed"} tags=[]
# # Clustering

# %% [markdown] papermill={"duration": 0.022473, "end_time": "2020-12-02T18:22:04.841464", "exception": false, "start_time": "2020-12-02T18:22:04.818991", "status": "completed"} tags=[]
# ## Generate ensemble

# %% papermill={"duration": 0.038865, "end_time": "2020-12-02T18:22:04.902241", "exception": false, "start_time": "2020-12-02T18:22:04.863376", "status": "completed"} tags=[]
from clustering.ensemble import generate_ensemble

# %% papermill={"duration": 1304.34656, "end_time": "2020-12-02T18:43:49.272375", "exception": false, "start_time": "2020-12-02T18:22:04.925815", "status": "completed"} tags=[]
ensemble = generate_ensemble(
    data,
    CLUSTERERS,
    attributes=CLUSTERING_ATTRIBUTES_TO_SAVE,
)

# %% papermill={"duration": 0.0839, "end_time": "2020-12-02T18:43:49.426385", "exception": false, "start_time": "2020-12-02T18:43:49.342485", "status": "completed"} tags=[]
# the number should be close to 295 (the number of partitions generated by k-means/spectral clustering)
ensemble.shape

# %% papermill={"duration": 0.091189, "end_time": "2020-12-02T18:43:49.588205", "exception": false, "start_time": "2020-12-02T18:43:49.497016", "status": "completed"} tags=[]
ensemble.head()

# %% papermill={"duration": 0.085114, "end_time": "2020-12-02T18:43:49.743266", "exception": false, "start_time": "2020-12-02T18:43:49.658152", "status": "completed"} tags=[]
ensemble["n_clusters"].value_counts().head()

# %% papermill={"duration": 0.085691, "end_time": "2020-12-02T18:43:49.898143", "exception": false, "start_time": "2020-12-02T18:43:49.812452", "status": "completed"} tags=[]
ensemble_stats = ensemble["n_clusters"].describe()
display(ensemble_stats)

# %% [markdown] papermill={"duration": 0.069375, "end_time": "2020-12-02T18:43:50.037181", "exception": false, "start_time": "2020-12-02T18:43:49.967806", "status": "completed"} tags=[]
# ## Testing

# %% papermill={"duration": 0.083149, "end_time": "2020-12-02T18:43:50.188996", "exception": false, "start_time": "2020-12-02T18:43:50.105847", "status": "completed"} tags=[]
assert ensemble_stats["min"] > 1

# %% papermill={"duration": 0.083573, "end_time": "2020-12-02T18:43:50.341548", "exception": false, "start_time": "2020-12-02T18:43:50.257975", "status": "completed"} tags=[]
assert not ensemble["n_clusters"].isna().any()

# %% papermill={"duration": 0.087081, "end_time": "2020-12-02T18:43:50.497841", "exception": false, "start_time": "2020-12-02T18:43:50.410760", "status": "completed"} tags=[]
assert ensemble.shape[0] == len(CLUSTERERS)

# %% papermill={"duration": 0.10085, "end_time": "2020-12-02T18:43:50.668430", "exception": false, "start_time": "2020-12-02T18:43:50.567580", "status": "completed"} tags=[]
# all partitions have the right size
assert np.all(
    [part["partition"].shape[0] == data.shape[0] for idx, part in ensemble.iterrows()]
)

# %% papermill={"duration": 0.10446, "end_time": "2020-12-02T18:43:50.843024", "exception": false, "start_time": "2020-12-02T18:43:50.738564", "status": "completed"} tags=[]
# no partition has negative clusters (noisy points)
assert not np.any([(part["partition"] < 0).any() for idx, part in ensemble.iterrows()])

# %% [markdown] papermill={"duration": 0.068896, "end_time": "2020-12-02T18:43:50.981376", "exception": false, "start_time": "2020-12-02T18:43:50.912480", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.085051, "end_time": "2020-12-02T18:43:51.137237", "exception": false, "start_time": "2020-12-02T18:43:51.052186", "status": "completed"} tags=[]
output_filename = Path(
    RESULTS_DIR,
    generate_result_set_name(
        CLUSTERING_OPTIONS,
        prefix=f"{clustering_method_name}-",
        suffix=".pkl",
    ),
).resolve()
display(output_filename)

# %% papermill={"duration": 0.106228, "end_time": "2020-12-02T18:43:51.314013", "exception": false, "start_time": "2020-12-02T18:43:51.207785", "status": "completed"} tags=[]
ensemble.to_pickle(output_filename)

# %% [markdown] papermill={"duration": 0.071866, "end_time": "2020-12-02T18:43:51.456874", "exception": false, "start_time": "2020-12-02T18:43:51.385008", "status": "completed"} tags=[]
# # Stability

# %% [markdown] papermill={"duration": 0.069979, "end_time": "2020-12-02T18:43:51.597097", "exception": false, "start_time": "2020-12-02T18:43:51.527118", "status": "completed"} tags=[]
# ## Group ensemble by n_clusters

# %% papermill={"duration": 0.097768, "end_time": "2020-12-02T18:43:51.764132", "exception": false, "start_time": "2020-12-02T18:43:51.666364", "status": "completed"} tags=[]
parts = ensemble.groupby("n_clusters").apply(
    lambda x: np.concatenate(x["partition"].apply(lambda x: x.reshape(1, -1)), axis=0)
)

# %% papermill={"duration": 0.096473, "end_time": "2020-12-02T18:43:51.931347", "exception": false, "start_time": "2020-12-02T18:43:51.834874", "status": "completed"} tags=[]
parts.head()

# %% papermill={"duration": 0.084798, "end_time": "2020-12-02T18:43:52.087435", "exception": false, "start_time": "2020-12-02T18:43:52.002637", "status": "completed"} tags=[]
assert np.all(
    [
        parts.loc[k].shape == (CLUSTERING_OPTIONS["N_REPS_PER_K"], data.shape[0])
        for k in parts.index
    ]
)

# %% [markdown] papermill={"duration": 0.069919, "end_time": "2020-12-02T18:43:52.228469", "exception": false, "start_time": "2020-12-02T18:43:52.158550", "status": "completed"} tags=[]
# ## Compute stability

# %% papermill={"duration": 0.085598, "end_time": "2020-12-02T18:43:52.384051", "exception": false, "start_time": "2020-12-02T18:43:52.298453", "status": "completed"} tags=[]
from sklearn.metrics import adjusted_rand_score as ari
from scipy.spatial.distance import squareform, pdist

# %% papermill={"duration": 0.76536, "end_time": "2020-12-02T18:43:53.221783", "exception": false, "start_time": "2020-12-02T18:43:52.456423", "status": "completed"} tags=[]
parts_ari = pd.Series(
    {k: pdist(parts.loc[k], metric=ari) for k in parts.index}, name="k"
)

# %% papermill={"duration": 0.086945, "end_time": "2020-12-02T18:43:53.379218", "exception": false, "start_time": "2020-12-02T18:43:53.292273", "status": "completed"} tags=[]
parts_ari_stability = parts_ari.apply(lambda x: x.mean())
display(parts_ari_stability.sort_values(ascending=False).head(15))

# %% papermill={"duration": 0.087633, "end_time": "2020-12-02T18:43:53.540095", "exception": false, "start_time": "2020-12-02T18:43:53.452462", "status": "completed"} tags=[]
parts_ari_df = pd.DataFrame.from_records(parts_ari.tolist()).set_index(
    parts_ari.index.copy()
)

# %% papermill={"duration": 0.084998, "end_time": "2020-12-02T18:43:53.695720", "exception": false, "start_time": "2020-12-02T18:43:53.610722", "status": "completed"} tags=[]
parts_ari_df.shape

# %% papermill={"duration": 0.085464, "end_time": "2020-12-02T18:43:53.851346", "exception": false, "start_time": "2020-12-02T18:43:53.765882", "status": "completed"} tags=[]
assert (
    int(
        (CLUSTERING_OPTIONS["N_REPS_PER_K"] * (CLUSTERING_OPTIONS["N_REPS_PER_K"] - 1))
        / 2
    )
    == parts_ari_df.shape[1]
)

# %% papermill={"duration": 0.093586, "end_time": "2020-12-02T18:43:54.017042", "exception": false, "start_time": "2020-12-02T18:43:53.923456", "status": "completed"} tags=[]
parts_ari_df.head()

# %% [markdown] papermill={"duration": 0.070752, "end_time": "2020-12-02T18:43:54.160846", "exception": false, "start_time": "2020-12-02T18:43:54.090094", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.086123, "end_time": "2020-12-02T18:43:54.317852", "exception": false, "start_time": "2020-12-02T18:43:54.231729", "status": "completed"} tags=[]
output_filename = Path(
    RESULTS_DIR,
    generate_result_set_name(
        CLUSTERING_OPTIONS,
        prefix=f"{clustering_method_name}-stability-",
        suffix=".pkl",
    ),
).resolve()
display(output_filename)

# %% papermill={"duration": 0.089318, "end_time": "2020-12-02T18:43:54.480862", "exception": false, "start_time": "2020-12-02T18:43:54.391544", "status": "completed"} tags=[]
parts_ari_df.to_pickle(output_filename)

# %% [markdown] papermill={"duration": 0.07067, "end_time": "2020-12-02T18:43:54.623657", "exception": false, "start_time": "2020-12-02T18:43:54.552987", "status": "completed"} tags=[]
# ## Stability plot

# %% papermill={"duration": 0.087657, "end_time": "2020-12-02T18:43:54.783062", "exception": false, "start_time": "2020-12-02T18:43:54.695405", "status": "completed"} tags=[]
parts_ari_df_plot = (
    parts_ari_df.stack()
    .reset_index()
    .rename(columns={"level_0": "k", "level_1": "idx", 0: "ari"})
)

# %% papermill={"duration": 0.087785, "end_time": "2020-12-02T18:43:54.943047", "exception": false, "start_time": "2020-12-02T18:43:54.855262", "status": "completed"} tags=[]
parts_ari_df_plot.dtypes

# %% papermill={"duration": 0.088645, "end_time": "2020-12-02T18:43:55.103744", "exception": false, "start_time": "2020-12-02T18:43:55.015099", "status": "completed"} tags=[]
parts_ari_df_plot.head()

# %% papermill={"duration": 2.938061, "end_time": "2020-12-02T18:43:58.114009", "exception": false, "start_time": "2020-12-02T18:43:55.175948", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.071793, "end_time": "2020-12-02T18:43:58.260068", "exception": false, "start_time": "2020-12-02T18:43:58.188275", "status": "completed"} tags=[]
