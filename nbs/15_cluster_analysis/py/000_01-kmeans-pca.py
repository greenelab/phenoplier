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

# %% [markdown] papermill={"duration": 0.055983, "end_time": "2021-01-05T16:42:31.123374", "exception": false, "start_time": "2021-01-05T16:42:31.067391", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.02066, "end_time": "2021-01-05T16:42:31.165060", "exception": false, "start_time": "2021-01-05T16:42:31.144400", "status": "completed"} tags=[]
# Runs k-means on the pca version of the data.

# %% [markdown] papermill={"duration": 0.020517, "end_time": "2021-01-05T16:42:31.206160", "exception": false, "start_time": "2021-01-05T16:42:31.185643", "status": "completed"} tags=[]
# # Environment variables

# %% papermill={"duration": 0.034553, "end_time": "2021-01-05T16:42:31.261241", "exception": false, "start_time": "2021-01-05T16:42:31.226688", "status": "completed"} tags=[]
from IPython.display import display

import conf

N_JOBS = conf.GENERAL["N_JOBS"]
display(N_JOBS)

# %% papermill={"duration": 0.027751, "end_time": "2021-01-05T16:42:31.311117", "exception": false, "start_time": "2021-01-05T16:42:31.283366", "status": "completed"} tags=[]
# %env MKL_NUM_THREADS=$N_JOBS
# %env OPEN_BLAS_NUM_THREADS=$N_JOBS
# %env NUMEXPR_NUM_THREADS=$N_JOBS
# %env OMP_NUM_THREADS=$N_JOBS

# %% [markdown] papermill={"duration": 0.023057, "end_time": "2021-01-05T16:42:31.356136", "exception": false, "start_time": "2021-01-05T16:42:31.333079", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.031937, "end_time": "2021-01-05T16:42:31.410085", "exception": false, "start_time": "2021-01-05T16:42:31.378148", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 1.59829, "end_time": "2021-01-05T16:42:33.030670", "exception": false, "start_time": "2021-01-05T16:42:31.432380", "status": "completed"} tags=[]
from pathlib import Path

import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns

from utils import generate_result_set_name

# %% [markdown] papermill={"duration": 0.021367, "end_time": "2021-01-05T16:42:33.075656", "exception": false, "start_time": "2021-01-05T16:42:33.054289", "status": "completed"} tags=[]
# # Settings

# %% papermill={"duration": 0.035011, "end_time": "2021-01-05T16:42:33.131775", "exception": false, "start_time": "2021-01-05T16:42:33.096764", "status": "completed"} tags=[]
INITIAL_RANDOM_STATE = 10000

# %% [markdown] papermill={"duration": 0.022108, "end_time": "2021-01-05T16:42:33.175363", "exception": false, "start_time": "2021-01-05T16:42:33.153255", "status": "completed"} tags=[]
# ## Input data

# %% papermill={"duration": 0.035234, "end_time": "2021-01-05T16:42:33.232028", "exception": false, "start_time": "2021-01-05T16:42:33.196794", "status": "completed"} tags=[]
INPUT_SUBSET = "pca"

# %% papermill={"duration": 0.035355, "end_time": "2021-01-05T16:42:33.289117", "exception": false, "start_time": "2021-01-05T16:42:33.253762", "status": "completed"} tags=[]
INPUT_STEM = "z_score_std-projection-smultixcan-efo_partial-mashr-zscores"

# %% papermill={"duration": 0.035561, "end_time": "2021-01-05T16:42:33.347573", "exception": false, "start_time": "2021-01-05T16:42:33.312012", "status": "completed"} tags=[]
DR_OPTIONS = {
    "n_components": 50,
    "svd_solver": "full",
    "random_state": 0,
}

# %% papermill={"duration": 0.03895, "end_time": "2021-01-05T16:42:33.409330", "exception": false, "start_time": "2021-01-05T16:42:33.370380", "status": "completed"} tags=[]
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

# %% [markdown] papermill={"duration": 0.021309, "end_time": "2021-01-05T16:42:33.452855", "exception": false, "start_time": "2021-01-05T16:42:33.431546", "status": "completed"} tags=[]
# ## Clustering

# %% papermill={"duration": 0.042049, "end_time": "2021-01-05T16:42:33.516640", "exception": false, "start_time": "2021-01-05T16:42:33.474591", "status": "completed"} tags=[]
from sklearn.cluster import KMeans

# %% papermill={"duration": 0.041208, "end_time": "2021-01-05T16:42:33.579842", "exception": false, "start_time": "2021-01-05T16:42:33.538634", "status": "completed"} tags=[]
CLUSTERING_ATTRIBUTES_TO_SAVE = ["n_clusters"]

# %% papermill={"duration": 0.036799, "end_time": "2021-01-05T16:42:33.659741", "exception": false, "start_time": "2021-01-05T16:42:33.622942", "status": "completed"} tags=[]
CLUSTERING_OPTIONS = {}

CLUSTERING_OPTIONS["K_MIN"] = 2
CLUSTERING_OPTIONS["K_MAX"] = 60  # sqrt(3749)
CLUSTERING_OPTIONS["N_REPS_PER_K"] = 5
CLUSTERING_OPTIONS["KMEANS_N_INIT"] = 10

display(CLUSTERING_OPTIONS)

# %% papermill={"duration": 0.037856, "end_time": "2021-01-05T16:42:33.719639", "exception": false, "start_time": "2021-01-05T16:42:33.681783", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.036944, "end_time": "2021-01-05T16:42:33.779079", "exception": false, "start_time": "2021-01-05T16:42:33.742135", "status": "completed"} tags=[]
display(len(CLUSTERERS))

# %% papermill={"duration": 0.038816, "end_time": "2021-01-05T16:42:33.840367", "exception": false, "start_time": "2021-01-05T16:42:33.801551", "status": "completed"} tags=[]
_iter = iter(CLUSTERERS.items())
display(next(_iter))
display(next(_iter))

# %% papermill={"duration": 0.037202, "end_time": "2021-01-05T16:42:33.900928", "exception": false, "start_time": "2021-01-05T16:42:33.863726", "status": "completed"} tags=[]
clustering_method_name = method_name
display(clustering_method_name)

# %% [markdown] papermill={"duration": 0.023366, "end_time": "2021-01-05T16:42:33.949046", "exception": false, "start_time": "2021-01-05T16:42:33.925680", "status": "completed"} tags=[]
# ## Output directory

# %% papermill={"duration": 0.038233, "end_time": "2021-01-05T16:42:34.009956", "exception": false, "start_time": "2021-01-05T16:42:33.971723", "status": "completed"} tags=[]
# output dir for this notebook
RESULTS_DIR = Path(
    conf.RESULTS["CLUSTERING_RUNS_DIR"],
    f"{INPUT_SUBSET}-{INPUT_STEM}",
).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %% [markdown] papermill={"duration": 0.02332, "end_time": "2021-01-05T16:42:34.057482", "exception": false, "start_time": "2021-01-05T16:42:34.034162", "status": "completed"} tags=[]
# # Load input file

# %% papermill={"duration": 0.038975, "end_time": "2021-01-05T16:42:34.119224", "exception": false, "start_time": "2021-01-05T16:42:34.080249", "status": "completed"} tags=[]
data = pd.read_pickle(input_filepath)

# %% papermill={"duration": 0.038235, "end_time": "2021-01-05T16:42:34.181529", "exception": false, "start_time": "2021-01-05T16:42:34.143294", "status": "completed"} tags=[]
data.shape

# %% papermill={"duration": 0.050245, "end_time": "2021-01-05T16:42:34.255732", "exception": false, "start_time": "2021-01-05T16:42:34.205487", "status": "completed"} tags=[]
data.head()

# %% papermill={"duration": 0.03853, "end_time": "2021-01-05T16:42:34.319029", "exception": false, "start_time": "2021-01-05T16:42:34.280499", "status": "completed"} tags=[]
assert not data.isna().any().any()

# %% [markdown] papermill={"duration": 0.025681, "end_time": "2021-01-05T16:42:34.369080", "exception": false, "start_time": "2021-01-05T16:42:34.343399", "status": "completed"} tags=[]
# # Clustering

# %% [markdown] papermill={"duration": 0.023362, "end_time": "2021-01-05T16:42:34.417551", "exception": false, "start_time": "2021-01-05T16:42:34.394189", "status": "completed"} tags=[]
# ## Generate ensemble

# %% papermill={"duration": 0.041034, "end_time": "2021-01-05T16:42:34.482771", "exception": false, "start_time": "2021-01-05T16:42:34.441737", "status": "completed"} tags=[]
from clustering.ensemble import generate_ensemble

# %% papermill={"duration": 353.849554, "end_time": "2021-01-05T16:48:28.356941", "exception": false, "start_time": "2021-01-05T16:42:34.507387", "status": "completed"} tags=[]
ensemble = generate_ensemble(
    data,
    CLUSTERERS,
    attributes=CLUSTERING_ATTRIBUTES_TO_SAVE,
)

# %% papermill={"duration": 0.085714, "end_time": "2021-01-05T16:48:28.511526", "exception": false, "start_time": "2021-01-05T16:48:28.425812", "status": "completed"} tags=[]
# the number should be close to 295 (the number of partitions generated by k-means/spectral clustering)
ensemble.shape

# %% papermill={"duration": 0.091134, "end_time": "2021-01-05T16:48:28.673665", "exception": false, "start_time": "2021-01-05T16:48:28.582531", "status": "completed"} tags=[]
ensemble.head()

# %% papermill={"duration": 0.088756, "end_time": "2021-01-05T16:48:28.832882", "exception": false, "start_time": "2021-01-05T16:48:28.744126", "status": "completed"} tags=[]
ensemble["n_clusters"].value_counts().head()

# %% papermill={"duration": 0.088174, "end_time": "2021-01-05T16:48:28.993393", "exception": false, "start_time": "2021-01-05T16:48:28.905219", "status": "completed"} tags=[]
ensemble_stats = ensemble["n_clusters"].describe()
display(ensemble_stats)

# %% [markdown] papermill={"duration": 0.070194, "end_time": "2021-01-05T16:48:29.134684", "exception": false, "start_time": "2021-01-05T16:48:29.064490", "status": "completed"} tags=[]
# ## Testing

# %% papermill={"duration": 0.084699, "end_time": "2021-01-05T16:48:29.289431", "exception": false, "start_time": "2021-01-05T16:48:29.204732", "status": "completed"} tags=[]
assert ensemble_stats["min"] > 1

# %% papermill={"duration": 0.08578, "end_time": "2021-01-05T16:48:29.445921", "exception": false, "start_time": "2021-01-05T16:48:29.360141", "status": "completed"} tags=[]
assert not ensemble["n_clusters"].isna().any()

# %% papermill={"duration": 0.085294, "end_time": "2021-01-05T16:48:29.602744", "exception": false, "start_time": "2021-01-05T16:48:29.517450", "status": "completed"} tags=[]
assert ensemble.shape[0] == len(CLUSTERERS)

# %% papermill={"duration": 0.101502, "end_time": "2021-01-05T16:48:29.773952", "exception": false, "start_time": "2021-01-05T16:48:29.672450", "status": "completed"} tags=[]
# all partitions have the right size
assert np.all(
    [part["partition"].shape[0] == data.shape[0] for idx, part in ensemble.iterrows()]
)

# %% papermill={"duration": 0.106083, "end_time": "2021-01-05T16:48:29.953802", "exception": false, "start_time": "2021-01-05T16:48:29.847719", "status": "completed"} tags=[]
# no partition has negative clusters (noisy points)
assert not np.any([(part["partition"] < 0).any() for idx, part in ensemble.iterrows()])

# %% [markdown] papermill={"duration": 0.070111, "end_time": "2021-01-05T16:48:30.095736", "exception": false, "start_time": "2021-01-05T16:48:30.025625", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.086134, "end_time": "2021-01-05T16:48:30.251755", "exception": false, "start_time": "2021-01-05T16:48:30.165621", "status": "completed"} tags=[]
output_filename = Path(
    RESULTS_DIR,
    generate_result_set_name(
        CLUSTERING_OPTIONS,
        prefix=f"{clustering_method_name}-",
        suffix=".pkl",
    ),
).resolve()
display(output_filename)

# %% papermill={"duration": 0.095038, "end_time": "2021-01-05T16:48:30.418035", "exception": false, "start_time": "2021-01-05T16:48:30.322997", "status": "completed"} tags=[]
ensemble.to_pickle(output_filename)

# %% [markdown] papermill={"duration": 0.070235, "end_time": "2021-01-05T16:48:30.561152", "exception": false, "start_time": "2021-01-05T16:48:30.490917", "status": "completed"} tags=[]
# # Stability

# %% [markdown] papermill={"duration": 0.07035, "end_time": "2021-01-05T16:48:30.702690", "exception": false, "start_time": "2021-01-05T16:48:30.632340", "status": "completed"} tags=[]
# ## Group ensemble by n_clusters

# %% papermill={"duration": 0.105543, "end_time": "2021-01-05T16:48:30.878562", "exception": false, "start_time": "2021-01-05T16:48:30.773019", "status": "completed"} tags=[]
parts = ensemble.groupby("n_clusters").apply(
    lambda x: np.concatenate(x["partition"].apply(lambda x: x.reshape(1, -1)), axis=0)
)

# %% papermill={"duration": 0.09709, "end_time": "2021-01-05T16:48:31.047644", "exception": false, "start_time": "2021-01-05T16:48:30.950554", "status": "completed"} tags=[]
parts.head()

# %% papermill={"duration": 0.086119, "end_time": "2021-01-05T16:48:31.205903", "exception": false, "start_time": "2021-01-05T16:48:31.119784", "status": "completed"} tags=[]
assert np.all(
    [
        parts.loc[k].shape == (CLUSTERING_OPTIONS["N_REPS_PER_K"], data.shape[0])
        for k in parts.index
    ]
)

# %% [markdown] papermill={"duration": 0.070178, "end_time": "2021-01-05T16:48:31.347306", "exception": false, "start_time": "2021-01-05T16:48:31.277128", "status": "completed"} tags=[]
# ## Compute stability

# %% papermill={"duration": 0.08551, "end_time": "2021-01-05T16:48:31.502903", "exception": false, "start_time": "2021-01-05T16:48:31.417393", "status": "completed"} tags=[]
from sklearn.metrics import adjusted_rand_score as ari
from scipy.spatial.distance import squareform, pdist

# %% papermill={"duration": 0.710978, "end_time": "2021-01-05T16:48:32.284392", "exception": false, "start_time": "2021-01-05T16:48:31.573414", "status": "completed"} tags=[]
parts_ari = pd.Series(
    {k: pdist(parts.loc[k], metric=ari) for k in parts.index}, name="k"
)

# %% papermill={"duration": 0.087639, "end_time": "2021-01-05T16:48:32.443213", "exception": false, "start_time": "2021-01-05T16:48:32.355574", "status": "completed"} tags=[]
parts_ari_stability = parts_ari.apply(lambda x: x.mean())
display(parts_ari_stability.sort_values(ascending=False).head(15))

# %% papermill={"duration": 0.086812, "end_time": "2021-01-05T16:48:32.602425", "exception": false, "start_time": "2021-01-05T16:48:32.515613", "status": "completed"} tags=[]
parts_ari_df = pd.DataFrame.from_records(parts_ari.tolist()).set_index(
    parts_ari.index.copy()
)

# %% papermill={"duration": 0.086694, "end_time": "2021-01-05T16:48:32.762001", "exception": false, "start_time": "2021-01-05T16:48:32.675307", "status": "completed"} tags=[]
parts_ari_df.shape

# %% papermill={"duration": 0.087704, "end_time": "2021-01-05T16:48:32.924328", "exception": false, "start_time": "2021-01-05T16:48:32.836624", "status": "completed"} tags=[]
assert (
    int(
        (CLUSTERING_OPTIONS["N_REPS_PER_K"] * (CLUSTERING_OPTIONS["N_REPS_PER_K"] - 1))
        / 2
    )
    == parts_ari_df.shape[1]
)

# %% papermill={"duration": 0.092118, "end_time": "2021-01-05T16:48:33.089997", "exception": false, "start_time": "2021-01-05T16:48:32.997879", "status": "completed"} tags=[]
parts_ari_df.head()

# %% [markdown] papermill={"duration": 0.07163, "end_time": "2021-01-05T16:48:33.234196", "exception": false, "start_time": "2021-01-05T16:48:33.162566", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.08636, "end_time": "2021-01-05T16:48:33.392194", "exception": false, "start_time": "2021-01-05T16:48:33.305834", "status": "completed"} tags=[]
output_filename = Path(
    RESULTS_DIR,
    generate_result_set_name(
        CLUSTERING_OPTIONS,
        prefix=f"{clustering_method_name}-stability-",
        suffix=".pkl",
    ),
).resolve()
display(output_filename)

# %% papermill={"duration": 0.086765, "end_time": "2021-01-05T16:48:33.551729", "exception": false, "start_time": "2021-01-05T16:48:33.464964", "status": "completed"} tags=[]
parts_ari_df.to_pickle(output_filename)

# %% [markdown] papermill={"duration": 0.071224, "end_time": "2021-01-05T16:48:33.696365", "exception": false, "start_time": "2021-01-05T16:48:33.625141", "status": "completed"} tags=[]
# ## Stability plot

# %% papermill={"duration": 0.090986, "end_time": "2021-01-05T16:48:33.859492", "exception": false, "start_time": "2021-01-05T16:48:33.768506", "status": "completed"} tags=[]
parts_ari_df_plot = (
    parts_ari_df.stack()
    .reset_index()
    .rename(columns={"level_0": "k", "level_1": "idx", 0: "ari"})
)

# %% papermill={"duration": 0.088543, "end_time": "2021-01-05T16:48:34.021119", "exception": false, "start_time": "2021-01-05T16:48:33.932576", "status": "completed"} tags=[]
parts_ari_df_plot.dtypes

# %% papermill={"duration": 0.089595, "end_time": "2021-01-05T16:48:34.183997", "exception": false, "start_time": "2021-01-05T16:48:34.094402", "status": "completed"} tags=[]
parts_ari_df_plot.head()

# %% papermill={"duration": 3.041707, "end_time": "2021-01-05T16:48:37.297615", "exception": false, "start_time": "2021-01-05T16:48:34.255908", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.073386, "end_time": "2021-01-05T16:48:37.445363", "exception": false, "start_time": "2021-01-05T16:48:37.371977", "status": "completed"} tags=[]
