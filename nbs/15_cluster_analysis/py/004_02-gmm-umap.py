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

# %% [markdown] papermill={"duration": 0.05798, "end_time": "2021-01-06T12:27:55.465174", "exception": false, "start_time": "2021-01-06T12:27:55.407194", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.020689, "end_time": "2021-01-06T12:27:55.507526", "exception": false, "start_time": "2021-01-06T12:27:55.486837", "status": "completed"} tags=[]
# Runs gaussian mixture model on the umap version of the data.

# %% [markdown] papermill={"duration": 0.020778, "end_time": "2021-01-06T12:27:55.548982", "exception": false, "start_time": "2021-01-06T12:27:55.528204", "status": "completed"} tags=[]
# # Environment variables

# %% papermill={"duration": 0.035654, "end_time": "2021-01-06T12:27:55.605689", "exception": false, "start_time": "2021-01-06T12:27:55.570035", "status": "completed"} tags=[]
from IPython.display import display

import conf

N_JOBS = conf.GENERAL["N_JOBS"]
display(N_JOBS)

# %% papermill={"duration": 0.02808, "end_time": "2021-01-06T12:27:55.655841", "exception": false, "start_time": "2021-01-06T12:27:55.627761", "status": "completed"} tags=[]
# %env MKL_NUM_THREADS=$N_JOBS
# %env OPEN_BLAS_NUM_THREADS=$N_JOBS
# %env NUMEXPR_NUM_THREADS=$N_JOBS
# %env OMP_NUM_THREADS=$N_JOBS

# %% [markdown] papermill={"duration": 0.021188, "end_time": "2021-01-06T12:27:55.698979", "exception": false, "start_time": "2021-01-06T12:27:55.677791", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.031905, "end_time": "2021-01-06T12:27:55.752265", "exception": false, "start_time": "2021-01-06T12:27:55.720360", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 1.609752, "end_time": "2021-01-06T12:27:57.383850", "exception": false, "start_time": "2021-01-06T12:27:55.774098", "status": "completed"} tags=[]
from pathlib import Path

import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns

from utils import generate_result_set_name

# %% [markdown] papermill={"duration": 0.021295, "end_time": "2021-01-06T12:27:57.428974", "exception": false, "start_time": "2021-01-06T12:27:57.407679", "status": "completed"} tags=[]
# # Settings

# %% papermill={"duration": 0.036317, "end_time": "2021-01-06T12:27:57.486713", "exception": false, "start_time": "2021-01-06T12:27:57.450396", "status": "completed"} tags=[]
INITIAL_RANDOM_STATE = 80000

# %% [markdown] papermill={"duration": 0.020968, "end_time": "2021-01-06T12:27:57.529448", "exception": false, "start_time": "2021-01-06T12:27:57.508480", "status": "completed"} tags=[]
# ## Input data

# %% papermill={"duration": 0.035299, "end_time": "2021-01-06T12:27:57.586034", "exception": false, "start_time": "2021-01-06T12:27:57.550735", "status": "completed"} tags=[]
INPUT_SUBSET = "umap"

# %% papermill={"duration": 0.035357, "end_time": "2021-01-06T12:27:57.643367", "exception": false, "start_time": "2021-01-06T12:27:57.608010", "status": "completed"} tags=[]
INPUT_STEM = "z_score_std-projection-smultixcan-efo_partial-mashr-zscores"

# %% papermill={"duration": 0.035295, "end_time": "2021-01-06T12:27:57.700342", "exception": false, "start_time": "2021-01-06T12:27:57.665047", "status": "completed"} tags=[]
DR_OPTIONS = {
    "n_components": 50,
    "metric": "euclidean",
    "n_neighbors": 15,
    "random_state": 0,
}

# %% papermill={"duration": 0.037576, "end_time": "2021-01-06T12:27:57.759617", "exception": false, "start_time": "2021-01-06T12:27:57.722041", "status": "completed"} tags=[]
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

# %% [markdown] papermill={"duration": 0.023053, "end_time": "2021-01-06T12:27:57.805804", "exception": false, "start_time": "2021-01-06T12:27:57.782751", "status": "completed"} tags=[]
# ## Clustering

# %% papermill={"duration": 0.044309, "end_time": "2021-01-06T12:27:57.874267", "exception": false, "start_time": "2021-01-06T12:27:57.829958", "status": "completed"} tags=[]
from sklearn.mixture import GaussianMixture

# %% papermill={"duration": 0.036168, "end_time": "2021-01-06T12:27:57.933807", "exception": false, "start_time": "2021-01-06T12:27:57.897639", "status": "completed"} tags=[]
CLUSTERING_ATTRIBUTES_TO_SAVE = ["n_clusters"]

# %% papermill={"duration": 0.037358, "end_time": "2021-01-06T12:27:57.993358", "exception": false, "start_time": "2021-01-06T12:27:57.956000", "status": "completed"} tags=[]
CLUSTERING_OPTIONS = {}

CLUSTERING_OPTIONS["K_MIN"] = 2
CLUSTERING_OPTIONS["K_MAX"] = 60  # sqrt(3749)
CLUSTERING_OPTIONS["N_REPS_PER_K"] = 5
CLUSTERING_OPTIONS["N_INIT"] = 10
CLUSTERING_OPTIONS["COVARIANCE_TYPE"] = "full"

display(CLUSTERING_OPTIONS)

# %% papermill={"duration": 0.03831, "end_time": "2021-01-06T12:27:58.054730", "exception": false, "start_time": "2021-01-06T12:27:58.016420", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.037862, "end_time": "2021-01-06T12:27:58.114865", "exception": false, "start_time": "2021-01-06T12:27:58.077003", "status": "completed"} tags=[]
display(len(CLUSTERERS))

# %% papermill={"duration": 0.0388, "end_time": "2021-01-06T12:27:58.177345", "exception": false, "start_time": "2021-01-06T12:27:58.138545", "status": "completed"} tags=[]
_iter = iter(CLUSTERERS.items())
display(next(_iter))
display(next(_iter))

# %% papermill={"duration": 0.037268, "end_time": "2021-01-06T12:27:58.237725", "exception": false, "start_time": "2021-01-06T12:27:58.200457", "status": "completed"} tags=[]
clustering_method_name = method_name
display(clustering_method_name)

# %% [markdown] papermill={"duration": 0.023472, "end_time": "2021-01-06T12:27:58.285222", "exception": false, "start_time": "2021-01-06T12:27:58.261750", "status": "completed"} tags=[]
# ## Output directory

# %% papermill={"duration": 0.038444, "end_time": "2021-01-06T12:27:58.346687", "exception": false, "start_time": "2021-01-06T12:27:58.308243", "status": "completed"} tags=[]
# output dir for this notebook
RESULTS_DIR = Path(
    conf.RESULTS["CLUSTERING_RUNS_DIR"],
    f"{INPUT_SUBSET}-{INPUT_STEM}",
).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %% [markdown] papermill={"duration": 0.023206, "end_time": "2021-01-06T12:27:58.393182", "exception": false, "start_time": "2021-01-06T12:27:58.369976", "status": "completed"} tags=[]
# # Load input file

# %% papermill={"duration": 0.038744, "end_time": "2021-01-06T12:27:58.455059", "exception": false, "start_time": "2021-01-06T12:27:58.416315", "status": "completed"} tags=[]
data = pd.read_pickle(input_filepath)

# %% papermill={"duration": 0.038511, "end_time": "2021-01-06T12:27:58.517404", "exception": false, "start_time": "2021-01-06T12:27:58.478893", "status": "completed"} tags=[]
data.shape

# %% papermill={"duration": 0.050419, "end_time": "2021-01-06T12:27:58.591917", "exception": false, "start_time": "2021-01-06T12:27:58.541498", "status": "completed"} tags=[]
data.head()

# %% papermill={"duration": 0.038261, "end_time": "2021-01-06T12:27:58.654795", "exception": false, "start_time": "2021-01-06T12:27:58.616534", "status": "completed"} tags=[]
assert not data.isna().any().any()

# %% [markdown] papermill={"duration": 0.023956, "end_time": "2021-01-06T12:27:58.702960", "exception": false, "start_time": "2021-01-06T12:27:58.679004", "status": "completed"} tags=[]
# # Clustering

# %% [markdown] papermill={"duration": 0.023423, "end_time": "2021-01-06T12:27:58.749696", "exception": false, "start_time": "2021-01-06T12:27:58.726273", "status": "completed"} tags=[]
# ## Generate ensemble

# %% papermill={"duration": 0.041483, "end_time": "2021-01-06T12:27:58.814686", "exception": false, "start_time": "2021-01-06T12:27:58.773203", "status": "completed"} tags=[]
from clustering.ensemble import generate_ensemble

# %% papermill={"duration": 3023.561491, "end_time": "2021-01-06T13:18:22.401749", "exception": false, "start_time": "2021-01-06T12:27:58.840258", "status": "completed"} tags=[]
ensemble = generate_ensemble(
    data,
    CLUSTERERS,
    attributes=CLUSTERING_ATTRIBUTES_TO_SAVE,
)

# %% papermill={"duration": 0.085931, "end_time": "2021-01-06T13:18:22.566547", "exception": false, "start_time": "2021-01-06T13:18:22.480616", "status": "completed"} tags=[]
# the number should be close to 295 (the number of partitions generated by k-means/spectral clustering)
ensemble.shape

# %% papermill={"duration": 0.093077, "end_time": "2021-01-06T13:18:22.730990", "exception": false, "start_time": "2021-01-06T13:18:22.637913", "status": "completed"} tags=[]
ensemble.head()

# %% papermill={"duration": 0.08756, "end_time": "2021-01-06T13:18:22.891782", "exception": false, "start_time": "2021-01-06T13:18:22.804222", "status": "completed"} tags=[]
ensemble["n_clusters"].value_counts().head()

# %% papermill={"duration": 0.088715, "end_time": "2021-01-06T13:18:23.055347", "exception": false, "start_time": "2021-01-06T13:18:22.966632", "status": "completed"} tags=[]
ensemble_stats = ensemble["n_clusters"].describe()
display(ensemble_stats)

# %% [markdown] papermill={"duration": 0.072326, "end_time": "2021-01-06T13:18:23.200013", "exception": false, "start_time": "2021-01-06T13:18:23.127687", "status": "completed"} tags=[]
# ## Testing

# %% papermill={"duration": 0.086219, "end_time": "2021-01-06T13:18:23.357732", "exception": false, "start_time": "2021-01-06T13:18:23.271513", "status": "completed"} tags=[]
assert ensemble_stats["min"] > 1

# %% papermill={"duration": 0.086922, "end_time": "2021-01-06T13:18:23.516766", "exception": false, "start_time": "2021-01-06T13:18:23.429844", "status": "completed"} tags=[]
assert not ensemble["n_clusters"].isna().any()

# %% papermill={"duration": 0.086172, "end_time": "2021-01-06T13:18:23.677297", "exception": false, "start_time": "2021-01-06T13:18:23.591125", "status": "completed"} tags=[]
assert ensemble.shape[0] == len(CLUSTERERS)

# %% papermill={"duration": 0.103686, "end_time": "2021-01-06T13:18:23.853747", "exception": false, "start_time": "2021-01-06T13:18:23.750061", "status": "completed"} tags=[]
# all partitions have the right size
assert np.all(
    [part["partition"].shape[0] == data.shape[0] for idx, part in ensemble.iterrows()]
)

# %% papermill={"duration": 0.110053, "end_time": "2021-01-06T13:18:24.036628", "exception": false, "start_time": "2021-01-06T13:18:23.926575", "status": "completed"} tags=[]
# no partition has negative clusters (noisy points)
assert not np.any([(part["partition"] < 0).any() for idx, part in ensemble.iterrows()])

# %% [markdown] papermill={"duration": 0.071948, "end_time": "2021-01-06T13:18:24.180324", "exception": false, "start_time": "2021-01-06T13:18:24.108376", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.087097, "end_time": "2021-01-06T13:18:24.339326", "exception": false, "start_time": "2021-01-06T13:18:24.252229", "status": "completed"} tags=[]
output_filename = Path(
    RESULTS_DIR,
    generate_result_set_name(
        CLUSTERING_OPTIONS,
        prefix=f"{clustering_method_name}-",
        suffix=".pkl",
    ),
).resolve()
display(output_filename)

# %% papermill={"duration": 0.096482, "end_time": "2021-01-06T13:18:24.508669", "exception": false, "start_time": "2021-01-06T13:18:24.412187", "status": "completed"} tags=[]
ensemble.to_pickle(output_filename)

# %% [markdown] papermill={"duration": 0.071556, "end_time": "2021-01-06T13:18:24.653671", "exception": false, "start_time": "2021-01-06T13:18:24.582115", "status": "completed"} tags=[]
# # Stability

# %% [markdown] papermill={"duration": 0.071251, "end_time": "2021-01-06T13:18:24.796576", "exception": false, "start_time": "2021-01-06T13:18:24.725325", "status": "completed"} tags=[]
# ## Group ensemble by n_clusters

# %% papermill={"duration": 0.105288, "end_time": "2021-01-06T13:18:24.973337", "exception": false, "start_time": "2021-01-06T13:18:24.868049", "status": "completed"} tags=[]
parts = ensemble.groupby("n_clusters").apply(
    lambda x: np.concatenate(x["partition"].apply(lambda x: x.reshape(1, -1)), axis=0)
)

# %% papermill={"duration": 0.097534, "end_time": "2021-01-06T13:18:25.143611", "exception": false, "start_time": "2021-01-06T13:18:25.046077", "status": "completed"} tags=[]
parts.head()

# %% papermill={"duration": 0.088903, "end_time": "2021-01-06T13:18:25.305854", "exception": false, "start_time": "2021-01-06T13:18:25.216951", "status": "completed"} tags=[]
assert np.all(
    [
        parts.loc[k].shape == (CLUSTERING_OPTIONS["N_REPS_PER_K"], data.shape[0])
        for k in parts.index
    ]
)

# %% [markdown] papermill={"duration": 0.072054, "end_time": "2021-01-06T13:18:25.450766", "exception": false, "start_time": "2021-01-06T13:18:25.378712", "status": "completed"} tags=[]
# ## Compute stability

# %% papermill={"duration": 0.08819, "end_time": "2021-01-06T13:18:25.611546", "exception": false, "start_time": "2021-01-06T13:18:25.523356", "status": "completed"} tags=[]
from sklearn.metrics import adjusted_rand_score as ari
from scipy.spatial.distance import squareform, pdist

# %% papermill={"duration": 0.777478, "end_time": "2021-01-06T13:18:26.462326", "exception": false, "start_time": "2021-01-06T13:18:25.684848", "status": "completed"} tags=[]
parts_ari = pd.Series(
    {k: pdist(parts.loc[k], metric=ari) for k in parts.index}, name="k"
)

# %% papermill={"duration": 0.089363, "end_time": "2021-01-06T13:18:26.623873", "exception": false, "start_time": "2021-01-06T13:18:26.534510", "status": "completed"} tags=[]
parts_ari_stability = parts_ari.apply(lambda x: x.mean())
display(parts_ari_stability.sort_values(ascending=False).head(15))

# %% papermill={"duration": 0.088936, "end_time": "2021-01-06T13:18:26.785782", "exception": false, "start_time": "2021-01-06T13:18:26.696846", "status": "completed"} tags=[]
parts_ari_df = pd.DataFrame.from_records(parts_ari.tolist()).set_index(
    parts_ari.index.copy()
)

# %% papermill={"duration": 0.089016, "end_time": "2021-01-06T13:18:26.949109", "exception": false, "start_time": "2021-01-06T13:18:26.860093", "status": "completed"} tags=[]
parts_ari_df.shape

# %% papermill={"duration": 0.087394, "end_time": "2021-01-06T13:18:27.112386", "exception": false, "start_time": "2021-01-06T13:18:27.024992", "status": "completed"} tags=[]
assert (
    int(
        (CLUSTERING_OPTIONS["N_REPS_PER_K"] * (CLUSTERING_OPTIONS["N_REPS_PER_K"] - 1))
        / 2
    )
    == parts_ari_df.shape[1]
)

# %% papermill={"duration": 0.093408, "end_time": "2021-01-06T13:18:27.278597", "exception": false, "start_time": "2021-01-06T13:18:27.185189", "status": "completed"} tags=[]
parts_ari_df.head()

# %% [markdown] papermill={"duration": 0.073016, "end_time": "2021-01-06T13:18:27.425974", "exception": false, "start_time": "2021-01-06T13:18:27.352958", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.091261, "end_time": "2021-01-06T13:18:27.590845", "exception": false, "start_time": "2021-01-06T13:18:27.499584", "status": "completed"} tags=[]
output_filename = Path(
    RESULTS_DIR,
    generate_result_set_name(
        CLUSTERING_OPTIONS,
        prefix=f"{clustering_method_name}-stability-",
        suffix=".pkl",
    ),
).resolve()
display(output_filename)

# %% papermill={"duration": 0.090156, "end_time": "2021-01-06T13:18:27.756894", "exception": false, "start_time": "2021-01-06T13:18:27.666738", "status": "completed"} tags=[]
parts_ari_df.to_pickle(output_filename)

# %% [markdown] papermill={"duration": 0.073719, "end_time": "2021-01-06T13:18:27.905531", "exception": false, "start_time": "2021-01-06T13:18:27.831812", "status": "completed"} tags=[]
# ## Stability plot

# %% papermill={"duration": 0.090629, "end_time": "2021-01-06T13:18:28.072530", "exception": false, "start_time": "2021-01-06T13:18:27.981901", "status": "completed"} tags=[]
parts_ari_df_plot = (
    parts_ari_df.stack()
    .reset_index()
    .rename(columns={"level_0": "k", "level_1": "idx", 0: "ari"})
)

# %% papermill={"duration": 0.090449, "end_time": "2021-01-06T13:18:28.238587", "exception": false, "start_time": "2021-01-06T13:18:28.148138", "status": "completed"} tags=[]
parts_ari_df_plot.dtypes

# %% papermill={"duration": 0.091456, "end_time": "2021-01-06T13:18:28.405291", "exception": false, "start_time": "2021-01-06T13:18:28.313835", "status": "completed"} tags=[]
parts_ari_df_plot.head()

# %% papermill={"duration": 2.946716, "end_time": "2021-01-06T13:18:31.425972", "exception": false, "start_time": "2021-01-06T13:18:28.479256", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.076358, "end_time": "2021-01-06T13:18:31.577809", "exception": false, "start_time": "2021-01-06T13:18:31.501451", "status": "completed"} tags=[]
