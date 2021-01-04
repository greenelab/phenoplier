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

# %% [markdown] papermill={"duration": 0.053854, "end_time": "2020-12-02T21:47:45.286086", "exception": false, "start_time": "2020-12-02T21:47:45.232232", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.017814, "end_time": "2020-12-02T21:47:45.323082", "exception": false, "start_time": "2020-12-02T21:47:45.305268", "status": "completed"} tags=[]
# Runs spectral clustering on the umap version of the data.

# %% [markdown] papermill={"duration": 0.01763, "end_time": "2020-12-02T21:47:45.358197", "exception": false, "start_time": "2020-12-02T21:47:45.340567", "status": "completed"} tags=[]
# # Environment variables

# %% papermill={"duration": 0.031617, "end_time": "2020-12-02T21:47:45.407466", "exception": false, "start_time": "2020-12-02T21:47:45.375849", "status": "completed"} tags=[]
from IPython.display import display

import conf

N_JOBS = conf.GENERAL['N_JOBS']
display(N_JOBS)

# %% papermill={"duration": 0.024947, "end_time": "2020-12-02T21:47:45.451298", "exception": false, "start_time": "2020-12-02T21:47:45.426351", "status": "completed"} tags=[]
# %env MKL_NUM_THREADS=$N_JOBS
# %env OPEN_BLAS_NUM_THREADS=$N_JOBS
# %env NUMEXPR_NUM_THREADS=$N_JOBS
# %env OMP_NUM_THREADS=$N_JOBS

# %% [markdown] papermill={"duration": 0.019795, "end_time": "2020-12-02T21:47:45.490104", "exception": false, "start_time": "2020-12-02T21:47:45.470309", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.029549, "end_time": "2020-12-02T21:47:45.539223", "exception": false, "start_time": "2020-12-02T21:47:45.509674", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 1.613247, "end_time": "2020-12-02T21:47:47.171428", "exception": false, "start_time": "2020-12-02T21:47:45.558181", "status": "completed"} tags=[]
from pathlib import Path

import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns

from utils import generate_result_set_name

# %% [markdown] papermill={"duration": 0.018844, "end_time": "2020-12-02T21:47:47.211338", "exception": false, "start_time": "2020-12-02T21:47:47.192494", "status": "completed"} tags=[]
# # Settings

# %% papermill={"duration": 0.032903, "end_time": "2020-12-02T21:47:47.262755", "exception": false, "start_time": "2020-12-02T21:47:47.229852", "status": "completed"} tags=[]
INITIAL_RANDOM_STATE = 50000

# %% [markdown] papermill={"duration": 0.018441, "end_time": "2020-12-02T21:47:47.299949", "exception": false, "start_time": "2020-12-02T21:47:47.281508", "status": "completed"} tags=[]
# ## Input data

# %% papermill={"duration": 0.032809, "end_time": "2020-12-02T21:47:47.351152", "exception": false, "start_time": "2020-12-02T21:47:47.318343", "status": "completed"} tags=[]
INPUT_SUBSET = 'umap'

# %% papermill={"duration": 0.032961, "end_time": "2020-12-02T21:47:47.403479", "exception": false, "start_time": "2020-12-02T21:47:47.370518", "status": "completed"} tags=[]
INPUT_STEM = 'z_score_std-projection-smultixcan-efo_partial-mashr-zscores'

# %% papermill={"duration": 0.032874, "end_time": "2020-12-02T21:47:47.455892", "exception": false, "start_time": "2020-12-02T21:47:47.423018", "status": "completed"} tags=[]
DR_OPTIONS = {
    'n_components': 50,
    'metric': 'euclidean',
    'n_neighbors': 15,
    'random_state': 0,
}

# %% papermill={"duration": 0.036983, "end_time": "2020-12-02T21:47:47.512291", "exception": false, "start_time": "2020-12-02T21:47:47.475308", "status": "completed"} tags=[]
input_filepath = Path(
    conf.RESULTS["DATA_TRANSFORMATIONS_DIR"],
    INPUT_SUBSET,
    generate_result_set_name(
        DR_OPTIONS,
        prefix=f'{INPUT_SUBSET}-{INPUT_STEM}-',
        suffix='.pkl'
    )
).resolve()
display(input_filepath)

assert input_filepath.exists(), 'Input file does not exist'

input_filepath_stem = input_filepath.stem
display(input_filepath_stem)

# %% [markdown] papermill={"duration": 0.019013, "end_time": "2020-12-02T21:47:47.551759", "exception": false, "start_time": "2020-12-02T21:47:47.532746", "status": "completed"} tags=[]
# ## Clustering

# %% papermill={"duration": 0.039294, "end_time": "2020-12-02T21:47:47.610098", "exception": false, "start_time": "2020-12-02T21:47:47.570804", "status": "completed"} tags=[]
from sklearn.cluster import SpectralClustering

# %% papermill={"duration": 0.033947, "end_time": "2020-12-02T21:47:47.664302", "exception": false, "start_time": "2020-12-02T21:47:47.630355", "status": "completed"} tags=[]
CLUSTERING_ATTRIBUTES_TO_SAVE = ['n_clusters']

# %% papermill={"duration": 0.034627, "end_time": "2020-12-02T21:47:47.718952", "exception": false, "start_time": "2020-12-02T21:47:47.684325", "status": "completed"} tags=[]
CLUSTERING_OPTIONS = {}

CLUSTERING_OPTIONS['K_MIN'] = 2
CLUSTERING_OPTIONS['K_MAX'] = 60 # sqrt(3749)
CLUSTERING_OPTIONS['N_REPS_PER_K'] = 5
CLUSTERING_OPTIONS['KMEANS_N_INIT'] = 10
CLUSTERING_OPTIONS['N_NEIGHBORS'] = None
CLUSTERING_OPTIONS['AFFINITY'] = 'rbf' # nearest neighbors does not work well with umap

display(CLUSTERING_OPTIONS)

# %% papermill={"duration": 0.035661, "end_time": "2020-12-02T21:47:47.774558", "exception": false, "start_time": "2020-12-02T21:47:47.738897", "status": "completed"} tags=[]
CLUSTERERS = {}

idx = 0
random_state = INITIAL_RANDOM_STATE

for k in range(CLUSTERING_OPTIONS['K_MIN'], CLUSTERING_OPTIONS['K_MAX']+1):
    for i in range(CLUSTERING_OPTIONS['N_REPS_PER_K']):
        clus = SpectralClustering(
                n_clusters=k,
                n_init=CLUSTERING_OPTIONS['KMEANS_N_INIT'],
                affinity=CLUSTERING_OPTIONS['AFFINITY'],
                n_neighbors=CLUSTERING_OPTIONS['N_NEIGHBORS'],
                random_state=random_state,
            )
        
        method_name = type(clus).__name__
        CLUSTERERS[f'{method_name} #{idx}'] = clus
        
        random_state = random_state + 1
        idx = idx + 1

# %% papermill={"duration": 0.034866, "end_time": "2020-12-02T21:47:47.829976", "exception": false, "start_time": "2020-12-02T21:47:47.795110", "status": "completed"} tags=[]
display(len(CLUSTERERS))

# %% papermill={"duration": 0.037685, "end_time": "2020-12-02T21:47:47.888582", "exception": false, "start_time": "2020-12-02T21:47:47.850897", "status": "completed"} tags=[]
_iter = iter(CLUSTERERS.items())
display(next(_iter))
display(next(_iter))

# %% papermill={"duration": 0.035137, "end_time": "2020-12-02T21:47:47.945325", "exception": false, "start_time": "2020-12-02T21:47:47.910188", "status": "completed"} tags=[]
clustering_method_name = method_name
display(clustering_method_name)

# %% [markdown] papermill={"duration": 0.020518, "end_time": "2020-12-02T21:47:47.987286", "exception": false, "start_time": "2020-12-02T21:47:47.966768", "status": "completed"} tags=[]
# ## Output directory

# %% papermill={"duration": 0.035431, "end_time": "2020-12-02T21:47:48.043077", "exception": false, "start_time": "2020-12-02T21:47:48.007646", "status": "completed"} tags=[]
# output dir for this notebook
RESULTS_DIR = Path(
    conf.RESULTS["CLUSTERING_RUNS_DIR"],
    f'{INPUT_SUBSET}-{INPUT_STEM}',
).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %% [markdown] papermill={"duration": 0.020924, "end_time": "2020-12-02T21:47:48.085917", "exception": false, "start_time": "2020-12-02T21:47:48.064993", "status": "completed"} tags=[]
# # Load input file

# %% papermill={"duration": 0.036154, "end_time": "2020-12-02T21:47:48.142916", "exception": false, "start_time": "2020-12-02T21:47:48.106762", "status": "completed"} tags=[]
data = pd.read_pickle(input_filepath)

# %% papermill={"duration": 0.036472, "end_time": "2020-12-02T21:47:48.201556", "exception": false, "start_time": "2020-12-02T21:47:48.165084", "status": "completed"} tags=[]
data.shape

# %% papermill={"duration": 0.048883, "end_time": "2020-12-02T21:47:48.272279", "exception": false, "start_time": "2020-12-02T21:47:48.223396", "status": "completed"} tags=[]
data.head()

# %% papermill={"duration": 0.036781, "end_time": "2020-12-02T21:47:48.330869", "exception": false, "start_time": "2020-12-02T21:47:48.294088", "status": "completed"} tags=[]
assert not data.isna().any().any()

# %% [markdown] papermill={"duration": 0.021327, "end_time": "2020-12-02T21:47:48.374437", "exception": false, "start_time": "2020-12-02T21:47:48.353110", "status": "completed"} tags=[]
# # Clustering

# %% [markdown] papermill={"duration": 0.021954, "end_time": "2020-12-02T21:47:48.417747", "exception": false, "start_time": "2020-12-02T21:47:48.395793", "status": "completed"} tags=[]
# ## Generate ensemble

# %% papermill={"duration": 0.038137, "end_time": "2020-12-02T21:47:48.477355", "exception": false, "start_time": "2020-12-02T21:47:48.439218", "status": "completed"} tags=[]
from clustering.ensemble import generate_ensemble

# %% papermill={"duration": 786.262319, "end_time": "2020-12-02T22:00:54.762844", "exception": false, "start_time": "2020-12-02T21:47:48.500525", "status": "completed"} tags=[]
ensemble = generate_ensemble(
    data,
    CLUSTERERS,
    attributes=CLUSTERING_ATTRIBUTES_TO_SAVE,
)

# %% papermill={"duration": 0.08616, "end_time": "2020-12-02T22:00:54.918787", "exception": false, "start_time": "2020-12-02T22:00:54.832627", "status": "completed"} tags=[]
# the number should be close to 295 (the number of partitions generated by k-means/spectral clustering)
ensemble.shape

# %% papermill={"duration": 0.092234, "end_time": "2020-12-02T22:00:55.106830", "exception": false, "start_time": "2020-12-02T22:00:55.014596", "status": "completed"} tags=[]
ensemble.head()

# %% papermill={"duration": 0.086274, "end_time": "2020-12-02T22:00:55.263705", "exception": false, "start_time": "2020-12-02T22:00:55.177431", "status": "completed"} tags=[]
ensemble['n_clusters'].value_counts().head()

# %% papermill={"duration": 0.086904, "end_time": "2020-12-02T22:00:55.421418", "exception": false, "start_time": "2020-12-02T22:00:55.334514", "status": "completed"} tags=[]
ensemble_stats = ensemble['n_clusters'].describe()
display(ensemble_stats)

# %% [markdown] papermill={"duration": 0.070422, "end_time": "2020-12-02T22:00:55.561798", "exception": false, "start_time": "2020-12-02T22:00:55.491376", "status": "completed"} tags=[]
# ## Testing

# %% papermill={"duration": 0.087008, "end_time": "2020-12-02T22:00:55.718710", "exception": false, "start_time": "2020-12-02T22:00:55.631702", "status": "completed"} tags=[]
assert ensemble_stats['min'] > 1

# %% papermill={"duration": 0.085194, "end_time": "2020-12-02T22:00:55.875328", "exception": false, "start_time": "2020-12-02T22:00:55.790134", "status": "completed"} tags=[]
assert not ensemble['n_clusters'].isna().any()

# %% papermill={"duration": 0.084767, "end_time": "2020-12-02T22:00:56.032205", "exception": false, "start_time": "2020-12-02T22:00:55.947438", "status": "completed"} tags=[]
assert ensemble.shape[0] == len(CLUSTERERS)

# %% papermill={"duration": 0.102963, "end_time": "2020-12-02T22:00:56.205462", "exception": false, "start_time": "2020-12-02T22:00:56.102499", "status": "completed"} tags=[]
# all partitions have the right size
assert np.all([
    part['partition'].shape[0] == data.shape[0]
    for idx, part in ensemble.iterrows()
])

# %% papermill={"duration": 0.105519, "end_time": "2020-12-02T22:00:56.381843", "exception": false, "start_time": "2020-12-02T22:00:56.276324", "status": "completed"} tags=[]
# no partition has negative clusters (noisy points)
assert not np.any([
    (part['partition'] < 0).any()
    for idx, part in ensemble.iterrows()
])

# %% [markdown] papermill={"duration": 0.069816, "end_time": "2020-12-02T22:00:56.522300", "exception": false, "start_time": "2020-12-02T22:00:56.452484", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.08759, "end_time": "2020-12-02T22:00:56.679799", "exception": false, "start_time": "2020-12-02T22:00:56.592209", "status": "completed"} tags=[]
output_filename = Path(
    RESULTS_DIR,
    generate_result_set_name(
        CLUSTERING_OPTIONS,
        prefix=f'{clustering_method_name}-',
        suffix='.pkl',
    )
).resolve()
display(output_filename)

# %% papermill={"duration": 0.091781, "end_time": "2020-12-02T22:00:56.843115", "exception": false, "start_time": "2020-12-02T22:00:56.751334", "status": "completed"} tags=[]
ensemble.to_pickle(output_filename)

# %% [markdown] papermill={"duration": 0.070318, "end_time": "2020-12-02T22:00:56.985893", "exception": false, "start_time": "2020-12-02T22:00:56.915575", "status": "completed"} tags=[]
# # Stability

# %% [markdown] papermill={"duration": 0.07092, "end_time": "2020-12-02T22:00:57.126992", "exception": false, "start_time": "2020-12-02T22:00:57.056072", "status": "completed"} tags=[]
# ## Group ensemble by n_clusters

# %% papermill={"duration": 0.099557, "end_time": "2020-12-02T22:00:57.297576", "exception": false, "start_time": "2020-12-02T22:00:57.198019", "status": "completed"} tags=[]
parts = ensemble.groupby('n_clusters').apply(lambda x: np.concatenate(x['partition'].apply(lambda x: x.reshape(1, -1)), axis=0))

# %% papermill={"duration": 0.097636, "end_time": "2020-12-02T22:00:57.467416", "exception": false, "start_time": "2020-12-02T22:00:57.369780", "status": "completed"} tags=[]
parts.head()

# %% papermill={"duration": 0.085621, "end_time": "2020-12-02T22:00:57.625003", "exception": false, "start_time": "2020-12-02T22:00:57.539382", "status": "completed"} tags=[]
assert np.all([
    parts.loc[k].shape == (CLUSTERING_OPTIONS['N_REPS_PER_K'], data.shape[0])
    for k in parts.index
])

# %% [markdown] papermill={"duration": 0.070766, "end_time": "2020-12-02T22:00:57.769070", "exception": false, "start_time": "2020-12-02T22:00:57.698304", "status": "completed"} tags=[]
# ## Compute stability

# %% papermill={"duration": 0.086882, "end_time": "2020-12-02T22:00:57.926604", "exception": false, "start_time": "2020-12-02T22:00:57.839722", "status": "completed"} tags=[]
from sklearn.metrics import adjusted_rand_score as ari
from scipy.spatial.distance import squareform, pdist

# %% papermill={"duration": 0.673624, "end_time": "2020-12-02T22:00:58.671674", "exception": false, "start_time": "2020-12-02T22:00:57.998050", "status": "completed"} tags=[]
parts_ari = pd.Series(
    {k: pdist(parts.loc[k], metric=ari) for k in parts.index},
    name='k'
)

# %% papermill={"duration": 0.087605, "end_time": "2020-12-02T22:00:58.830626", "exception": false, "start_time": "2020-12-02T22:00:58.743021", "status": "completed"} tags=[]
parts_ari_stability = parts_ari.apply(lambda x: x.mean())
display(parts_ari_stability.sort_values(ascending=False).head(15))

# %% papermill={"duration": 0.087909, "end_time": "2020-12-02T22:00:58.991443", "exception": false, "start_time": "2020-12-02T22:00:58.903534", "status": "completed"} tags=[]
parts_ari_df = pd.DataFrame.from_records(parts_ari.tolist()).set_index(parts_ari.index.copy())

# %% papermill={"duration": 0.086253, "end_time": "2020-12-02T22:00:59.150370", "exception": false, "start_time": "2020-12-02T22:00:59.064117", "status": "completed"} tags=[]
parts_ari_df.shape

# %% papermill={"duration": 0.086208, "end_time": "2020-12-02T22:00:59.308880", "exception": false, "start_time": "2020-12-02T22:00:59.222672", "status": "completed"} tags=[]
assert int( (CLUSTERING_OPTIONS['N_REPS_PER_K'] * (CLUSTERING_OPTIONS['N_REPS_PER_K'] - 1) ) / 2) == parts_ari_df.shape[1]

# %% papermill={"duration": 0.092194, "end_time": "2020-12-02T22:00:59.472186", "exception": false, "start_time": "2020-12-02T22:00:59.379992", "status": "completed"} tags=[]
parts_ari_df.head()

# %% [markdown] papermill={"duration": 0.070819, "end_time": "2020-12-02T22:00:59.615699", "exception": false, "start_time": "2020-12-02T22:00:59.544880", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.089826, "end_time": "2020-12-02T22:00:59.779937", "exception": false, "start_time": "2020-12-02T22:00:59.690111", "status": "completed"} tags=[]
output_filename = Path(
    RESULTS_DIR,
    generate_result_set_name(
        CLUSTERING_OPTIONS,
        prefix=f'{clustering_method_name}-stability-',
        suffix='.pkl',
    )
).resolve()
display(output_filename)

# %% papermill={"duration": 0.088262, "end_time": "2020-12-02T22:00:59.940677", "exception": false, "start_time": "2020-12-02T22:00:59.852415", "status": "completed"} tags=[]
parts_ari_df.to_pickle(output_filename)

# %% [markdown] papermill={"duration": 0.071719, "end_time": "2020-12-02T22:01:00.086321", "exception": false, "start_time": "2020-12-02T22:01:00.014602", "status": "completed"} tags=[]
# ## Stability plot

# %% papermill={"duration": 0.089332, "end_time": "2020-12-02T22:01:00.247195", "exception": false, "start_time": "2020-12-02T22:01:00.157863", "status": "completed"} tags=[]
parts_ari_df_plot = parts_ari_df.stack().reset_index().rename(columns={'level_0': 'k', 'level_1': 'idx', 0: 'ari'})

# %% papermill={"duration": 0.08874, "end_time": "2020-12-02T22:01:00.408746", "exception": false, "start_time": "2020-12-02T22:01:00.320006", "status": "completed"} tags=[]
parts_ari_df_plot.dtypes

# %% papermill={"duration": 0.090084, "end_time": "2020-12-02T22:01:00.571884", "exception": false, "start_time": "2020-12-02T22:01:00.481800", "status": "completed"} tags=[]
parts_ari_df_plot.head()

# %% papermill={"duration": 2.957284, "end_time": "2020-12-02T22:01:03.603077", "exception": false, "start_time": "2020-12-02T22:01:00.645793", "status": "completed"} tags=[]
# with sns.axes_style('whitegrid', {'grid.linestyle': '--'}):
with sns.plotting_context('talk', font_scale=0.75), sns.axes_style('whitegrid', {'grid.linestyle': '--'}):
    fig = plt.figure(figsize=(12, 6))
    ax = sns.pointplot(data=parts_ari_df_plot, x='k', y='ari')
    ax.set_ylabel('Averange ARI')
    ax.set_xlabel('Number of clusters ($k$)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
#     ax.set_ylim(0.0, 1.0)
#     ax.set_xlim(CLUSTERING_OPTIONS['K_MIN'], CLUSTERING_OPTIONS['K_MAX'])
    plt.grid(True)
    plt.tight_layout()

# %% papermill={"duration": 0.076538, "end_time": "2020-12-02T22:01:03.756225", "exception": false, "start_time": "2020-12-02T22:01:03.679687", "status": "completed"} tags=[]
