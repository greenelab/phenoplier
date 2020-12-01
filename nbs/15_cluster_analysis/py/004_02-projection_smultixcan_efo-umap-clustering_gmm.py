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

# %% [markdown] papermill={"duration": 0.027958, "end_time": "2020-12-02T16:52:23.816602", "exception": false, "start_time": "2020-12-02T16:52:23.788644", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.017652, "end_time": "2020-12-02T16:52:23.852232", "exception": false, "start_time": "2020-12-02T16:52:23.834580", "status": "completed"} tags=[]
# Runs gaussian mixture model on the umap version of the data.

# %% [markdown] papermill={"duration": 0.017282, "end_time": "2020-12-02T16:52:23.887127", "exception": false, "start_time": "2020-12-02T16:52:23.869845", "status": "completed"} tags=[]
# # Environment variables

# %% papermill={"duration": 0.031233, "end_time": "2020-12-02T16:52:23.935889", "exception": false, "start_time": "2020-12-02T16:52:23.904656", "status": "completed"} tags=[]
from IPython.display import display

import conf

N_JOBS = conf.GENERAL['N_JOBS']
display(N_JOBS)

# %% papermill={"duration": 0.02484, "end_time": "2020-12-02T16:52:23.979365", "exception": false, "start_time": "2020-12-02T16:52:23.954525", "status": "completed"} tags=[]
# %env MKL_NUM_THREADS=$N_JOBS
# %env OPEN_BLAS_NUM_THREADS=$N_JOBS
# %env NUMEXPR_NUM_THREADS=$N_JOBS
# %env OMP_NUM_THREADS=$N_JOBS

# %% [markdown] papermill={"duration": 0.018246, "end_time": "2020-12-02T16:52:24.016462", "exception": false, "start_time": "2020-12-02T16:52:23.998216", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.028318, "end_time": "2020-12-02T16:52:24.062849", "exception": false, "start_time": "2020-12-02T16:52:24.034531", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 1.608536, "end_time": "2020-12-02T16:52:25.690085", "exception": false, "start_time": "2020-12-02T16:52:24.081549", "status": "completed"} tags=[]
from pathlib import Path

import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns

from utils import generate_result_set_name

# %% [markdown] papermill={"duration": 0.018292, "end_time": "2020-12-02T16:52:25.729097", "exception": false, "start_time": "2020-12-02T16:52:25.710805", "status": "completed"} tags=[]
# # Settings

# %% papermill={"duration": 0.032019, "end_time": "2020-12-02T16:52:25.778951", "exception": false, "start_time": "2020-12-02T16:52:25.746932", "status": "completed"} tags=[]
INITIAL_RANDOM_STATE = 80000

# %% [markdown] papermill={"duration": 0.020386, "end_time": "2020-12-02T16:52:25.818553", "exception": false, "start_time": "2020-12-02T16:52:25.798167", "status": "completed"} tags=[]
# ## Input data

# %% papermill={"duration": 0.032183, "end_time": "2020-12-02T16:52:25.869240", "exception": false, "start_time": "2020-12-02T16:52:25.837057", "status": "completed"} tags=[]
INPUT_SUBSET = 'umap'

# %% papermill={"duration": 0.032292, "end_time": "2020-12-02T16:52:25.920249", "exception": false, "start_time": "2020-12-02T16:52:25.887957", "status": "completed"} tags=[]
INPUT_STEM = 'z_score_std-projection-smultixcan-efo_partial-mashr-zscores'

# %% papermill={"duration": 0.03286, "end_time": "2020-12-02T16:52:25.972536", "exception": false, "start_time": "2020-12-02T16:52:25.939676", "status": "completed"} tags=[]
DR_OPTIONS = {
    'n_components': 50,
    'metric': 'euclidean',
    'n_neighbors': 15,
    'random_state': 0,
}

# %% papermill={"duration": 0.03511, "end_time": "2020-12-02T16:52:26.027199", "exception": false, "start_time": "2020-12-02T16:52:25.992089", "status": "completed"} tags=[]
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

# %% [markdown] papermill={"duration": 0.018877, "end_time": "2020-12-02T16:52:26.065859", "exception": false, "start_time": "2020-12-02T16:52:26.046982", "status": "completed"} tags=[]
# ## Clustering

# %% papermill={"duration": 0.040341, "end_time": "2020-12-02T16:52:26.124952", "exception": false, "start_time": "2020-12-02T16:52:26.084611", "status": "completed"} tags=[]
from sklearn.mixture import GaussianMixture

# %% papermill={"duration": 0.033843, "end_time": "2020-12-02T16:52:26.178892", "exception": false, "start_time": "2020-12-02T16:52:26.145049", "status": "completed"} tags=[]
CLUSTERING_ATTRIBUTES_TO_SAVE = ['n_clusters']

# %% papermill={"duration": 0.034859, "end_time": "2020-12-02T16:52:26.234445", "exception": false, "start_time": "2020-12-02T16:52:26.199586", "status": "completed"} tags=[]
CLUSTERING_OPTIONS = {}

CLUSTERING_OPTIONS['K_MIN'] = 2
CLUSTERING_OPTIONS['K_MAX'] = 60 # sqrt(3749)
CLUSTERING_OPTIONS['N_REPS_PER_K'] = 5
CLUSTERING_OPTIONS['N_INIT'] = 10
CLUSTERING_OPTIONS['COVARIANCE_TYPE'] = 'full'

display(CLUSTERING_OPTIONS)

# %% papermill={"duration": 0.035585, "end_time": "2020-12-02T16:52:26.290357", "exception": false, "start_time": "2020-12-02T16:52:26.254772", "status": "completed"} tags=[]
CLUSTERERS = {}

idx = 0
random_state = INITIAL_RANDOM_STATE

for k in range(CLUSTERING_OPTIONS['K_MIN'], CLUSTERING_OPTIONS['K_MAX']+1):
    for i in range(CLUSTERING_OPTIONS['N_REPS_PER_K']):
        clus = GaussianMixture(
                n_components=k,
                n_init=CLUSTERING_OPTIONS['N_INIT'],
                covariance_type=CLUSTERING_OPTIONS['COVARIANCE_TYPE'],
                random_state=random_state,
            )
        
        method_name = type(clus).__name__
        CLUSTERERS[f'{method_name} #{idx}'] = clus
        
        random_state = random_state + 1
        idx = idx + 1

# %% papermill={"duration": 0.034584, "end_time": "2020-12-02T16:52:26.344796", "exception": false, "start_time": "2020-12-02T16:52:26.310212", "status": "completed"} tags=[]
display(len(CLUSTERERS))

# %% papermill={"duration": 0.037027, "end_time": "2020-12-02T16:52:26.402302", "exception": false, "start_time": "2020-12-02T16:52:26.365275", "status": "completed"} tags=[]
_iter = iter(CLUSTERERS.items())
display(next(_iter))
display(next(_iter))

# %% papermill={"duration": 0.036233, "end_time": "2020-12-02T16:52:26.460010", "exception": false, "start_time": "2020-12-02T16:52:26.423777", "status": "completed"} tags=[]
clustering_method_name = method_name
display(clustering_method_name)

# %% [markdown] papermill={"duration": 0.020877, "end_time": "2020-12-02T16:52:26.502904", "exception": false, "start_time": "2020-12-02T16:52:26.482027", "status": "completed"} tags=[]
# ## Output directory

# %% papermill={"duration": 0.035691, "end_time": "2020-12-02T16:52:26.559241", "exception": false, "start_time": "2020-12-02T16:52:26.523550", "status": "completed"} tags=[]
# output dir for this notebook
RESULTS_DIR = Path(
    conf.RESULTS["CLUSTERING_RUNS_DIR"],
    f'{INPUT_SUBSET}-{INPUT_STEM}',
).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %% [markdown] papermill={"duration": 0.020811, "end_time": "2020-12-02T16:52:26.601012", "exception": false, "start_time": "2020-12-02T16:52:26.580201", "status": "completed"} tags=[]
# # Load input file

# %% papermill={"duration": 0.036953, "end_time": "2020-12-02T16:52:26.658392", "exception": false, "start_time": "2020-12-02T16:52:26.621439", "status": "completed"} tags=[]
data = pd.read_pickle(input_filepath)

# %% papermill={"duration": 0.035818, "end_time": "2020-12-02T16:52:26.715333", "exception": false, "start_time": "2020-12-02T16:52:26.679515", "status": "completed"} tags=[]
data.shape

# %% papermill={"duration": 0.048714, "end_time": "2020-12-02T16:52:26.785464", "exception": false, "start_time": "2020-12-02T16:52:26.736750", "status": "completed"} tags=[]
data.head()

# %% papermill={"duration": 0.038357, "end_time": "2020-12-02T16:52:26.846029", "exception": false, "start_time": "2020-12-02T16:52:26.807672", "status": "completed"} tags=[]
assert not data.isna().any().any()

# %% [markdown] papermill={"duration": 0.021142, "end_time": "2020-12-02T16:52:26.890492", "exception": false, "start_time": "2020-12-02T16:52:26.869350", "status": "completed"} tags=[]
# # Clustering

# %% [markdown] papermill={"duration": 0.021082, "end_time": "2020-12-02T16:52:26.933180", "exception": false, "start_time": "2020-12-02T16:52:26.912098", "status": "completed"} tags=[]
# ## Generate ensemble

# %% papermill={"duration": 0.037908, "end_time": "2020-12-02T16:52:26.992298", "exception": false, "start_time": "2020-12-02T16:52:26.954390", "status": "completed"} tags=[]
from clustering.ensemble import generate_ensemble

# %% papermill={"duration": 3018.004788, "end_time": "2020-12-02T17:42:45.018829", "exception": false, "start_time": "2020-12-02T16:52:27.014041", "status": "completed"} tags=[]
ensemble = generate_ensemble(
    data,
    CLUSTERERS,
    attributes=CLUSTERING_ATTRIBUTES_TO_SAVE,
)

# %% papermill={"duration": 0.083356, "end_time": "2020-12-02T17:42:45.169989", "exception": false, "start_time": "2020-12-02T17:42:45.086633", "status": "completed"} tags=[]
# the number should be close to 295 (the number of partitions generated by k-means/spectral clustering)
ensemble.shape

# %% papermill={"duration": 0.09153, "end_time": "2020-12-02T17:42:45.330754", "exception": false, "start_time": "2020-12-02T17:42:45.239224", "status": "completed"} tags=[]
ensemble.head()

# %% papermill={"duration": 0.083812, "end_time": "2020-12-02T17:42:45.483765", "exception": false, "start_time": "2020-12-02T17:42:45.399953", "status": "completed"} tags=[]
ensemble['n_clusters'].value_counts().head()

# %% papermill={"duration": 0.085768, "end_time": "2020-12-02T17:42:45.637586", "exception": false, "start_time": "2020-12-02T17:42:45.551818", "status": "completed"} tags=[]
ensemble_stats = ensemble['n_clusters'].describe()
display(ensemble_stats)

# %% [markdown] papermill={"duration": 0.068029, "end_time": "2020-12-02T17:42:45.774931", "exception": false, "start_time": "2020-12-02T17:42:45.706902", "status": "completed"} tags=[]
# ## Testing

# %% papermill={"duration": 0.090963, "end_time": "2020-12-02T17:42:45.953899", "exception": false, "start_time": "2020-12-02T17:42:45.862936", "status": "completed"} tags=[]
assert ensemble_stats['min'] > 1

# %% papermill={"duration": 0.083582, "end_time": "2020-12-02T17:42:46.107074", "exception": false, "start_time": "2020-12-02T17:42:46.023492", "status": "completed"} tags=[]
assert not ensemble['n_clusters'].isna().any()

# %% papermill={"duration": 0.083859, "end_time": "2020-12-02T17:42:46.262551", "exception": false, "start_time": "2020-12-02T17:42:46.178692", "status": "completed"} tags=[]
assert ensemble.shape[0] == len(CLUSTERERS)

# %% papermill={"duration": 0.100831, "end_time": "2020-12-02T17:42:46.434113", "exception": false, "start_time": "2020-12-02T17:42:46.333282", "status": "completed"} tags=[]
# all partitions have the right size
assert np.all([
    part['partition'].shape[0] == data.shape[0]
    for idx, part in ensemble.iterrows()
])

# %% papermill={"duration": 0.104514, "end_time": "2020-12-02T17:42:46.609152", "exception": false, "start_time": "2020-12-02T17:42:46.504638", "status": "completed"} tags=[]
# no partition has negative clusters (noisy points)
assert not np.any([
    (part['partition'] < 0).any()
    for idx, part in ensemble.iterrows()
])

# %% [markdown] papermill={"duration": 0.068169, "end_time": "2020-12-02T17:42:46.747387", "exception": false, "start_time": "2020-12-02T17:42:46.679218", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.088373, "end_time": "2020-12-02T17:42:46.904862", "exception": false, "start_time": "2020-12-02T17:42:46.816489", "status": "completed"} tags=[]
output_filename = Path(
    RESULTS_DIR,
    generate_result_set_name(
        CLUSTERING_OPTIONS,
        prefix=f'{clustering_method_name}-',
        suffix='.pkl',
    )
).resolve()
display(output_filename)

# %% papermill={"duration": 0.095494, "end_time": "2020-12-02T17:42:47.071275", "exception": false, "start_time": "2020-12-02T17:42:46.975781", "status": "completed"} tags=[]
ensemble.to_pickle(output_filename)

# %% [markdown] papermill={"duration": 0.069206, "end_time": "2020-12-02T17:42:47.212244", "exception": false, "start_time": "2020-12-02T17:42:47.143038", "status": "completed"} tags=[]
# # Stability

# %% [markdown] papermill={"duration": 0.070092, "end_time": "2020-12-02T17:42:47.352054", "exception": false, "start_time": "2020-12-02T17:42:47.281962", "status": "completed"} tags=[]
# ## Group ensemble by n_clusters

# %% papermill={"duration": 0.102152, "end_time": "2020-12-02T17:42:47.525423", "exception": false, "start_time": "2020-12-02T17:42:47.423271", "status": "completed"} tags=[]
parts = ensemble.groupby('n_clusters').apply(lambda x: np.concatenate(x['partition'].apply(lambda x: x.reshape(1, -1)), axis=0))

# %% papermill={"duration": 0.096042, "end_time": "2020-12-02T17:42:47.691975", "exception": false, "start_time": "2020-12-02T17:42:47.595933", "status": "completed"} tags=[]
parts.head()

# %% papermill={"duration": 0.085423, "end_time": "2020-12-02T17:42:47.847347", "exception": false, "start_time": "2020-12-02T17:42:47.761924", "status": "completed"} tags=[]
assert np.all([
    parts.loc[k].shape == (CLUSTERING_OPTIONS['N_REPS_PER_K'], data.shape[0])
    for k in parts.index
])

# %% [markdown] papermill={"duration": 0.070475, "end_time": "2020-12-02T17:42:47.991924", "exception": false, "start_time": "2020-12-02T17:42:47.921449", "status": "completed"} tags=[]
# ## Compute stability

# %% papermill={"duration": 0.084984, "end_time": "2020-12-02T17:42:48.147709", "exception": false, "start_time": "2020-12-02T17:42:48.062725", "status": "completed"} tags=[]
from sklearn.metrics import adjusted_rand_score as ari
from scipy.spatial.distance import squareform, pdist

# %% papermill={"duration": 0.776181, "end_time": "2020-12-02T17:42:48.995236", "exception": false, "start_time": "2020-12-02T17:42:48.219055", "status": "completed"} tags=[]
parts_ari = pd.Series(
    {k: pdist(parts.loc[k], metric=ari) for k in parts.index},
    name='k'
)

# %% papermill={"duration": 0.087149, "end_time": "2020-12-02T17:42:49.153545", "exception": false, "start_time": "2020-12-02T17:42:49.066396", "status": "completed"} tags=[]
parts_ari_stability = parts_ari.apply(lambda x: x.mean())
display(parts_ari_stability.sort_values(ascending=False).head(15))

# %% papermill={"duration": 0.086713, "end_time": "2020-12-02T17:42:49.310884", "exception": false, "start_time": "2020-12-02T17:42:49.224171", "status": "completed"} tags=[]
parts_ari_df = pd.DataFrame.from_records(parts_ari.tolist()).set_index(parts_ari.index.copy())

# %% papermill={"duration": 0.085308, "end_time": "2020-12-02T17:42:49.469567", "exception": false, "start_time": "2020-12-02T17:42:49.384259", "status": "completed"} tags=[]
parts_ari_df.shape

# %% papermill={"duration": 0.086193, "end_time": "2020-12-02T17:42:49.629172", "exception": false, "start_time": "2020-12-02T17:42:49.542979", "status": "completed"} tags=[]
assert int( (CLUSTERING_OPTIONS['N_REPS_PER_K'] * (CLUSTERING_OPTIONS['N_REPS_PER_K'] - 1) ) / 2) == parts_ari_df.shape[1]

# %% papermill={"duration": 0.091127, "end_time": "2020-12-02T17:42:49.793252", "exception": false, "start_time": "2020-12-02T17:42:49.702125", "status": "completed"} tags=[]
parts_ari_df.head()

# %% [markdown] papermill={"duration": 0.074565, "end_time": "2020-12-02T17:42:49.940325", "exception": false, "start_time": "2020-12-02T17:42:49.865760", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.087322, "end_time": "2020-12-02T17:42:50.099052", "exception": false, "start_time": "2020-12-02T17:42:50.011730", "status": "completed"} tags=[]
output_filename = Path(
    RESULTS_DIR,
    generate_result_set_name(
        CLUSTERING_OPTIONS,
        prefix=f'{clustering_method_name}-stability-',
        suffix='.pkl',
    )
).resolve()
display(output_filename)

# %% papermill={"duration": 0.085821, "end_time": "2020-12-02T17:42:50.256271", "exception": false, "start_time": "2020-12-02T17:42:50.170450", "status": "completed"} tags=[]
parts_ari_df.to_pickle(output_filename)

# %% [markdown] papermill={"duration": 0.071862, "end_time": "2020-12-02T17:42:50.400311", "exception": false, "start_time": "2020-12-02T17:42:50.328449", "status": "completed"} tags=[]
# ## Stability plot

# %% papermill={"duration": 0.087971, "end_time": "2020-12-02T17:42:50.560200", "exception": false, "start_time": "2020-12-02T17:42:50.472229", "status": "completed"} tags=[]
parts_ari_df_plot = parts_ari_df.stack().reset_index().rename(columns={'level_0': 'k', 'level_1': 'idx', 0: 'ari'})

# %% papermill={"duration": 0.087384, "end_time": "2020-12-02T17:42:50.719239", "exception": false, "start_time": "2020-12-02T17:42:50.631855", "status": "completed"} tags=[]
parts_ari_df_plot.dtypes

# %% papermill={"duration": 0.091015, "end_time": "2020-12-02T17:42:50.882560", "exception": false, "start_time": "2020-12-02T17:42:50.791545", "status": "completed"} tags=[]
parts_ari_df_plot.head()

# %% papermill={"duration": 2.974325, "end_time": "2020-12-02T17:42:53.932062", "exception": false, "start_time": "2020-12-02T17:42:50.957737", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.073868, "end_time": "2020-12-02T17:42:54.081639", "exception": false, "start_time": "2020-12-02T17:42:54.007771", "status": "completed"} tags=[]
