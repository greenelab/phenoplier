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

# %% [markdown] papermill={"duration": 0.05412, "end_time": "2020-12-02T13:55:53.827527", "exception": false, "start_time": "2020-12-02T13:55:53.773407", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.017442, "end_time": "2020-12-02T13:55:53.863947", "exception": false, "start_time": "2020-12-02T13:55:53.846505", "status": "completed"} tags=[]
# Runs gaussian mixture model on the pca version of the data.

# %% [markdown] papermill={"duration": 0.01744, "end_time": "2020-12-02T13:55:53.899056", "exception": false, "start_time": "2020-12-02T13:55:53.881616", "status": "completed"} tags=[]
# # Environment variables

# %% papermill={"duration": 0.031661, "end_time": "2020-12-02T13:55:53.948184", "exception": false, "start_time": "2020-12-02T13:55:53.916523", "status": "completed"} tags=[]
from IPython.display import display

import conf

N_JOBS = conf.GENERAL['N_JOBS']
display(N_JOBS)

# %% papermill={"duration": 0.025128, "end_time": "2020-12-02T13:55:53.992926", "exception": false, "start_time": "2020-12-02T13:55:53.967798", "status": "completed"} tags=[]
# %env MKL_NUM_THREADS=$N_JOBS
# %env OPEN_BLAS_NUM_THREADS=$N_JOBS
# %env NUMEXPR_NUM_THREADS=$N_JOBS
# %env OMP_NUM_THREADS=$N_JOBS

# %% [markdown] papermill={"duration": 0.018641, "end_time": "2020-12-02T13:55:54.030880", "exception": false, "start_time": "2020-12-02T13:55:54.012239", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.028756, "end_time": "2020-12-02T13:55:54.077878", "exception": false, "start_time": "2020-12-02T13:55:54.049122", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 1.612403, "end_time": "2020-12-02T13:55:55.708919", "exception": false, "start_time": "2020-12-02T13:55:54.096516", "status": "completed"} tags=[]
from pathlib import Path

import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns

from utils import generate_result_set_name

# %% [markdown] papermill={"duration": 0.018453, "end_time": "2020-12-02T13:55:55.748175", "exception": false, "start_time": "2020-12-02T13:55:55.729722", "status": "completed"} tags=[]
# # Settings

# %% papermill={"duration": 0.032443, "end_time": "2020-12-02T13:55:55.799086", "exception": false, "start_time": "2020-12-02T13:55:55.766643", "status": "completed"} tags=[]
INITIAL_RANDOM_STATE = 70000

# %% [markdown] papermill={"duration": 0.018417, "end_time": "2020-12-02T13:55:55.836539", "exception": false, "start_time": "2020-12-02T13:55:55.818122", "status": "completed"} tags=[]
# ## Input data

# %% papermill={"duration": 0.032101, "end_time": "2020-12-02T13:55:55.887495", "exception": false, "start_time": "2020-12-02T13:55:55.855394", "status": "completed"} tags=[]
INPUT_SUBSET = 'pca'

# %% papermill={"duration": 0.032645, "end_time": "2020-12-02T13:55:55.939118", "exception": false, "start_time": "2020-12-02T13:55:55.906473", "status": "completed"} tags=[]
INPUT_STEM = 'z_score_std-projection-smultixcan-efo_partial-mashr-zscores'

# %% papermill={"duration": 0.03242, "end_time": "2020-12-02T13:55:55.990511", "exception": false, "start_time": "2020-12-02T13:55:55.958091", "status": "completed"} tags=[]
DR_OPTIONS = {
    'n_components': 50,
    'svd_solver': 'full',
    'random_state': 0,
}

# %% papermill={"duration": 0.035549, "end_time": "2020-12-02T13:55:56.045591", "exception": false, "start_time": "2020-12-02T13:55:56.010042", "status": "completed"} tags=[]
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

# %% [markdown] papermill={"duration": 0.019641, "end_time": "2020-12-02T13:55:56.085490", "exception": false, "start_time": "2020-12-02T13:55:56.065849", "status": "completed"} tags=[]
# ## Clustering

# %% papermill={"duration": 0.039917, "end_time": "2020-12-02T13:55:56.144271", "exception": false, "start_time": "2020-12-02T13:55:56.104354", "status": "completed"} tags=[]
from sklearn.mixture import GaussianMixture

# %% papermill={"duration": 0.034392, "end_time": "2020-12-02T13:55:56.198723", "exception": false, "start_time": "2020-12-02T13:55:56.164331", "status": "completed"} tags=[]
CLUSTERING_ATTRIBUTES_TO_SAVE = ['n_clusters']

# %% papermill={"duration": 0.034606, "end_time": "2020-12-02T13:55:56.253406", "exception": false, "start_time": "2020-12-02T13:55:56.218800", "status": "completed"} tags=[]
CLUSTERING_OPTIONS = {}

CLUSTERING_OPTIONS['K_MIN'] = 2
CLUSTERING_OPTIONS['K_MAX'] = 60 # sqrt(3749)
CLUSTERING_OPTIONS['N_REPS_PER_K'] = 5
CLUSTERING_OPTIONS['N_INIT'] = 10
CLUSTERING_OPTIONS['COVARIANCE_TYPE'] = 'full'

display(CLUSTERING_OPTIONS)

# %% papermill={"duration": 0.036116, "end_time": "2020-12-02T13:55:56.309982", "exception": false, "start_time": "2020-12-02T13:55:56.273866", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.034944, "end_time": "2020-12-02T13:55:56.366015", "exception": false, "start_time": "2020-12-02T13:55:56.331071", "status": "completed"} tags=[]
display(len(CLUSTERERS))

# %% papermill={"duration": 0.037395, "end_time": "2020-12-02T13:55:56.424853", "exception": false, "start_time": "2020-12-02T13:55:56.387458", "status": "completed"} tags=[]
_iter = iter(CLUSTERERS.items())
display(next(_iter))
display(next(_iter))

# %% papermill={"duration": 0.034936, "end_time": "2020-12-02T13:55:56.481036", "exception": false, "start_time": "2020-12-02T13:55:56.446100", "status": "completed"} tags=[]
clustering_method_name = method_name
display(clustering_method_name)

# %% [markdown] papermill={"duration": 0.020413, "end_time": "2020-12-02T13:55:56.522167", "exception": false, "start_time": "2020-12-02T13:55:56.501754", "status": "completed"} tags=[]
# ## Output directory

# %% papermill={"duration": 0.036034, "end_time": "2020-12-02T13:55:56.578313", "exception": false, "start_time": "2020-12-02T13:55:56.542279", "status": "completed"} tags=[]
# output dir for this notebook
RESULTS_DIR = Path(
    conf.RESULTS["CLUSTERING_RUNS_DIR"],
    f'{INPUT_SUBSET}-{INPUT_STEM}',
).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %% [markdown] papermill={"duration": 0.021292, "end_time": "2020-12-02T13:55:56.621864", "exception": false, "start_time": "2020-12-02T13:55:56.600572", "status": "completed"} tags=[]
# # Load input file

# %% papermill={"duration": 0.036636, "end_time": "2020-12-02T13:55:56.679006", "exception": false, "start_time": "2020-12-02T13:55:56.642370", "status": "completed"} tags=[]
data = pd.read_pickle(input_filepath)

# %% papermill={"duration": 0.036053, "end_time": "2020-12-02T13:55:56.736433", "exception": false, "start_time": "2020-12-02T13:55:56.700380", "status": "completed"} tags=[]
data.shape

# %% papermill={"duration": 0.048674, "end_time": "2020-12-02T13:55:56.806756", "exception": false, "start_time": "2020-12-02T13:55:56.758082", "status": "completed"} tags=[]
data.head()

# %% papermill={"duration": 0.036388, "end_time": "2020-12-02T13:55:56.865416", "exception": false, "start_time": "2020-12-02T13:55:56.829028", "status": "completed"} tags=[]
assert not data.isna().any().any()

# %% [markdown] papermill={"duration": 0.02155, "end_time": "2020-12-02T13:55:56.909425", "exception": false, "start_time": "2020-12-02T13:55:56.887875", "status": "completed"} tags=[]
# # Clustering

# %% [markdown] papermill={"duration": 0.021425, "end_time": "2020-12-02T13:55:56.952035", "exception": false, "start_time": "2020-12-02T13:55:56.930610", "status": "completed"} tags=[]
# ## Generate ensemble

# %% papermill={"duration": 0.03778, "end_time": "2020-12-02T13:55:57.010881", "exception": false, "start_time": "2020-12-02T13:55:56.973101", "status": "completed"} tags=[]
from clustering.ensemble import generate_ensemble

# %% papermill={"duration": 10575.928328, "end_time": "2020-12-02T16:52:12.961654", "exception": false, "start_time": "2020-12-02T13:55:57.033326", "status": "completed"} tags=[]
ensemble = generate_ensemble(
    data,
    CLUSTERERS,
    attributes=CLUSTERING_ATTRIBUTES_TO_SAVE,
)

# %% papermill={"duration": 0.083268, "end_time": "2020-12-02T16:52:13.113510", "exception": false, "start_time": "2020-12-02T16:52:13.030242", "status": "completed"} tags=[]
# the number should be close to 295 (the number of partitions generated by k-means/spectral clustering)
ensemble.shape

# %% papermill={"duration": 0.091776, "end_time": "2020-12-02T16:52:13.275003", "exception": false, "start_time": "2020-12-02T16:52:13.183227", "status": "completed"} tags=[]
ensemble.head()

# %% papermill={"duration": 0.085218, "end_time": "2020-12-02T16:52:13.430049", "exception": false, "start_time": "2020-12-02T16:52:13.344831", "status": "completed"} tags=[]
ensemble['n_clusters'].value_counts().head()

# %% papermill={"duration": 0.085807, "end_time": "2020-12-02T16:52:13.585240", "exception": false, "start_time": "2020-12-02T16:52:13.499433", "status": "completed"} tags=[]
ensemble_stats = ensemble['n_clusters'].describe()
display(ensemble_stats)

# %% [markdown] papermill={"duration": 0.068617, "end_time": "2020-12-02T16:52:13.722561", "exception": false, "start_time": "2020-12-02T16:52:13.653944", "status": "completed"} tags=[]
# ## Testing

# %% papermill={"duration": 0.084634, "end_time": "2020-12-02T16:52:13.878722", "exception": false, "start_time": "2020-12-02T16:52:13.794088", "status": "completed"} tags=[]
assert ensemble_stats['min'] > 1

# %% papermill={"duration": 0.08325, "end_time": "2020-12-02T16:52:14.030873", "exception": false, "start_time": "2020-12-02T16:52:13.947623", "status": "completed"} tags=[]
assert not ensemble['n_clusters'].isna().any()

# %% papermill={"duration": 0.08298, "end_time": "2020-12-02T16:52:14.183491", "exception": false, "start_time": "2020-12-02T16:52:14.100511", "status": "completed"} tags=[]
assert ensemble.shape[0] == len(CLUSTERERS)

# %% papermill={"duration": 0.101547, "end_time": "2020-12-02T16:52:14.355330", "exception": false, "start_time": "2020-12-02T16:52:14.253783", "status": "completed"} tags=[]
# all partitions have the right size
assert np.all([
    part['partition'].shape[0] == data.shape[0]
    for idx, part in ensemble.iterrows()
])

# %% papermill={"duration": 0.104165, "end_time": "2020-12-02T16:52:14.530177", "exception": false, "start_time": "2020-12-02T16:52:14.426012", "status": "completed"} tags=[]
# no partition has negative clusters (noisy points)
assert not np.any([
    (part['partition'] < 0).any()
    for idx, part in ensemble.iterrows()
])

# %% [markdown] papermill={"duration": 0.069153, "end_time": "2020-12-02T16:52:14.670237", "exception": false, "start_time": "2020-12-02T16:52:14.601084", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.117227, "end_time": "2020-12-02T16:52:14.856864", "exception": false, "start_time": "2020-12-02T16:52:14.739637", "status": "completed"} tags=[]
output_filename = Path(
    RESULTS_DIR,
    generate_result_set_name(
        CLUSTERING_OPTIONS,
        prefix=f'{clustering_method_name}-',
        suffix='.pkl',
    )
).resolve()
display(output_filename)

# %% papermill={"duration": 0.094269, "end_time": "2020-12-02T16:52:15.029713", "exception": false, "start_time": "2020-12-02T16:52:14.935444", "status": "completed"} tags=[]
ensemble.to_pickle(output_filename)

# %% [markdown] papermill={"duration": 0.069746, "end_time": "2020-12-02T16:52:15.170875", "exception": false, "start_time": "2020-12-02T16:52:15.101129", "status": "completed"} tags=[]
# # Stability

# %% [markdown] papermill={"duration": 0.070453, "end_time": "2020-12-02T16:52:15.311629", "exception": false, "start_time": "2020-12-02T16:52:15.241176", "status": "completed"} tags=[]
# ## Group ensemble by n_clusters

# %% papermill={"duration": 0.101283, "end_time": "2020-12-02T16:52:15.482713", "exception": false, "start_time": "2020-12-02T16:52:15.381430", "status": "completed"} tags=[]
parts = ensemble.groupby('n_clusters').apply(lambda x: np.concatenate(x['partition'].apply(lambda x: x.reshape(1, -1)), axis=0))

# %% papermill={"duration": 0.096932, "end_time": "2020-12-02T16:52:15.650535", "exception": false, "start_time": "2020-12-02T16:52:15.553603", "status": "completed"} tags=[]
parts.head()

# %% papermill={"duration": 0.088046, "end_time": "2020-12-02T16:52:15.809983", "exception": false, "start_time": "2020-12-02T16:52:15.721937", "status": "completed"} tags=[]
assert np.all([
    parts.loc[k].shape == (CLUSTERING_OPTIONS['N_REPS_PER_K'], data.shape[0])
    for k in parts.index
])

# %% [markdown] papermill={"duration": 0.070881, "end_time": "2020-12-02T16:52:15.952665", "exception": false, "start_time": "2020-12-02T16:52:15.881784", "status": "completed"} tags=[]
# ## Compute stability

# %% papermill={"duration": 0.085401, "end_time": "2020-12-02T16:52:16.107714", "exception": false, "start_time": "2020-12-02T16:52:16.022313", "status": "completed"} tags=[]
from sklearn.metrics import adjusted_rand_score as ari
from scipy.spatial.distance import squareform, pdist

# %% papermill={"duration": 0.709356, "end_time": "2020-12-02T16:52:16.888360", "exception": false, "start_time": "2020-12-02T16:52:16.179004", "status": "completed"} tags=[]
parts_ari = pd.Series(
    {k: pdist(parts.loc[k], metric=ari) for k in parts.index},
    name='k'
)

# %% papermill={"duration": 0.087276, "end_time": "2020-12-02T16:52:17.046873", "exception": false, "start_time": "2020-12-02T16:52:16.959597", "status": "completed"} tags=[]
parts_ari_stability = parts_ari.apply(lambda x: x.mean())
display(parts_ari_stability.sort_values(ascending=False).head(15))

# %% papermill={"duration": 0.08664, "end_time": "2020-12-02T16:52:17.205456", "exception": false, "start_time": "2020-12-02T16:52:17.118816", "status": "completed"} tags=[]
parts_ari_df = pd.DataFrame.from_records(parts_ari.tolist()).set_index(parts_ari.index.copy())

# %% papermill={"duration": 0.085838, "end_time": "2020-12-02T16:52:17.362458", "exception": false, "start_time": "2020-12-02T16:52:17.276620", "status": "completed"} tags=[]
parts_ari_df.shape

# %% papermill={"duration": 0.085888, "end_time": "2020-12-02T16:52:17.519591", "exception": false, "start_time": "2020-12-02T16:52:17.433703", "status": "completed"} tags=[]
assert int( (CLUSTERING_OPTIONS['N_REPS_PER_K'] * (CLUSTERING_OPTIONS['N_REPS_PER_K'] - 1) ) / 2) == parts_ari_df.shape[1]

# %% papermill={"duration": 0.091284, "end_time": "2020-12-02T16:52:17.681725", "exception": false, "start_time": "2020-12-02T16:52:17.590441", "status": "completed"} tags=[]
parts_ari_df.head()

# %% [markdown] papermill={"duration": 0.074213, "end_time": "2020-12-02T16:52:17.828397", "exception": false, "start_time": "2020-12-02T16:52:17.754184", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.08708, "end_time": "2020-12-02T16:52:17.987345", "exception": false, "start_time": "2020-12-02T16:52:17.900265", "status": "completed"} tags=[]
output_filename = Path(
    RESULTS_DIR,
    generate_result_set_name(
        CLUSTERING_OPTIONS,
        prefix=f'{clustering_method_name}-stability-',
        suffix='.pkl',
    )
).resolve()
display(output_filename)

# %% papermill={"duration": 0.085441, "end_time": "2020-12-02T16:52:18.144504", "exception": false, "start_time": "2020-12-02T16:52:18.059063", "status": "completed"} tags=[]
parts_ari_df.to_pickle(output_filename)

# %% [markdown] papermill={"duration": 0.071104, "end_time": "2020-12-02T16:52:18.288391", "exception": false, "start_time": "2020-12-02T16:52:18.217287", "status": "completed"} tags=[]
# ## Stability plot

# %% papermill={"duration": 0.088761, "end_time": "2020-12-02T16:52:18.448639", "exception": false, "start_time": "2020-12-02T16:52:18.359878", "status": "completed"} tags=[]
parts_ari_df_plot = parts_ari_df.stack().reset_index().rename(columns={'level_0': 'k', 'level_1': 'idx', 0: 'ari'})

# %% papermill={"duration": 0.087794, "end_time": "2020-12-02T16:52:18.608980", "exception": false, "start_time": "2020-12-02T16:52:18.521186", "status": "completed"} tags=[]
parts_ari_df_plot.dtypes

# %% papermill={"duration": 0.090272, "end_time": "2020-12-02T16:52:18.772518", "exception": false, "start_time": "2020-12-02T16:52:18.682246", "status": "completed"} tags=[]
parts_ari_df_plot.head()

# %% papermill={"duration": 2.941994, "end_time": "2020-12-02T16:52:21.790441", "exception": false, "start_time": "2020-12-02T16:52:18.848447", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.074624, "end_time": "2020-12-02T16:52:21.941985", "exception": false, "start_time": "2020-12-02T16:52:21.867361", "status": "completed"} tags=[]
