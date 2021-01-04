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

# %% [markdown] papermill={"duration": 0.025384, "end_time": "2020-12-02T18:44:00.130390", "exception": false, "start_time": "2020-12-02T18:44:00.105006", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.020369, "end_time": "2020-12-02T18:44:00.171588", "exception": false, "start_time": "2020-12-02T18:44:00.151219", "status": "completed"} tags=[]
# Runs k-means on the pca version of the data.

# %% [markdown] papermill={"duration": 0.020223, "end_time": "2020-12-02T18:44:00.212568", "exception": false, "start_time": "2020-12-02T18:44:00.192345", "status": "completed"} tags=[]
# # Environment variables

# %% papermill={"duration": 0.034183, "end_time": "2020-12-02T18:44:00.267157", "exception": false, "start_time": "2020-12-02T18:44:00.232974", "status": "completed"} tags=[]
from IPython.display import display

import conf

N_JOBS = conf.GENERAL['N_JOBS']
display(N_JOBS)

# %% papermill={"duration": 0.027466, "end_time": "2020-12-02T18:44:00.315803", "exception": false, "start_time": "2020-12-02T18:44:00.288337", "status": "completed"} tags=[]
# %env MKL_NUM_THREADS=$N_JOBS
# %env OPEN_BLAS_NUM_THREADS=$N_JOBS
# %env NUMEXPR_NUM_THREADS=$N_JOBS
# %env OMP_NUM_THREADS=$N_JOBS

# %% [markdown] papermill={"duration": 0.020801, "end_time": "2020-12-02T18:44:00.357812", "exception": false, "start_time": "2020-12-02T18:44:00.337011", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.031702, "end_time": "2020-12-02T18:44:00.410670", "exception": false, "start_time": "2020-12-02T18:44:00.378968", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 1.600114, "end_time": "2020-12-02T18:44:02.032227", "exception": false, "start_time": "2020-12-02T18:44:00.432113", "status": "completed"} tags=[]
from pathlib import Path

import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns

from utils import generate_result_set_name

# %% [markdown] papermill={"duration": 0.020482, "end_time": "2020-12-02T18:44:02.076536", "exception": false, "start_time": "2020-12-02T18:44:02.056054", "status": "completed"} tags=[]
# # Settings

# %% papermill={"duration": 0.034898, "end_time": "2020-12-02T18:44:02.132026", "exception": false, "start_time": "2020-12-02T18:44:02.097128", "status": "completed"} tags=[]
INITIAL_RANDOM_STATE = 10000

# %% [markdown] papermill={"duration": 0.020714, "end_time": "2020-12-02T18:44:02.174061", "exception": false, "start_time": "2020-12-02T18:44:02.153347", "status": "completed"} tags=[]
# ## Input data

# %% papermill={"duration": 0.034585, "end_time": "2020-12-02T18:44:02.229262", "exception": false, "start_time": "2020-12-02T18:44:02.194677", "status": "completed"} tags=[]
INPUT_SUBSET = 'pca'

# %% papermill={"duration": 0.035828, "end_time": "2020-12-02T18:44:02.286646", "exception": false, "start_time": "2020-12-02T18:44:02.250818", "status": "completed"} tags=[]
INPUT_STEM = 'z_score_std-projection-smultixcan-efo_partial-mashr-zscores'

# %% papermill={"duration": 0.035402, "end_time": "2020-12-02T18:44:02.343836", "exception": false, "start_time": "2020-12-02T18:44:02.308434", "status": "completed"} tags=[]
DR_OPTIONS = {
    'n_components': 50,
    'svd_solver': 'full',
    'random_state': 0,
}

# %% papermill={"duration": 0.037498, "end_time": "2020-12-02T18:44:02.402742", "exception": false, "start_time": "2020-12-02T18:44:02.365244", "status": "completed"} tags=[]
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

# %% [markdown] papermill={"duration": 0.021284, "end_time": "2020-12-02T18:44:02.445669", "exception": false, "start_time": "2020-12-02T18:44:02.424385", "status": "completed"} tags=[]
# ## Clustering

# %% papermill={"duration": 0.043987, "end_time": "2020-12-02T18:44:02.510945", "exception": false, "start_time": "2020-12-02T18:44:02.466958", "status": "completed"} tags=[]
from sklearn.cluster import KMeans

# %% papermill={"duration": 0.035489, "end_time": "2020-12-02T18:44:02.570690", "exception": false, "start_time": "2020-12-02T18:44:02.535201", "status": "completed"} tags=[]
CLUSTERING_ATTRIBUTES_TO_SAVE = ['n_clusters']

# %% papermill={"duration": 0.036435, "end_time": "2020-12-02T18:44:02.628834", "exception": false, "start_time": "2020-12-02T18:44:02.592399", "status": "completed"} tags=[]
CLUSTERING_OPTIONS = {}

CLUSTERING_OPTIONS['K_MIN'] = 2
CLUSTERING_OPTIONS['K_MAX'] = 60 # sqrt(3749)
CLUSTERING_OPTIONS['N_REPS_PER_K'] = 5
CLUSTERING_OPTIONS['KMEANS_N_INIT'] = 10

display(CLUSTERING_OPTIONS)

# %% papermill={"duration": 0.037482, "end_time": "2020-12-02T18:44:02.688067", "exception": false, "start_time": "2020-12-02T18:44:02.650585", "status": "completed"} tags=[]
CLUSTERERS = {}

idx = 0
random_state = INITIAL_RANDOM_STATE

for k in range(CLUSTERING_OPTIONS['K_MIN'], CLUSTERING_OPTIONS['K_MAX']+1):
    for i in range(CLUSTERING_OPTIONS['N_REPS_PER_K']):
        clus = KMeans(
                n_clusters=k,
                n_init=CLUSTERING_OPTIONS['KMEANS_N_INIT'],
                random_state=random_state,
            )
        
        method_name = type(clus).__name__
        CLUSTERERS[f'{method_name} #{idx}'] = clus
        
        random_state = random_state + 1
        idx = idx + 1

# %% papermill={"duration": 0.036636, "end_time": "2020-12-02T18:44:02.746796", "exception": false, "start_time": "2020-12-02T18:44:02.710160", "status": "completed"} tags=[]
display(len(CLUSTERERS))

# %% papermill={"duration": 0.038325, "end_time": "2020-12-02T18:44:02.807662", "exception": false, "start_time": "2020-12-02T18:44:02.769337", "status": "completed"} tags=[]
_iter = iter(CLUSTERERS.items())
display(next(_iter))
display(next(_iter))

# %% papermill={"duration": 0.037032, "end_time": "2020-12-02T18:44:02.867281", "exception": false, "start_time": "2020-12-02T18:44:02.830249", "status": "completed"} tags=[]
clustering_method_name = method_name
display(clustering_method_name)

# %% [markdown] papermill={"duration": 0.022251, "end_time": "2020-12-02T18:44:02.912541", "exception": false, "start_time": "2020-12-02T18:44:02.890290", "status": "completed"} tags=[]
# ## Output directory

# %% papermill={"duration": 0.038231, "end_time": "2020-12-02T18:44:02.973227", "exception": false, "start_time": "2020-12-02T18:44:02.934996", "status": "completed"} tags=[]
# output dir for this notebook
RESULTS_DIR = Path(
    conf.RESULTS["CLUSTERING_RUNS_DIR"],
    f'{INPUT_SUBSET}-{INPUT_STEM}',
).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_DIR)

# %% [markdown] papermill={"duration": 0.022726, "end_time": "2020-12-02T18:44:03.018883", "exception": false, "start_time": "2020-12-02T18:44:02.996157", "status": "completed"} tags=[]
# # Load input file

# %% papermill={"duration": 0.038658, "end_time": "2020-12-02T18:44:03.080793", "exception": false, "start_time": "2020-12-02T18:44:03.042135", "status": "completed"} tags=[]
data = pd.read_pickle(input_filepath)

# %% papermill={"duration": 0.037297, "end_time": "2020-12-02T18:44:03.141020", "exception": false, "start_time": "2020-12-02T18:44:03.103723", "status": "completed"} tags=[]
data.shape

# %% papermill={"duration": 0.050399, "end_time": "2020-12-02T18:44:03.214534", "exception": false, "start_time": "2020-12-02T18:44:03.164135", "status": "completed"} tags=[]
data.head()

# %% papermill={"duration": 0.037768, "end_time": "2020-12-02T18:44:03.275545", "exception": false, "start_time": "2020-12-02T18:44:03.237777", "status": "completed"} tags=[]
assert not data.isna().any().any()

# %% [markdown] papermill={"duration": 0.023336, "end_time": "2020-12-02T18:44:03.323050", "exception": false, "start_time": "2020-12-02T18:44:03.299714", "status": "completed"} tags=[]
# # Clustering

# %% [markdown] papermill={"duration": 0.022785, "end_time": "2020-12-02T18:44:03.368779", "exception": false, "start_time": "2020-12-02T18:44:03.345994", "status": "completed"} tags=[]
# ## Generate ensemble

# %% papermill={"duration": 0.039536, "end_time": "2020-12-02T18:44:03.431048", "exception": false, "start_time": "2020-12-02T18:44:03.391512", "status": "completed"} tags=[]
from clustering.ensemble import generate_ensemble

# %% papermill={"duration": 352.216708, "end_time": "2020-12-02T18:49:55.671008", "exception": false, "start_time": "2020-12-02T18:44:03.454300", "status": "completed"} tags=[]
ensemble = generate_ensemble(
    data,
    CLUSTERERS,
    attributes=CLUSTERING_ATTRIBUTES_TO_SAVE,
)

# %% papermill={"duration": 0.084385, "end_time": "2020-12-02T18:49:55.824934", "exception": false, "start_time": "2020-12-02T18:49:55.740549", "status": "completed"} tags=[]
# the number should be close to 295 (the number of partitions generated by k-means/spectral clustering)
ensemble.shape

# %% papermill={"duration": 0.095265, "end_time": "2020-12-02T18:49:55.991583", "exception": false, "start_time": "2020-12-02T18:49:55.896318", "status": "completed"} tags=[]
ensemble.head()

# %% papermill={"duration": 0.087114, "end_time": "2020-12-02T18:49:56.150031", "exception": false, "start_time": "2020-12-02T18:49:56.062917", "status": "completed"} tags=[]
ensemble['n_clusters'].value_counts().head()

# %% papermill={"duration": 0.087293, "end_time": "2020-12-02T18:49:56.307746", "exception": false, "start_time": "2020-12-02T18:49:56.220453", "status": "completed"} tags=[]
ensemble_stats = ensemble['n_clusters'].describe()
display(ensemble_stats)

# %% [markdown] papermill={"duration": 0.069676, "end_time": "2020-12-02T18:49:56.448660", "exception": false, "start_time": "2020-12-02T18:49:56.378984", "status": "completed"} tags=[]
# ## Testing

# %% papermill={"duration": 0.085961, "end_time": "2020-12-02T18:49:56.604792", "exception": false, "start_time": "2020-12-02T18:49:56.518831", "status": "completed"} tags=[]
assert ensemble_stats['min'] > 1

# %% papermill={"duration": 0.084809, "end_time": "2020-12-02T18:49:56.759997", "exception": false, "start_time": "2020-12-02T18:49:56.675188", "status": "completed"} tags=[]
assert not ensemble['n_clusters'].isna().any()

# %% papermill={"duration": 0.089788, "end_time": "2020-12-02T18:49:56.922118", "exception": false, "start_time": "2020-12-02T18:49:56.832330", "status": "completed"} tags=[]
assert ensemble.shape[0] == len(CLUSTERERS)

# %% papermill={"duration": 0.102423, "end_time": "2020-12-02T18:49:57.096774", "exception": false, "start_time": "2020-12-02T18:49:56.994351", "status": "completed"} tags=[]
# all partitions have the right size
assert np.all([
    part['partition'].shape[0] == data.shape[0]
    for idx, part in ensemble.iterrows()
])

# %% papermill={"duration": 0.105329, "end_time": "2020-12-02T18:49:57.273648", "exception": false, "start_time": "2020-12-02T18:49:57.168319", "status": "completed"} tags=[]
# no partition has negative clusters (noisy points)
assert not np.any([
    (part['partition'] < 0).any()
    for idx, part in ensemble.iterrows()
])

# %% [markdown] papermill={"duration": 0.069854, "end_time": "2020-12-02T18:49:57.414383", "exception": false, "start_time": "2020-12-02T18:49:57.344529", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.086382, "end_time": "2020-12-02T18:49:57.571735", "exception": false, "start_time": "2020-12-02T18:49:57.485353", "status": "completed"} tags=[]
output_filename = Path(
    RESULTS_DIR,
    generate_result_set_name(
        CLUSTERING_OPTIONS,
        prefix=f'{clustering_method_name}-',
        suffix='.pkl',
    )
).resolve()
display(output_filename)

# %% papermill={"duration": 0.097078, "end_time": "2020-12-02T18:49:57.741110", "exception": false, "start_time": "2020-12-02T18:49:57.644032", "status": "completed"} tags=[]
ensemble.to_pickle(output_filename)

# %% [markdown] papermill={"duration": 0.071857, "end_time": "2020-12-02T18:49:57.885018", "exception": false, "start_time": "2020-12-02T18:49:57.813161", "status": "completed"} tags=[]
# # Stability

# %% [markdown] papermill={"duration": 0.073026, "end_time": "2020-12-02T18:49:58.031352", "exception": false, "start_time": "2020-12-02T18:49:57.958326", "status": "completed"} tags=[]
# ## Group ensemble by n_clusters

# %% papermill={"duration": 0.102501, "end_time": "2020-12-02T18:49:58.205548", "exception": false, "start_time": "2020-12-02T18:49:58.103047", "status": "completed"} tags=[]
parts = ensemble.groupby('n_clusters').apply(lambda x: np.concatenate(x['partition'].apply(lambda x: x.reshape(1, -1)), axis=0))

# %% papermill={"duration": 0.098028, "end_time": "2020-12-02T18:49:58.375195", "exception": false, "start_time": "2020-12-02T18:49:58.277167", "status": "completed"} tags=[]
parts.head()

# %% papermill={"duration": 0.086388, "end_time": "2020-12-02T18:49:58.533513", "exception": false, "start_time": "2020-12-02T18:49:58.447125", "status": "completed"} tags=[]
assert np.all([
    parts.loc[k].shape == (CLUSTERING_OPTIONS['N_REPS_PER_K'], data.shape[0])
    for k in parts.index
])

# %% [markdown] papermill={"duration": 0.070928, "end_time": "2020-12-02T18:49:58.677810", "exception": false, "start_time": "2020-12-02T18:49:58.606882", "status": "completed"} tags=[]
# ## Compute stability

# %% papermill={"duration": 0.084959, "end_time": "2020-12-02T18:49:58.833190", "exception": false, "start_time": "2020-12-02T18:49:58.748231", "status": "completed"} tags=[]
from sklearn.metrics import adjusted_rand_score as ari
from scipy.spatial.distance import squareform, pdist

# %% papermill={"duration": 0.706572, "end_time": "2020-12-02T18:49:59.611460", "exception": false, "start_time": "2020-12-02T18:49:58.904888", "status": "completed"} tags=[]
parts_ari = pd.Series(
    {k: pdist(parts.loc[k], metric=ari) for k in parts.index},
    name='k'
)

# %% papermill={"duration": 0.088132, "end_time": "2020-12-02T18:49:59.770925", "exception": false, "start_time": "2020-12-02T18:49:59.682793", "status": "completed"} tags=[]
parts_ari_stability = parts_ari.apply(lambda x: x.mean())
display(parts_ari_stability.sort_values(ascending=False).head(15))

# %% papermill={"duration": 0.088741, "end_time": "2020-12-02T18:49:59.931085", "exception": false, "start_time": "2020-12-02T18:49:59.842344", "status": "completed"} tags=[]
parts_ari_df = pd.DataFrame.from_records(parts_ari.tolist()).set_index(parts_ari.index.copy())

# %% papermill={"duration": 0.086709, "end_time": "2020-12-02T18:50:00.090807", "exception": false, "start_time": "2020-12-02T18:50:00.004098", "status": "completed"} tags=[]
parts_ari_df.shape

# %% papermill={"duration": 0.086538, "end_time": "2020-12-02T18:50:00.249615", "exception": false, "start_time": "2020-12-02T18:50:00.163077", "status": "completed"} tags=[]
assert int( (CLUSTERING_OPTIONS['N_REPS_PER_K'] * (CLUSTERING_OPTIONS['N_REPS_PER_K'] - 1) ) / 2) == parts_ari_df.shape[1]

# %% papermill={"duration": 0.091877, "end_time": "2020-12-02T18:50:00.412992", "exception": false, "start_time": "2020-12-02T18:50:00.321115", "status": "completed"} tags=[]
parts_ari_df.head()

# %% [markdown] papermill={"duration": 0.071426, "end_time": "2020-12-02T18:50:00.556560", "exception": false, "start_time": "2020-12-02T18:50:00.485134", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.087135, "end_time": "2020-12-02T18:50:00.716403", "exception": false, "start_time": "2020-12-02T18:50:00.629268", "status": "completed"} tags=[]
output_filename = Path(
    RESULTS_DIR,
    generate_result_set_name(
        CLUSTERING_OPTIONS,
        prefix=f'{clustering_method_name}-stability-',
        suffix='.pkl',
    )
).resolve()
display(output_filename)

# %% papermill={"duration": 0.086837, "end_time": "2020-12-02T18:50:00.876058", "exception": false, "start_time": "2020-12-02T18:50:00.789221", "status": "completed"} tags=[]
parts_ari_df.to_pickle(output_filename)

# %% [markdown] papermill={"duration": 0.072722, "end_time": "2020-12-02T18:50:01.024911", "exception": false, "start_time": "2020-12-02T18:50:00.952189", "status": "completed"} tags=[]
# ## Stability plot

# %% papermill={"duration": 0.088727, "end_time": "2020-12-02T18:50:01.185534", "exception": false, "start_time": "2020-12-02T18:50:01.096807", "status": "completed"} tags=[]
parts_ari_df_plot = parts_ari_df.stack().reset_index().rename(columns={'level_0': 'k', 'level_1': 'idx', 0: 'ari'})

# %% papermill={"duration": 0.08795, "end_time": "2020-12-02T18:50:01.346509", "exception": false, "start_time": "2020-12-02T18:50:01.258559", "status": "completed"} tags=[]
parts_ari_df_plot.dtypes

# %% papermill={"duration": 0.091281, "end_time": "2020-12-02T18:50:01.511410", "exception": false, "start_time": "2020-12-02T18:50:01.420129", "status": "completed"} tags=[]
parts_ari_df_plot.head()

# %% papermill={"duration": 2.990997, "end_time": "2020-12-02T18:50:04.576103", "exception": false, "start_time": "2020-12-02T18:50:01.585106", "status": "completed"} tags=[]
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

# %% papermill={"duration": 0.0742, "end_time": "2020-12-02T18:50:04.724791", "exception": false, "start_time": "2020-12-02T18:50:04.650591", "status": "completed"} tags=[]
