# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
#     formats: ipynb,py//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] tags=[]
# # Description

# %% [markdown] tags=[]
# * reads MultiXcan results on a random phenotype file (using Elastic Net models)
# * runs PhenoPLIER on all LVs to compute the null

# %% [markdown] tags=[]
# # Environment variables

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
import conf

# %% tags=[]
# N_JOBS = conf.GENERAL["N_JOBS"]
# set N_JOBS manually, because we are parallelizing outside
N_JOBS = 1
display(N_JOBS)

# %% tags=[]
# %env MKL_NUM_THREADS=$N_JOBS
# %env OPEN_BLAS_NUM_THREADS=$N_JOBS
# %env NUMEXPR_NUM_THREADS=$N_JOBS
# %env OMP_NUM_THREADS=$N_JOBS

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy import stats
import pandas as pd
from tqdm import tqdm

from utils import chunker
from entity import Gene
from gls import GLSPhenoplier

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
N_SIMULATED_PHENOTYPES = 10
CHUNK_SIZE = 50
EQTL_MODEL = "ELASTIC_NET"

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
OUTPUT_DIR = conf.RESULTS["GLS"] / "null_simulations"
display(OUTPUT_DIR)

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# %% tags=[]
OUTPUT_FILENAME = OUTPUT_DIR / "en-null_simulations.pkl"
display(OUTPUT_FILENAME)

# %% [markdown] tags=[]
# # Load data

# %% [markdown] tags=[]
# ## MultiXcan on random phenotype

# %% [markdown] tags=[]
# This result was downloaded from the MultiXcan paper here: https://github.com/hakyimlab/multixcan-paper

# %%
multixcan_random_phenotype = pd.read_csv(
    conf.PHENOMEXCAN["BASE_DIR"] / "random__ccn30__mt_results.txt",
    sep="\t",
    usecols=["gene", "pvalue"],
)

# %%
multixcan_random_phenotype.shape

# %%
multixcan_random_phenotype.head()

# %%
multixcan_random_phenotype["gene"] = multixcan_random_phenotype["gene"].str.split(
    ".", n=1, expand=True
)[0]

# %%
multixcan_random_phenotype = multixcan_random_phenotype.set_index("gene")

# %%
multixcan_random_phenotype.head()

# %%
assert multixcan_random_phenotype.index.is_unique

# %% [markdown] tags=[]
# ## MultiPLIER Z matrix

# %% tags=[]
# multiplier_z_matrix = pd.read_pickle(conf.MULTIPLIER["MODEL_Z_MATRIX_FILE"])

# %% tags=[]
# multiplier_z_matrix.shape

# %% tags=[]
# multiplier_z_matrix.head()

# %% tags=[]
# lv_codes = list(multiplier_z_matrix.columns)
# display(lv_codes[:5])

# %% [markdown]
# # Preprocess MultiXcan results

# %% [markdown] tags=[]
# ## Convert gene IDs to Gene names

# %% tags=[]
smultixcan_results = multixcan_random_phenotype.rename(index=Gene.GENE_ID_TO_NAME_MAP)

# %% tags=[]
smultixcan_results.shape

# %% tags=[]
smultixcan_results.head()

# %% [markdown] tags=[]
# ## Remove duplicated gene entries

# %% tags=[]
_tmp = smultixcan_results.index[smultixcan_results.index.duplicated(keep="first")]
assert _tmp.shape[0] == 0

# %% [markdown] tags=[]
# ## Convert p-values to z-scores

# %%
smultixcan_results = smultixcan_results.assign(
    zscore=np.abs(stats.norm.ppf(smultixcan_results["pvalue"].to_numpy() / 2))
)

# %%
smultixcan_results = smultixcan_results.drop(columns="pvalue").squeeze()

# %%
smultixcan_results.head()

# %%
smultixcan_results.describe()

# %% [markdown] tags=[]
# ## Some checks

# %% tags=[]
# the data should have no NaN values
assert smultixcan_results.shape == smultixcan_results.dropna(how="any").shape

# %% [markdown] tags=[]
# # GLSPhenoplier

# %% [markdown] tags=[]
# ## Identify clusters of non-related genes

# %% tags=[]
en_gene_corr = GLSPhenoplier._get_data(
    conf.PHENOMEXCAN["SMULTIXCAN_EFO_PARTIAL_MASHR_ZSCORES_FILE"],
    model_type="ELASTIC_NET",
)[0]

# %%
_comm_genes = en_gene_corr.index.intersection(smultixcan_results.index)

# %%
en_gene_corr = en_gene_corr.loc[_comm_genes, _comm_genes]

# %%
en_gene_corr.shape

# %%
en_gene_corr.head()

# %%
from sklearn.cluster import AgglomerativeClustering

# %%
en_gene_dist = en_gene_corr.abs().copy()
np.fill_diagonal(en_gene_dist.values, 0.0)

# %%
en_gene_dist

# %%
_tmp = en_gene_dist.unstack()
_tmp = _tmp[(_tmp > 0.0) & (_tmp < 1.0)]

# %%
_tmp.sort_values()

# %%
ac = AgglomerativeClustering(
    n_clusters=None,
    compute_full_tree=True,
    linkage="complete",
    affinity="precomputed",
    distance_threshold=1e-100,
)

# %%
ac.fit(en_gene_dist)

# %%
gene_part = pd.Series(ac.labels_)
display(gene_part.value_counts())

# %%
en_gene_dist.loc[(ac.labels_ == 141), (ac.labels_ == 141)]

# %%
phenotype_gene_clusters = {
    cluster_id: smultixcan_results.loc[en_gene_dist.index[gene_part == cluster_id]]
    for cluster_id in gene_part.value_counts().index
}

# %%
phenotype_gene_clusters[141]

# %% [markdown] tags=[]
# ## Functions

# %%
rs = np.random.RandomState(0)


# %%
def get_shuffled_phenotype():
    shuffled_gene_clusters = []
    for cluster_id, gene_assoc_cluster in phenotype_gene_clusters.items():
        gc = gene_assoc_cluster.copy()
        rs.shuffle(gc)
        shuffled_gene_clusters.append(gc)

    return pd.concat(shuffled_gene_clusters)


# %%
def get_df_from_results(results_list):
    df = pd.DataFrame(results_list).astype(
        {
            "phenotype": "category",
            "lv": "category",
        }
    )

    return df


# %% [markdown] tags=[]
# ## Load `phenotype_assocs` and `lv_weights`

# %% tags=[]
lv_weights = GLSPhenoplier._get_data(
    conf.PHENOMEXCAN["SMULTIXCAN_EFO_PARTIAL_MASHR_ZSCORES_FILE"],
    model_type=EQTL_MODEL,
)[2]

# %% tags=[]
lv_weights.shape

# %% tags=[]
lv_weights.head()

# %% [markdown] tags=[]
# ## Generate simulated phenotypes

# %%
# phenotype_codes = rs.choice(phenotype_list, size=N_SIMULATED_PHENOTYPES, replace=False)
# display(phenotype_codes[:3])
# display(len(phenotype_codes))
# assert len(phenotype_codes) == N_SIMULATED_PHENOTYPES

simulated_phenotypes = {
    "smultixcan phenotype 0": smultixcan_results.loc[en_gene_dist.index]
}

for idx in tqdm(range(1, N_SIMULATED_PHENOTYPES)):
    simulated_phenotypes[f"smultixcan phenotype {idx}"] = get_shuffled_phenotype()

# %%
display(len(simulated_phenotypes))
assert len(simulated_phenotypes) == N_SIMULATED_PHENOTYPES

# %%
simulated_phenotypes[list(simulated_phenotypes.keys())[0]]

# %%
simulated_phenotypes = pd.DataFrame(simulated_phenotypes)

# %%
simulated_phenotypes.shape

# %%
simulated_phenotypes.head()

# %%
simulated_phenotypes.describe()

# %% [markdown] tags=[]
# ## Merge simulated phenotypes and LVs into one dataframe

# %%
# smultixcan_results = smultixcan_results.loc[smultixcan_results.index.intersection(lv_weights.index)]

# %%
# smultixcan_results.shape

# %%
# smultixcan_results.head()

# %%
# assert not smultixcan_results.isna().any()

# %%
# simulated_phenotypes = pd.DataFrame({"smultixcan_random_phenotype": smultixcan_results})

# %%
# simulated_phenotypes.shape

# %%
# simulated_phenotypes.head()

# %%
run_confs = pd.DataFrame(
    data=itertools.product(
        list(simulated_phenotypes.columns), list(lv_weights.columns)
    ),
    columns=["phenotype", "lv"],
)

# %%
run_confs

# %% [markdown] tags=[]
# ## Split run configurations

# %%
run_confs_chunks = chunker(run_confs.sample(frac=1, random_state=rs), CHUNK_SIZE)


# %% [markdown] tags=[]
# ## Run

# %%
def run(run_confs_subset):
    results = []

    for phenotype_code, lv_code in run_confs_subset.itertuples(name=None, index=False):
        phenotype = simulated_phenotypes[phenotype_code]

        gls_model = GLSPhenoplier(
            smultixcan_result_set_filepath=conf.PHENOMEXCAN[
                "SMULTIXCAN_EFO_PARTIAL_MASHR_ZSCORES_FILE"
            ],
            model_type=EQTL_MODEL,
        )
        gls_model.fit_named(lv_code, phenotype)
        res = gls_model.results

        results.append(
            {
                "phenotype": phenotype_code,
                "lv": lv_code,
                "coef": res.params.loc["lv"],
                "pvalue": res.pvalues_onesided.loc["lv"],
            }
        )

    return get_df_from_results(results)


# %% tags=[]
all_results = []

with tqdm(total=run_confs.shape[0]) as pbar:
    with ProcessPoolExecutor(max_workers=conf.GENERAL["N_JOBS"]) as executor:
        tasks = [executor.submit(run, chunk) for chunk in run_confs_chunks]

        for future in as_completed(tasks):
            res = future.result()
            all_results.append(res)

            if (len(all_results) % conf.GENERAL["N_JOBS"]) == 0:
                df = pd.concat(all_results, ignore_index=True)
                df.to_pickle(OUTPUT_FILENAME)

            pbar.update(res.shape[0])

# %%
pd.concat(all_results, ignore_index=True)

# %% tags=[]
# results = get_df_from_results(results)

# %% tags=[]
all_results.shape

# %% tags=[]
all_results.head()

# %% tags=[]
all_results.sort_values("pvalue").head(10)

# %% [markdown] tags=[]
# ## Save

# %% tags=[]
all_results.to_pickle(OUTPUT_FILENAME)

# %% tags=[]
