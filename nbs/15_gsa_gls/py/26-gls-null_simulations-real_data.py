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
# TODO

# %% [markdown] tags=[]
# # Environment variables

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
import conf

# %% tags=[]
# N_JOBS = conf.GENERAL["N_JOBS"]
# set N_JOBS manually
N_JOBS = 2
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

import numpy as np
import pandas as pd
from tqdm import tqdm

from gls import GLSPhenoplier

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
N_SIMULATED_PHENOTYPES = 100

# %% tags=[]
OUTPUT_DIR = conf.RESULTS["GLS"]
display(OUTPUT_DIR)

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# %% tags=[]
OUTPUT_FILENAME = OUTPUT_DIR / "gls-null_simulations-real_data.pkl"
display(OUTPUT_FILENAME)

# %% [markdown] tags=[]
# # Load data

# %% [markdown] tags=[]
# ## MultiPLIER Z matrix

# %% tags=[]
multiplier_z_matrix = pd.read_pickle(conf.MULTIPLIER["MODEL_Z_MATRIX_FILE"])

# %% tags=[]
multiplier_z_matrix.shape

# %% tags=[]
multiplier_z_matrix.head()

# %% tags=[]
lv_codes = list(multiplier_z_matrix.columns)
display(lv_codes[:5])

# %% [markdown] tags=[]
# # GLSPhenoplier

# %% [markdown] tags=[]
# ## Functions

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
phenotype_assocs, lv_weights = GLSPhenoplier._get_data(
    conf.PHENOMEXCAN["SMULTIXCAN_EFO_PARTIAL_MASHR_ZSCORES_FILE"]
)[1:]

# %% tags=[]
phenotype_assocs.shape

# %% tags=[]
phenotype_assocs.head()

# %%
phenotype_list = list(phenotype_assocs.columns)
display(phenotype_list[:5])

# %% tags=[]
lv_weights.shape

# %% tags=[]
lv_weights.head()

# %% [markdown] tags=[]
# ## Generate simulated phenotypes

# %% tags=[]
rs = np.random.RandomState(0)

# %%
phenotype_codes = rs.choice(phenotype_list, size=N_SIMULATED_PHENOTYPES, replace=False)
display(phenotype_codes[:3])
display(len(phenotype_codes))
assert len(phenotype_codes) == N_SIMULATED_PHENOTYPES

simulated_phenotypes = {}

for phenotype_code in phenotype_codes:
    phenotype = phenotype_assocs[phenotype_code].copy()
    rs.shuffle(phenotype)

    simulated_phenotypes[phenotype_code] = phenotype

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
run_confs = pd.DataFrame(
    data=itertools.product(list(simulated_phenotypes.columns), lv_codes),
    columns=["phenotype", "lv"],
)

# %%
display(run_confs)
assert run_confs.shape[0] == int(N_SIMULATED_PHENOTYPES * len(lv_codes))

# %% [markdown] tags=[]
# ## Run

# %% tags=[]
results = []

pbar = tqdm(total=run_confs.shape[0])

for phenotype_code, lv_code in run_confs.sample(frac=1, random_state=rs).itertuples(
    name=None, index=False
):
    pbar.set_description(f"{phenotype_code} - {lv_code}")

    phenotype = simulated_phenotypes[phenotype_code]

    gls_model = GLSPhenoplier(
        smultixcan_result_set_filepath=conf.PHENOMEXCAN[
            "SMULTIXCAN_EFO_PARTIAL_MASHR_ZSCORES_FILE"
        ]
    )
    gls_model.fit_named(lv_code, phenotype)
    res = gls_model.results

    results.append(
        {
            "phenotype": phenotype_code,
            "lv": lv_code,
            "coef": res.params.loc["lv"],
            "pvalue": res.pvalues_onesided.loc["lv"],
            #                 "pvalue_twosided": res.pvalues.loc["lv"],
            #                 "summary": gls_model.results_summary,
        }
    )

    # save results every 10 models trained
    if (len(results) % 10) == 0:
        get_df_from_results(results).to_pickle(OUTPUT_FILENAME)

    pbar.update(1)

pbar.close()

# %% tags=[]
results = get_df_from_results(results)

# %% tags=[]
results.shape

# %% tags=[]
results.head()

# %% tags=[]
results.sort_values("pvalue").head(10)

# %% [markdown] tags=[]
# ## Save

# %% tags=[]
results.to_pickle(OUTPUT_FILENAME)

# %% tags=[]
