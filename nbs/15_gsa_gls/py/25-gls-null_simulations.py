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
N_JOBS = conf.GENERAL["N_JOBS"]
display(N_JOBS)

# %% tags=[]
# %env MKL_NUM_THREADS=$N_JOBS
# %env OPEN_BLAS_NUM_THREADS=$N_JOBS
# %env NUMEXPR_NUM_THREADS=$N_JOBS
# %env OMP_NUM_THREADS=$N_JOBS

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
from pathlib import Path

import statsmodels.api as sm
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from tqdm import tqdm

from gls import GLSPhenoplier

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
N_SIMULATED_PHENOTYPES = 10

# %% tags=[]
OUTPUT_DIR = conf.RESULTS["GLS"]
display(OUTPUT_DIR)

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# %% tags=[]
OUTPUT_FILENAME = OUTPUT_DIR / "gls-null_simulations.pkl"
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
# ## PhenomeXcan (S-MultiXcan)

# %% tags=[]
# INPUT_SUBSET = "z_score_std"

# %% tags=[]
# INPUT_STEM = "projection-smultixcan-efo_partial-mashr-zscores"

# %% tags=[]
# input_filepath = Path(
#     conf.RESULTS["DATA_TRANSFORMATIONS_DIR"],
#     INPUT_SUBSET,
#     f"{INPUT_SUBSET}-{INPUT_STEM}.pkl",
# ).resolve()

# %% tags=[]
# data = pd.read_pickle(input_filepath)

# %% tags=[]
# data.shape

# %% tags=[]
# data.head()

# %% [markdown] tags=[]
# # GLSPhenoplier

# %% [markdown] tags=[]
# ## Load `lv_weights`

# %% tags=[]
lv_weights = GLSPhenoplier._get_data(
    conf.PHENOMEXCAN["SMULTIXCAN_EFO_PARTIAL_MASHR_ZSCORES_FILE"]
)[2]

# %% tags=[]
lv_weights.shape

# %% tags=[]
lv_weights.head()

# %% [markdown] tags=[]
# ## Run

# %% tags=[]
rs = np.random.RandomState(0)

# %% tags=[]
results = []

pbar = tqdm(total=int(N_SIMULATED_PHENOTYPES * len(lv_codes)))

for idx in range(N_SIMULATED_PHENOTYPES):
    # generate a random phenotype
    phenotype_code = f"random_normal-{idx}"

    phenotype = pd.Series(
        # use abs to simulate MultiPLIER z-scores (always positives)
        np.abs(rs.normal(size=lv_weights.shape[0])),
        index=lv_weights.index.copy(),
        name=phenotype_code,
    )

    # compute an association for all LVs
    for lv_code in lv_codes:
        pbar.set_description(f"{phenotype_code} - {lv_code}")

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
                "pvalue_twosided": res.pvalues.loc["lv"],
                "summary": gls_model.results_summary,
            }
        )

        # save results every 10 models trained
        if (len(results) % 10) == 0:
            pd.DataFrame(results).to_pickle(OUTPUT_FILENAME)

        pbar.update(1)

pbar.close()

# %% tags=[]
results = pd.DataFrame(results)

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
