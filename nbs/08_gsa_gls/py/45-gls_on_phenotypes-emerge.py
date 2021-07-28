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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] tags=[]
# # Description

# %% [markdown] tags=[]
# This notebook is similar to `30-gls_on_phenotypes-phenomexcan.ipynb` but for eMERGE S-MultiXcan results instead of PhenomeXcan.
#
# Since we don't have partition/clusters of eMERGE results, we selected the same set of LVs from the run on PhenomeXcan (`30-gls_on_phenotypes-phenomexcan.ipynb`); and regarding traits, we take the top 20 traits from each LVs to create a list of trait/LVs pairs to run GLSPhenoplier on.

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
N_TOP_TRAITS_FROM_LV = 20

# %% tags=[]
OUTPUT_DIR = conf.RESULTS["GLS"]
display(OUTPUT_DIR)

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# %% tags=[]
OUTPUT_FILENAME = OUTPUT_DIR / "gls_phenotypes-emerge.pkl"
display(OUTPUT_FILENAME)

# %% [markdown] tags=[]
# # Load data

# %% [markdown] tags=[]
# ## eMERGE traits info

# %% tags=[]
# FIXME: in the future, there will be a specific entry in config for the eMERGE directory that should be replaced here
input_filepath = Path(
    conf.DATA_DIR,
    "emerge",
    "eMERGE_III_PMBB_GSA_v2_2020_phecode_AFR_EUR_cc50_counts_w_dictionary.txt",
).resolve()
display(input_filepath)

# %% tags=[]
emerge_traits_df = pd.read_csv(
    input_filepath,
    sep="\t",
    dtype={"phecode": str},
    usecols=["phecode", "phenotype", "category"],
)

# %% tags=[]
emerge_traits_df = emerge_traits_df.rename(
    columns={
        "phenotype": "phecode_phenotype",
        "category": "phecode_category",
    }
)

# %% tags=[]
emerge_traits_df.shape

# %% tags=[]
emerge_traits_df.head()

# %% [markdown] tags=[]
# ## eMERGE (S-MultiXcan) projection

# %% tags=[]
input_filepath = Path(
    conf.RESULTS["PROJECTIONS_DIR"],
    "projection-emerge-smultixcan-mashr-zscores.pkl",
).resolve()
display(input_filepath)

# %% tags=[]
emerge_projection = pd.read_pickle(input_filepath)

# %% tags=[]
emerge_projection.shape

# %% tags=[]
emerge_projection.head()

# %% [markdown] tags=[]
# ## eMERGE (S-MultiXcan)

# %% tags=[]
# FIXME: in the future, there will be a specific entry in config for the eMERGE directory that should be replaced here
emerge_smultixcan_zscores_filepath = Path(
    conf.DATA_DIR,
    "emerge",
    "gene_assoc",
    "emerge-smultixcan-mashr-zscores.pkl",
).resolve()

display(emerge_smultixcan_zscores_filepath)

# %% tags=[]
_tmp = pd.read_pickle(emerge_smultixcan_zscores_filepath)

# %% tags=[]
_tmp.shape

# %% tags=[]
_tmp.head()

# %% [markdown] tags=[]
# ## GLS results on PhenomeXcan

# %% [markdown] tags=[]
# Read results obtained with `30-gls_on_phenotypes.ipynb` (PhenomeXcan)

# %% tags=[]
input_filepath = conf.RESULTS["GLS"] / "gls_phenotypes-phenomexcan.pkl"
display(input_filepath)

# %% tags=[]
gls_phenomexcan = pd.read_pickle(input_filepath)

# %% tags=[]
gls_phenomexcan.shape

# %% tags=[]
gls_phenomexcan.head()

# %% [markdown] tags=[]
# ## MultiPLIER summary

# %% tags=[]
multiplier_model_summary = pd.read_pickle(conf.MULTIPLIER["MODEL_SUMMARY_FILE"])

# %% tags=[]
multiplier_model_summary.shape

# %% tags=[]
multiplier_model_summary.head()

# %% tags=[]
well_aligned_lvs = multiplier_model_summary[
    (multiplier_model_summary["FDR"] < 0.05) | (multiplier_model_summary["AUC"] >= 0.75)
]

display(well_aligned_lvs.shape)
display(well_aligned_lvs.head())

# %% tags=[]
well_aligned_lv_codes = set([f"LV{lvi}" for lvi in well_aligned_lvs["LV index"]])

# %% tags=[]
len(well_aligned_lv_codes)

# %% tags=[]
list(well_aligned_lv_codes)[:5]

# %% [markdown] tags=[]
# # Select LVs

# %% tags=[]
gls_phenomexcan_lvs = [
    # from crispr analysis
    "LV246",
    "LV607",
    "LV74",
    "LV865",
    "LV612",
    "LV838",
    # drug-disease prediction
    "LV116",
    "LV931",
    "LV66",
    # cluster analysis
    "LV928",
    "LV30",
    "LV730",
    "LV598",
    "LV155",
    "LV844",
    "LV57",
    "LV54",
    "LV847",
    "LV136",
    "LV93",
    "LV206",
    "LV260",
    "LV21",
    "LV5",
    "LV434",
]

# %% tags=[]
len(gls_phenomexcan_lvs)

# %% tags=[]
assert len(gls_phenomexcan_lvs) == len(set(gls_phenomexcan_lvs)), "Repeated LVs in list"

# %% [markdown] tags=[]
# # GLSPhenoplier

# %% [markdown] tags=[]
# ## Get list of phenotypes/lvs pairs

# %% tags=[]
traits = emerge_traits_df["phecode"].unique()
display(traits.shape)

# %% tags=[]
phenotypes_lvs_pairs = []

# for each LV that was run on PhenomeXcan, I take the top `N_TOP_TRAITS_FROM_LV` traits
# in eMERGE
# ~~~+ global mapped phenotypes (variable `gls_emerge_phecodes`)~~~
for lv in gls_phenomexcan_lvs:
    #     lv_traits = emerge_projection.loc[lv]
    #     lv_traits = lv_traits[lv_traits > 0.0]
    #     lv_traits = lv_traits.sort_values(ascending=False).head(N_TOP_TRAITS_FROM_LV)

    for phenotype_code in traits:  # + gls_emerge_phecodes):
        phenotypes_lvs_pairs.append(
            {
                "phenotype": phenotype_code,
                "lv": lv,
            }
        )

phenotypes_lvs_pairs = pd.DataFrame(phenotypes_lvs_pairs).drop_duplicates()

# %% tags=[]
phenotypes_lvs_pairs = phenotypes_lvs_pairs.sort_values("phenotype").reset_index(
    drop=True
)

# %% tags=[]
phenotypes_lvs_pairs.shape

# %% tags=[]
phenotypes_lvs_pairs.head()

# %% [markdown] tags=[]
# ## Run

# %% tags=[]
results = []

pbar = tqdm(total=phenotypes_lvs_pairs.shape[0])

for idx, row in phenotypes_lvs_pairs.iterrows():
    phenotype_code = row["phenotype"]
    lv_code = row["lv"]

    pbar.set_description(f"{phenotype_code} - {lv_code}")

    gls_model = GLSPhenoplier(
        smultixcan_result_set_filepath=emerge_smultixcan_zscores_filepath
    )
    gls_model.fit_named(lv_code, phenotype_code)
    res = gls_model.results

    results.append(
        {
            "phenotype": phenotype_code,
            "lv": lv_code,
            "lv_with_pathway": lv_code in well_aligned_lv_codes,
            "coef": res.params.loc["lv"],
            "pvalue": res.pvalues_onesided.loc["lv"],
            "pvalue_twosided": res.pvalues.loc["lv"],
            "summary": gls_model.results_summary,
        }
    )

    # save results every 10 models trained
    if (idx % 10) == 0:
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
