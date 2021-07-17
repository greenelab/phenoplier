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
# This notebook is similar to `40` (using LVs that we found to be significantly enriched for the lipids CRISPR analysis), but traits here are from eMERGE.

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

import pandas as pd
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
OUTPUT_FILENAME = OUTPUT_DIR / "gls_phenotypes-crispr_lvs-emerge.pkl"
display(OUTPUT_FILENAME)

# %% [markdown] tags=[]
# # Load data

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

# %% [markdown]
# ## eMERGE (S-MultiXcan)

# %%
# FIXME: in the future, there will be a specific entry in config for the eMERGE directory that should be replaced here
emerge_smultixcan_zscores_filepath = Path(
    conf.DATA_DIR,
    "emerge",
    "gene_assoc",
    "emerge-smultixcan-mashr-zscores.pkl",
).resolve()

display(emerge_smultixcan_zscores_filepath)

# %%
_tmp = pd.read_pickle(emerge_smultixcan_zscores_filepath)

# %%
_tmp.shape

# %%
_tmp.head()

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
# ## GLS results on PhenomeXcan

# %% tags=[]
input_filepath = conf.RESULTS["GLS"] / "gls_phenotypes-crispr_lvs-phenomexcan.pkl"
display(input_filepath)

# %% tags=[]
gls_phenomexcan_crispr = pd.read_pickle(input_filepath)

# %% tags=[]
gls_phenomexcan_crispr.shape

# %% tags=[]
gls_phenomexcan_crispr.head()

# %% [markdown] tags=[]
# # Select LV from previous GLS run on PhenomeXcan

# %% [markdown] tags=[]
# This selectes the LVs enriched for the lipids CRISPR anaysis.

# %% tags=[]
gls_phenomexcan_lvs = (
    gls_phenomexcan_crispr[["lv", "lv_set"]].drop_duplicates().reset_index(drop=True)
)

# %% tags=[]
gls_phenomexcan_lvs.shape

# %% tags=[]
gls_phenomexcan_lvs.head()

# %% [markdown] tags=[]
# # Select relevant traits from eMERGE

# %% [markdown] tags=[]
# ~Here we don't have partitions/cluters as with PhenomeXcan. So for this analysis (LVs related to lipids), I select relevant categories of traits from eMERGE (we have a "category" column for this results).~

# %% tags=[]
# emerge_traits_df["phecode_category"].unique()

# %% tags=[]
# gls_traits = emerge_traits_df[
#     emerge_traits_df["phecode_category"].isin(
#         [
#             "circulatory system",
#             "endocrine/metabolic",
#             "neurological",
#             "mental disorders",
#         ]
#     )
# ]["phecode"].unique()

# %% tags=[]
# gls_traits.shape

# %% tags=[]
# gls_traits

# %% [markdown] tags=[]
# # GLSPhenoplier

# %% [markdown] tags=[]
# ## Get list of phenotypes/lvs pairs

# %% tags=[]
phenotypes_lvs_pairs = []

# for each LV, I take the top `N_TOP_TRAITS_FROM_LV` traits in eMERGE
for idx, row in gls_phenomexcan_lvs.iterrows():
    lv_name = row["lv"]
    lv_set = row["lv_set"]

    lv_traits = emerge_projection.loc[lv_name]
    lv_traits = lv_traits[lv_traits > 0.0]
    lv_traits = lv_traits.sort_values(ascending=False).head(N_TOP_TRAITS_FROM_LV)

    for phenotype_code in set(lv_traits.index.tolist()):
        phenotypes_lvs_pairs.append(
            {
                "phenotype": phenotype_code,
                "lv": lv_name,
                "lv_set": row["lv_set"],
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
            "lv_set": row["lv_set"],
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
