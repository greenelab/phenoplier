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
OUTPUT_DIR = conf.RESULTS["GLS"]
display(OUTPUT_DIR)

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# %% [markdown] tags=[]
# # Load data

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
# FIXME: hardcoded
input_filepath = Path(
    "/home/miltondp/projects/labs/greenelab/phenoplier/base/data/emerge",
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
# ## eMERGE (S-MultiXcan)

# %% tags=[]
# FIXME: path hardcoded
emerge_smultixcan_zscores_filepath = Path(
    "/home/miltondp/projects/labs/greenelab/phenoplier/base/data/emerge/gene_assoc/emerge-smultixcan-mashr-zscores.pkl"
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

# %% tags=[]
input_filepath = conf.RESULTS["GLS"] / "gls_phenotypes-crispr_lvs.pkl"
display(input_filepath)

# %% tags=[]
gls_phenomexcan_crispr = pd.read_pickle(input_filepath)

# %% tags=[]
gls_phenomexcan_crispr.shape

# %% tags=[]
gls_phenomexcan_crispr.head()

# %% [markdown] tags=[]
# ## GLS results on eMERGE

# %% tags=[]
# input_filepath = conf.RESULTS["GLS"] / "gls_phenotypes-emerge.pkl"
# display(input_filepath)

# %% tags=[]
# gls_emerge = pd.read_pickle(input_filepath)

# %% tags=[]
# gls_emerge.shape

# %% tags=[]
# gls_emerge.head()

# %% [markdown] tags=[]
# # Select LV from previous GLS run on PhenomeXcan

# %% tags=[]
gls_phenomexcan_lvs = (
    gls_phenomexcan_crispr[["lv", "lv_set"]].drop_duplicates().reset_index(drop=True)
)

# %% tags=[]
gls_phenomexcan_lvs.shape

# %% tags=[]
gls_phenomexcan_lvs.head()

# %% [markdown] tags=[]
# # Select traits from previous GLS run on eMERGE

# %%
emerge_traits_df["phecode_category"].unique()

# %%
gls_traits = emerge_traits_df[
    emerge_traits_df["phecode_category"].isin(
        [
            #     "hematopoietic",
            "circulatory system",
            "endocrine/metabolic",
        ]
    )
]["phecode"].unique()

# %%
gls_traits.shape

# %%
gls_traits

# %% [markdown] tags=[]
# # GLSPhenoplier

# %% [markdown] tags=[]
# ## Get list of phenotypes/lvs pairs

# %% tags=[]
phenotypes_lvs_pairs = []

for idx, lv_row in gls_phenomexcan_lvs.iterrows():
    for phenotype_code in gls_traits:
        phenotypes_lvs_pairs.append(
            {
                "phenotype": phenotype_code,
                "lv": lv_row["lv"],
                "lv_set": lv_row["lv_set"],
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
output_file = OUTPUT_DIR / "gls_phenotypes-emerge-crispr_lvs.pkl"
display(output_file)

# %%
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
            "pvalue": res.pvalues.loc["lv"],
            "summary": gls_model.results_summary,
        }
    )

    if (idx % 10) == 0:
        pd.DataFrame(results).to_pickle(output_file)

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
results.to_pickle(output_file)

# %% tags=[]
