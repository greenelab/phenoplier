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

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import conf

# %% tags=[]
N_JOBS = conf.GENERAL["N_JOBS"]
display(N_JOBS)

# %% tags=[]
# %env MKL_NUM_THREADS=$N_JOBS
# %env OPEN_BLAS_NUM_THREADS=$N_JOBS
# %env NUMEXPR_NUM_THREADS=$N_JOBS
# %env OMP_NUM_THREADS=$N_JOBS

# %% [markdown]
# # Modules

# %%
from pathlib import Path

import statsmodels.api as sm
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from tqdm import tqdm

from gls import GLSPhenoplier

# %% [markdown]
# # Settings

# %%
OUTPUT_DIR = conf.RESULTS["GLS"]
display(OUTPUT_DIR)

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# # Load data

# %% [markdown]
# ## eMERGE to PhenomeXcan maps

# %%
# FIXME: hardcoded
input_filepath = Path(
    "/home/miltondp/projects/labs/greenelab/phenoplier/base/data/emerge",
    "eMERGE_III_PMBB_GSA_v2_2020_phecode_AFR_EUR_cc50_counts_w_dictionary.txt",
).resolve()
display(input_filepath)

# %%
emerge_traits_df = pd.read_csv(
    input_filepath,
    sep="\t",
    dtype={"phecode": str},
    usecols=["phecode", "phenotype", "category"],
)

# %%
emerge_traits_df = emerge_traits_df.rename(
    columns={
        "phenotype": "phecode_phenotype",
        "category": "phecode_category",
    }
)

# %%
emerge_traits_df.shape

# %%
emerge_traits_df.head()

# %% [markdown]
# ## eMERGE to PhenomeXcan maps

# %%
# FIXME: hardcoded
emerge_phenomexcan_maps_filepath = Path(
    "/home/miltondp/projects/labs/greenelab/phenoplier/base/data/emerge",
    "phecodes_phenomexcan_maps.tsv",
).resolve()
display(emerge_phenomexcan_maps_filepath)

# %%
emerge_phenomexcan_maps = pd.read_csv(
    emerge_phenomexcan_maps_filepath, sep="\t", dtype={"phecode": str}
)

# %%
emerge_phenomexcan_maps = emerge_phenomexcan_maps.dropna(
    subset=["phecode", "phenomexcan"], how="any"
)

# %%
emerge_phenomexcan_maps.shape

# %%
emerge_phenomexcan_maps.head()

# %% [markdown]
# ## eMERGE (S-MultiXcan) projection

# %%
# FIXME hardcoded
input_filepath = Path(
    "/home/miltondp/projects/labs/greenelab/phenoplier/base/results/projections",
    "projection-emerge-smultixcan-mashr-zscores.pkl",
).resolve()
display(input_filepath)

# %%
emerge_projection = pd.read_pickle(input_filepath)

# %%
emerge_projection.shape

# %%
emerge_projection.head()

# %% [markdown]
# ## eMERGE (S-MultiXcan) projection

# %%
# FIXME: path hardcoded
emerge_smultixcan_projection_filepath = Path(
    "/home/miltondp/projects/labs/greenelab/phenoplier/base/data/emerge/gene_assoc/emerge-smultixcan-mashr-zscores.pkl"
).resolve()

display(emerge_smultixcan_projection_filepath)

# %%
_tmp = pd.read_pickle(emerge_smultixcan_projection_filepath)

# %%
_tmp.shape

# %%
_tmp.head()

# %% [markdown]
# ## GLS results on PhenomeXcan

# %%
input_filepath = conf.RESULTS["GLS"] / "gls_phenotypes.pkl"
display(input_filepath)

# %%
gls_phenomexcan = pd.read_pickle(input_filepath)

# %%
gls_phenomexcan.shape

# %%
gls_phenomexcan.head()

# %% [markdown]
# ## MultiPLIER summary

# %%
multiplier_model_summary = pd.read_pickle(conf.MULTIPLIER["MODEL_SUMMARY_FILE"])

# %%
multiplier_model_summary.shape

# %%
multiplier_model_summary.head()

# %%
well_aligned_lvs = multiplier_model_summary[
    (multiplier_model_summary["FDR"] < 0.05) | (multiplier_model_summary["AUC"] >= 0.75)
]

display(well_aligned_lvs.shape)
display(well_aligned_lvs.head())

# %%
well_aligned_lv_codes = set([f"LV{lvi}" for lvi in well_aligned_lvs["LV index"]])

# %%
len(well_aligned_lv_codes)

# %%
list(well_aligned_lv_codes)[:5]

# %% [markdown]
# # Select LV from previous GLS run on PhenomeXcan

# %%
gls_phenomexcan_lvs = gls_phenomexcan["lv"].unique()

# %%
gls_phenomexcan_lvs.shape

# %%
gls_phenomexcan_lvs

# %% [markdown]
# # Select eMERGE traits

# %%
gls_phenomexcan_traits = gls_phenomexcan["phenotype"].unique()

# %%
gls_phenomexcan_traits.shape

# %%
gls_phenomexcan_in_emerge = emerge_phenomexcan_maps[
    (emerge_phenomexcan_maps["efo"].isin(gls_phenomexcan_traits))
    | (emerge_phenomexcan_maps["phenomexcan"].isin(gls_phenomexcan_traits))
]

# %%
gls_phenomexcan_in_emerge

# %%
gls_emerge_phecodes = gls_phenomexcan_in_emerge["phecode"].unique().tolist()

# %%
# these are the mapped traits from PhenomeXcan to phecodes
gls_emerge_phecodes

# %%
# phecode_to_desc_map = emerge_traits_df[["phecode", "phecode_phenotype"]].set_index("phecode").squeeze().to_dict()

# %% [markdown]
# # GLSPhenoplier

# %% [markdown]
# ## Get list of phenotypes/lvs pairs

# %%
phenotypes_lvs_pairs = []

# for lvs run for PhenomeXcan, I take the top traits in eMERGE + global mapped phenotypes
for lv in gls_phenomexcan_lvs:
    lv_traits = emerge_projection.loc[lv]
    lv_traits = lv_traits[lv_traits > 0.0]
    lv_traits = lv_traits.sort_values(ascending=False).head(20)

    for phenotype_code in set(lv_traits.index.tolist() + gls_emerge_phecodes):
        phenotypes_lvs_pairs.append(
            {
                "phenotype": phenotype_code,
                "lv": lv,
            }
        )

phenotypes_lvs_pairs = pd.DataFrame(phenotypes_lvs_pairs).drop_duplicates()

# %%
phenotypes_lvs_pairs = phenotypes_lvs_pairs.sort_values("phenotype").reset_index(
    drop=True
)

# %%
phenotypes_lvs_pairs.shape

# %%
phenotypes_lvs_pairs.head()

# %% [markdown]
# ## Run

# %%
output_file = OUTPUT_DIR / "gls_phenotypes-emerge.pkl"
display(output_file)

# %%
results = []

pbar = tqdm(total=phenotypes_lvs_pairs.shape[0])

for idx, row in phenotypes_lvs_pairs.iterrows():
    phenotype_code = row["phenotype"]
    lv_code = row["lv"]

    pbar.set_description(f"{phenotype_code} - {lv_code}")

    gls_model = GLSPhenoplier(
        smultixcan_result_set_filepath=emerge_smultixcan_projection_filepath
    )
    gls_model.fit_named(lv_code, phenotype_code)
    res = gls_model.results

    results.append(
        {
            "phenotype": phenotype_code,
            "lv": lv_code,
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

# %%
results = pd.DataFrame(results)

# %%
results.shape

# %%
results.head()

# %%
results.sort_values("pvalue").head(10)

# %% [markdown]
# ## Save

# %%
results.to_pickle(output_file)

# %%
