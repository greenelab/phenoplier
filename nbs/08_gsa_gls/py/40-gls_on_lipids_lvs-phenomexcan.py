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
# This notebook is similar to `30` and `35`, but here I use the LVs that we found to be significantly enriched for the lipids CRISPR analysis, which might or might not coincide with the previously used LVs (those that discriminate clusters).
# The traits here are from PhenomeXcan.

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
OUTPUT_FILENAME = OUTPUT_DIR / "gls_phenotypes-crispr_lvs-phenomexcan.pkl"
display(OUTPUT_FILENAME)

# %% [markdown] tags=[]
# # Load data

# %% [markdown] tags=[]
# ## PhenomeXcan (S-MultiXcan)

# %% tags=[]
INPUT_SUBSET = "z_score_std"

# %% tags=[]
INPUT_STEM = "projection-smultixcan-efo_partial-mashr-zscores"

# %% tags=[]
input_filepath = Path(
    conf.RESULTS["DATA_TRANSFORMATIONS_DIR"],
    INPUT_SUBSET,
    f"{INPUT_SUBSET}-{INPUT_STEM}.pkl",
).resolve()

# %% tags=[]
phenomexcan_projection = pd.read_pickle(input_filepath)

# %% tags=[]
phenomexcan_projection.shape

# %% tags=[]
phenomexcan_projection.head()

# %% [markdown] tags=[]
# ## Clustering results

# %% tags=[]
CONSENSUS_CLUSTERING_DIR = Path(
    conf.RESULTS["CLUSTERING_DIR"], "consensus_clustering"
).resolve()

display(CONSENSUS_CLUSTERING_DIR)

# %% tags=[]
input_file = Path(CONSENSUS_CLUSTERING_DIR, "best_partitions_by_k.pkl").resolve()
display(input_file)

# %% tags=[]
best_partitions = pd.read_pickle(input_file)

# %% tags=[]
best_partitions.shape

# %% tags=[]
best_partitions.head()

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
# # Select LVs from CRISPR analysis

# %% tags=[]
# FIXME: there will be a specific folder for crispr analysis in the future, that should be replaced here
deg_enrich = pd.read_csv(
    Path(
        conf.RESULTS["BASE_DIR"],
        "crispr_analyses",
        "fgsea-hi_conf-all_lvs.tsv",
    ).resolve(),
    sep="\t",
)

# %% tags=[]
deg_enrich.shape

# %% tags=[]
deg_enrich.head()

# %% tags=[]
# deg_enrich_max_idx = deg_enrich.groupby(["lv", "pathway"])["pval"].idxmax()

# %% tags=[]
# deg_enrich = deg_enrich.loc[deg_enrich_max_idx].reset_index(drop=True)
# display(deg_enrich.shape)
# display(deg_enrich.head())

# %% [markdown] tags=[]
# ## Lipids-increasing gene sets

# %% tags=[]
deg_increase = deg_enrich[
    deg_enrich["pathway"].isin(("gene_set_increase",)) & (deg_enrich["pval"] < 0.01)
].sort_values("pval", ascending=True)

# %% tags=[]
deg_increase.shape

# %% tags=[]
deg_increase.head()

# %% tags=[]
lvs_increase = deg_increase["lv"].unique()

# %% tags=[]
lvs_increase.shape

# %% tags=[]
lvs_increase

# %% [markdown] tags=[]
# ## Lipids-decreasing gene sets

# %% tags=[]
deg_decrease = deg_enrich[
    deg_enrich["pathway"].isin(("gene_set_decrease",)) & (deg_enrich["pval"] < 0.01)
].sort_values("pval", ascending=True)

# %% tags=[]
deg_decrease.shape

# %% tags=[]
deg_decrease.head()

# %% tags=[]
lvs_decrease = deg_decrease["lv"].unique()

# %% tags=[]
lvs_decrease.shape

# %% tags=[]
lvs_decrease

# %% [markdown] tags=[]
# ## Merge into one dataframe

# %% tags=[]
_tmp0 = pd.DataFrame({"lv": lvs_increase, "lv_set": "lipids-increasing"})

_tmp1 = pd.DataFrame({"lv": lvs_decrease, "lv_set": "lipids-decreasing"})

# %% tags=[]
gls_selected_lvs = pd.concat([_tmp0, _tmp1], ignore_index=True)

# %% tags=[]
gls_selected_lvs.shape

# %% tags=[]
gls_selected_lvs.head()

# %% [markdown] tags=[]
# # Select traits from specific partition/cluster

# %% [markdown] tags=[]
# For this run on the LVs related to the lipids CRISPR analysis, I'm only interested in the main clusters of the cardiovascular sub-branch.

# %% tags=[]
# PHENOTYPES_CONFIG = [
#     # cardiovascular
#     (29, 14),
#     (29, 16),
#     (29, 11),
#     (29, 21),
#     (29, 17),
# ]

# %% [markdown] tags=[]
# # GLSPhenoplier

# %% [markdown] tags=[]
# ## Get list of phenotypes/lvs pairs

# %% tags=[]
phenotypes_lvs_pairs = []

# for each LV, I take the top `N_TOP_TRAITS_FROM_LV` traits in eMERGE
for idx, row in gls_selected_lvs.iterrows():
    lv_name = row["lv"]
    lv_set = row["lv_set"]

    lv_traits = phenomexcan_projection.loc[lv_name]
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
        smultixcan_result_set_filepath=conf.PHENOMEXCAN[
            "SMULTIXCAN_EFO_PARTIAL_MASHR_ZSCORES_FILE"
        ]
    )
    gls_model.fit_named(lv_code, phenotype_code)
    res = gls_model.results

    results.append(
        {
            "part_k": row["phenotype_part_k"],
            "cluster_id": row["phenotype_cluster_id"],
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
