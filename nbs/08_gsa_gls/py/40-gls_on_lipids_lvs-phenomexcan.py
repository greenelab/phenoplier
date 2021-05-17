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
data = pd.read_pickle(input_filepath)

# %% tags=[]
data.shape

# %% tags=[]
data.head()

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
# FIXME hardcoded
deg_enrich = pd.read_csv(
    Path(
        conf.RESULTS["BASE_DIR"],
        "crispr_analyses",
        "fgsea-all_lvs.tsv",
    ).resolve(),
    sep="\t",
)

# %%
deg_enrich.shape

# %%
deg_enrich.head()

# %%
deg_enrich_max_idx = deg_enrich.groupby(["lv", "pathway"])["padj"].idxmax()

# %%
deg_enrich = deg_enrich.loc[deg_enrich_max_idx].reset_index(drop=True)
display(deg_enrich.shape)
display(deg_enrich.head())

# %% [markdown]
# ## Lipids-increasing gene sets

# %%
deg_increase = deg_enrich[
    deg_enrich["pathway"].isin(("gene_set_increase",)) & (deg_enrich["padj"] < 0.05)
].sort_values("padj", ascending=True)

# %%
deg_increase.shape

# %%
deg_increase.head()

# %%
lvs_increase = deg_increase["lv"].unique()

# %%
lvs_increase.shape

# %%
lvs_increase

# %% [markdown]
# ## Lipids-decreasing gene sets

# %%
deg_decrease = deg_enrich[
    deg_enrich["pathway"].isin(("gene_set_decrease",)) & (deg_enrich["padj"] < 0.05)
].sort_values("padj", ascending=True)

# %%
deg_decrease.shape

# %%
deg_decrease.head()

# %%
lvs_decrease = deg_decrease["lv"].unique()

# %%
lvs_decrease.shape

# %%
lvs_decrease

# %% [markdown]
# ## Merge final

# %% tags=[]
_tmp0 = pd.DataFrame({"lv": lvs_increase, "lv_set": "lipids-increasing"})

_tmp1 = pd.DataFrame({"lv": lvs_decrease, "lv_set": "lipids-decreasing"})

# %%
gls_selected_lvs = pd.concat([_tmp0, _tmp1], ignore_index=True)

# %%
gls_selected_lvs.shape

# %%
gls_selected_lvs.head()

# %% [markdown] tags=[]
# # Select traits from specific partition/cluster

# %% tags=[]
PHENOTYPES_CONFIG = [
    # cardiovascular
    (29, 14),
    (29, 16),
    (29, 11),
    (29, 21),
    (29, 17),
]

# %% [markdown] tags=[]
# # GLSPhenoplier

# %% [markdown] tags=[]
# ## Get list of phenotypes/lvs pairs

# %% tags=[]
phenotypes_lvs_pairs = []

for part_k, cluster_id in PHENOTYPES_CONFIG:
    part = best_partitions.loc[part_k, "partition"]

    # get traits
    cluster_traits = data.index[part == cluster_id]

    #     # get extra lvs
    #     lv_list = _get_lvs_data(part_k, cluster_id)

    #     for extra_part_k, extra_cluster_id in extra_for_lvs:
    #         extra_lv_list = _get_lvs_data(part_k, cluster_id)
    #         lv_list.extend(extra_lv_list)

    for phenotype_code in cluster_traits:
        for idx, lv_row in gls_selected_lvs.iterrows():
            phenotypes_lvs_pairs.append(
                {
                    "phenotype_part_k": part_k,
                    "phenotype_cluster_id": cluster_id,
                    "phenotype": phenotype_code,
                    "lv": lv_row["lv"],
                    "lv_set": lv_row["lv_set"],
                }
            )

phenotypes_lvs_pairs = pd.DataFrame(phenotypes_lvs_pairs).drop_duplicates()

# %% tags=[]
phenotypes_lvs_pairs = phenotypes_lvs_pairs.sort_values(
    ["phenotype", "lv"]
).reset_index(drop=True)

# %% tags=[]
phenotypes_lvs_pairs.shape

# %% tags=[]
phenotypes_lvs_pairs.head()

# %% [markdown] tags=[]
# ## Run

# %% tags=[]
output_file = OUTPUT_DIR / "gls_phenotypes-crispr_lvs.pkl"
display(output_file)

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
