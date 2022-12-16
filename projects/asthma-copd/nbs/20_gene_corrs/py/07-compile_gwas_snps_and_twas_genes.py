# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
#     formats: ipynb,py//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] tags=[]
# # Description

# %% [markdown] tags=[]
# (Please, take a look at the README.md file in this directory for instructions on how to run this notebook)
#
# This notebook compiles information about the GWAS and TWAS for a particular cohort. For example, the set of GWAS variants, variance of predicted expression of genes, etc.
#
# It has specicfic parameters for papermill (see under `Settings` below).
#
# This notebook is not directly run. See README.md.

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

import conf
from entity import Gene

# %% [markdown] tags=[]
# # Settings

# %% tags=["parameters"]
# a cohort name (it could be something like UK_BIOBANK, etc)
COHORT_NAME = None

# reference panel such as 1000G or GTEX_V8
REFERENCE_PANEL = "GTEX_V8"

# predictions models such as MASHR or ELASTIC_NET
EQTL_MODEL = "MASHR"

# a string with a path pointing to an imputed GWAS
GWAS_FILE = None

# a string with a path pointing where S-PrediXcan results (tissue-specific are located
SPREDIXCAN_FOLDER = None

# an f-string with one placeholder {tissue}
SPREDIXCAN_FILE_PATTERN = None

# a string with a path pointing to an S-MultiXcan result
SMULTIXCAN_FILE = None

# output dir
OUTPUT_DIR_BASE = None

# %%
assert COHORT_NAME is not None and len(COHORT_NAME) > 0, "A cohort name must be given"

COHORT_NAME = COHORT_NAME.lower()
display(f"Cohort name: {COHORT_NAME}")

# %%
assert (
    REFERENCE_PANEL is not None and len(REFERENCE_PANEL) > 0
), "A reference panel must be given"

display(f"Reference panel: {REFERENCE_PANEL}")

# %%
assert GWAS_FILE is not None and len(GWAS_FILE) > 0, "A GWAS file path must be given"
GWAS_FILE = Path(GWAS_FILE).resolve()
assert GWAS_FILE.exists(), "GWAS file does not exist"

display(f"GWAS file path: {str(GWAS_FILE)}")

# %%
assert (
    SPREDIXCAN_FOLDER is not None and len(SPREDIXCAN_FOLDER) > 0
), "An S-PrediXcan folder path must be given"
SPREDIXCAN_FOLDER = Path(SPREDIXCAN_FOLDER).resolve()
assert (
    SPREDIXCAN_FOLDER.exists()
), f"S-PrediXcan folder does not exist: {str(SPREDIXCAN_FOLDER)}"

display(f"S-PrediXcan folder path: {str(SPREDIXCAN_FOLDER)}")

# %%
assert (
    SPREDIXCAN_FILE_PATTERN is not None and len(SPREDIXCAN_FILE_PATTERN) > 0
), "An S-PrediXcan file pattern must be given"
assert (
    "{tissue}" in SPREDIXCAN_FILE_PATTERN
), "S-PrediXcan file pattern must have a '{tissue}' placeholder"

display(f"S-PrediXcan file template: {SPREDIXCAN_FILE_PATTERN}")

# %%
assert (
    SMULTIXCAN_FILE is not None and len(SMULTIXCAN_FILE) > 0
), "An S-MultiXcan result file path must be given"
SMULTIXCAN_FILE = Path(SMULTIXCAN_FILE).resolve()
assert (
    SMULTIXCAN_FILE.exists()
), f"S-MultiXcan result file does not exist: {str(SMULTIXCAN_FILE)}"

display(f"S-MultiXcan file path: {str(SMULTIXCAN_FILE)}")

# %%
assert (
    EQTL_MODEL is not None and len(EQTL_MODEL) > 0
), "A prediction/eQTL model must be given"

display(f"eQTL model: {EQTL_MODEL}")

# %%
assert (
    OUTPUT_DIR_BASE is not None and len(OUTPUT_DIR_BASE) > 0
), "Output directory path must be given"

OUTPUT_DIR_BASE = (
    Path(OUTPUT_DIR_BASE)
    / "gene_corrs"
    / COHORT_NAME
    # / REFERENCE_PANEL.lower()
    # / EQTL_MODEL.lower()
).resolve()

OUTPUT_DIR_BASE.mkdir(parents=True, exist_ok=True)

display(f"Using output dir base: {OUTPUT_DIR_BASE}")

# %% [markdown] tags=[]
# # Load MultiPLIER Z genes

# %% tags=[]
multiplier_z_genes = pd.read_pickle(
    conf.MULTIPLIER["MODEL_Z_MATRIX_FILE"]
).index.tolist()

# %% tags=[]
len(multiplier_z_genes)

# %%
assert len(multiplier_z_genes) == len(set(multiplier_z_genes))

# %% tags=[]
multiplier_z_genes[:5]

# %% [markdown] tags=[]
# # GWAS

# %%
gwas_file_columns = pd.read_csv(GWAS_FILE, sep="\t", nrows=2).columns

assert (
    "panel_variant_id" in gwas_file_columns
), "The GWAS file must be the final imputed one using the TWAS imputation tools with column 'panel_variant_id'"

assert (
    "pvalue" in gwas_file_columns
), "The GWAS file must be the final imputed one using the TWAS imputation tools with column 'pvalue'"

assert (
    "zscore" in gwas_file_columns
), "The GWAS file must be the final imputed one using the TWAS imputation tools with column 'zscore'"

# %%
gwas_data = pd.read_csv(
    GWAS_FILE,
    sep="\t",
    usecols=["panel_variant_id", "pvalue", "zscore"],
)

# %%
gwas_data.shape

# %%
gwas_data.head()

# %%
gwas_data.dropna().shape

# %%
# remove SNPs with no results
gwas_data = gwas_data.dropna()

# %%
gwas_data.shape

# %% [markdown] tags=[]
# ## Save GWAS variants

# %%
gwas_data.head()

# %%
# in eMERGE's results, some values here are repeated (will be removed later by taking the unique set of variant IDs).
gwas_data["panel_variant_id"].is_unique

# %%
gwas_variants_ids_set = frozenset(gwas_data["panel_variant_id"])
list(gwas_variants_ids_set)[:5]

# %%
with open(OUTPUT_DIR_BASE / "gwas_variant_ids.pkl", "wb") as handle:
    pickle.dump(gwas_variants_ids_set, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %% [markdown] tags=[]
# # TWAS

# %% [markdown] tags=[]
# ## Available tissues for eQTL model

# %%
prediction_model_tissues = conf.PHENOMEXCAN["PREDICTION_MODELS"][
    f"{EQTL_MODEL}_TISSUES"
].split(" ")

# %%
len(prediction_model_tissues)

# %%
prediction_model_tissues[:5]

# %% [markdown] tags=[]
# ## S-MultiXcan results

# %%
smultixcan_results = pd.read_csv(
    SMULTIXCAN_FILE, sep="\t", usecols=["gene", "gene_name", "pvalue", "n", "n_indep"]
)

# %%
smultixcan_results.shape

# %%
smultixcan_results = smultixcan_results.dropna()

# %%
smultixcan_results.shape

# %%
smultixcan_results = smultixcan_results.assign(
    gene_id=smultixcan_results["gene"].apply(lambda g: g.split(".")[0])
)

# %%
smultixcan_results.head()

# %%
assert smultixcan_results["gene_id"].is_unique

# %% [markdown] tags=[]
# ### Get common genes with MultiPLIER

# %%
common_genes = set(multiplier_z_genes).intersection(
    set(smultixcan_results["gene_name"])
)

# %%
len(common_genes)

# %%
sorted(list(common_genes))[:5]

# %% [markdown] tags=[]
# ## Genes info

# %% tags=[]
multiplier_gene_obj = {
    gene_name: Gene(name=gene_name)
    for gene_name in common_genes
    if gene_name in Gene.GENE_NAME_TO_ID_MAP
}

# %%
# delete common_genes, from now on, genes_info should be used for common genes
del common_genes

# %% tags=[]
len(multiplier_gene_obj)

# %% tags=[]
assert multiplier_gene_obj["GAS6"].ensembl_id == "ENSG00000183087"

# %% tags=[]
_gene_obj = list(multiplier_gene_obj.values())

genes_info = pd.DataFrame(
    {
        "name": [g.name for g in _gene_obj],
        "id": [g.ensembl_id for g in _gene_obj],
        "chr": [g.chromosome for g in _gene_obj],
        "band": [g.band for g in _gene_obj],
        "start_position": [g.get_attribute("start_position") for g in _gene_obj],
        "end_position": [g.get_attribute("end_position") for g in _gene_obj],
    }
)

# %%
genes_info = genes_info.assign(
    gene_length=genes_info.apply(
        lambda x: x["end_position"] - x["start_position"], axis=1
    )
)

# %%
genes_info.dtypes

# %%
_tmp = genes_info[genes_info.isna().any(axis=1)]
display(_tmp)
assert _tmp.shape[0] < 5

# %%
genes_info = genes_info.dropna()

# %%
genes_info["chr"] = genes_info["chr"].apply(pd.to_numeric, downcast="integer")
genes_info["start_position"] = genes_info["start_position"].astype(int)
genes_info["end_position"] = genes_info["end_position"].astype(int)
genes_info["gene_length"] = genes_info["gene_length"].astype(int)

# %%
genes_info.dtypes

# %%
assert genes_info["name"].is_unique

# %%
assert genes_info["id"].is_unique

# %% tags=[]
genes_info.shape

# %% tags=[]
genes_info.head()

# %%
genes_info.sort_values("chr")

# %% [markdown]
# ### Save

# %%
genes_info.to_pickle(OUTPUT_DIR_BASE / "genes_info.pkl")

# %% [markdown] tags=[]
# ## S-PrediXcan results

# %% [markdown] tags=[]
# ### Load results across all tissues

# %%
spredixcan_result_files = {
    t: SPREDIXCAN_FOLDER / SPREDIXCAN_FILE_PATTERN.format(tissue=t)
    for t in prediction_model_tissues
}

# %%
assert len(spredixcan_result_files) == len(prediction_model_tissues)
display(list(spredixcan_result_files.values())[:5])

# %%
# look at the structure of one result
pd.read_csv(spredixcan_result_files["Whole_Blood"]).head()

# %%
assert all(f.exists() for f in spredixcan_result_files.values())

# %%
spredixcan_dfs = [
    pd.read_csv(
        f,
        usecols=[
            "gene",
            "zscore",
            "pvalue",
            "n_snps_used",
            "n_snps_in_model",
        ],
    )
    .dropna(subset=["gene", "zscore", "pvalue"])
    .assign(tissue=t)
    for t, f in spredixcan_result_files.items()
]

# %%
assert len(spredixcan_dfs) == len(prediction_model_tissues)

# %%
spredixcan_dfs = pd.concat(spredixcan_dfs)

# %%
assert spredixcan_dfs["tissue"].unique().shape[0] == len(prediction_model_tissues)

# %%
spredixcan_dfs.shape

# %%
spredixcan_dfs = spredixcan_dfs.assign(
    gene_id=spredixcan_dfs["gene"].apply(lambda g: g.split(".")[0])
)

# %%
spredixcan_dfs.head()

# %%
# leave only common genes
spredixcan_dfs = spredixcan_dfs[spredixcan_dfs["gene_id"].isin(set(genes_info["id"]))]

# %%
spredixcan_dfs.shape

# %% [markdown] tags=[]
# ### Count number of tissues available per gene

# %%
spredixcan_genes_n_models = spredixcan_dfs.groupby("gene_id")["tissue"].nunique()

# %%
spredixcan_genes_n_models

# %%
# testing that in S-MultiXcan I get the same number of tissues per gene
_tmp_smultixcan_results_n_models = (
    smultixcan_results.set_index("gene_id")["n"].astype(int).rename("tissue")
)

_cg = _tmp_smultixcan_results_n_models.index.intersection(
    spredixcan_genes_n_models.index
)
_tmp_smultixcan_results_n_models = _tmp_smultixcan_results_n_models.loc[_cg]
_spredixcan = spredixcan_genes_n_models.loc[_cg]

assert _spredixcan.shape[0] == _tmp_smultixcan_results_n_models.shape[0]
assert _spredixcan.equals(_tmp_smultixcan_results_n_models.loc[_spredixcan.index])

# %% [markdown] tags=[]
# ### Get tissues available per gene

# %%
spredixcan_genes_models = spredixcan_dfs.groupby("gene_id")["tissue"].apply(
    lambda x: frozenset(x.tolist())
)

# %%
spredixcan_genes_models

# %%
assert spredixcan_genes_n_models.shape[0] == spredixcan_genes_models.shape[0]

# %%
assert spredixcan_genes_n_models.index.equals(spredixcan_genes_models.index)

# %%
assert (spredixcan_genes_models.apply(len) <= len(prediction_model_tissues)).all()

# %%
spredixcan_genes_models.apply(len).describe()

# %%
# testing that I obtained the right number of tissues
assert (
    spredixcan_genes_models.loc[spredixcan_genes_n_models.index]
    .apply(len)
    .equals(spredixcan_genes_n_models)
)

# %% [markdown]
# ### Add gene name and set index

# %%
spredixcan_genes_models = spredixcan_genes_models.to_frame().reset_index()

# %%
spredixcan_genes_models.head()

# %%
spredixcan_genes_models = spredixcan_genes_models.assign(
    gene_name=spredixcan_genes_models["gene_id"].apply(
        lambda g: Gene.GENE_ID_TO_NAME_MAP[g]
    )
)

# %%
spredixcan_genes_models = spredixcan_genes_models[["gene_id", "gene_name", "tissue"]]

# %%
spredixcan_genes_models = spredixcan_genes_models.set_index("gene_id")

# %%
spredixcan_genes_models.head()

# %% [markdown]
# ### Add number of tissues

# %%
spredixcan_genes_models = spredixcan_genes_models.assign(
    n_tissues=spredixcan_genes_models["tissue"].apply(len)
)

# %%
spredixcan_genes_models.head()

# %% [markdown]
# ### Save

# %% [markdown] tags=[]
# Here I quickly save these results to a file, given that the next steps (covariates) are slow to compute.

# %%
# this is important, other scripts depend on gene_name to be unique
assert spredixcan_genes_models["gene_name"].is_unique

# %%
assert not spredixcan_genes_models.isna().any(None)

# %%
spredixcan_genes_models.to_pickle(OUTPUT_DIR_BASE / "gene_tissues.pkl")

# %% [markdown] tags=[]
# ## Add covariates based on S-PrediXcan results

# %% [markdown] tags=[]
# This extend the previous file with more columns

# %% [markdown]
# ### Get gene's objects

# %%
spredixcan_gene_obj = {
    gene_id: Gene(ensembl_id=gene_id) for gene_id in spredixcan_genes_models.index
}

# %%
len(spredixcan_gene_obj)

# %% [markdown]
# ### Add genes' variance captured by principal components

# %%
# def _get_gene_pc_variance(gene_row):
#     gene_id = gene_row.name
#     gene_tissues = gene_row["tissue"]
#     gene_obj = spredixcan_gene_obj[gene_id]

#     u, s, vt = gene_obj.get_tissues_correlations_svd(
#         tissues=gene_tissues,
#         snps_subset=gwas_variants_ids_set,
#         reference_panel=REFERENCE_PANEL,
#         model_type=EQTL_MODEL,
#         # use_covariance_matrix=True,
#     )

#     return s

# %%
# _tmp = spredixcan_genes_models.loc["ENSG00000188976"]
# _get_gene_pc_variance(_tmp)

# %%
# spredixcan_genes_tissues_pc_variance = spredixcan_genes_models.apply(
#     _get_gene_pc_variance, axis=1
# )

# %%
# spredixcan_genes_tissues_pc_variance

# %%
# # testing
# assert spredixcan_genes_tissues_pc_variance.loc[
#     "ENSG00000188976"
# ].sum() == pytest.approx(44.01605629086847)
# # this is using the covariance:
# # assert spredixcan_genes_tissues_pc_variance.loc["ENSG00000188976"].sum() == pytest.approx(1.1492946006449425)

# %%
# # add to spredixcan_genes_models
# spredixcan_genes_models = spredixcan_genes_models.join(
#     spredixcan_genes_tissues_pc_variance.rename("tissues_pc_variances")
# )

# %%
# spredixcan_genes_models.shape

# %%
# spredixcan_genes_models.head()

# %% [markdown]
# ### Add gene variance per tissue

# %%
# def _get_gene_variances(gene_row):
#     gene_id = gene_row.name
#     gene_tissues = gene_row["tissue"]

#     tissue_variances = {}
#     gene_obj = spredixcan_gene_obj[gene_id]

#     for tissue in gene_tissues:
#         tissue_var = gene_obj.get_pred_expression_variance(
#             tissue=tissue,
#             reference_panel=REFERENCE_PANEL,
#             model_type=EQTL_MODEL,
#             snps_subset=gwas_variants_ids_set,
#         )

#         if tissue_var is not None:
#             tissue_variances[tissue] = tissue_var

#     return tissue_variances

# %%
# _tmp = spredixcan_genes_models.loc["ENSG00000000419"]
# _get_gene_variances(_tmp)

# %%
# spredixcan_genes_tissues_variance = spredixcan_genes_models.apply(
#     _get_gene_variances, axis=1
# )

# %%
# spredixcan_genes_tissues_variance

# %%
# # testing
# _gene_id = "ENSG00000188976"
# x = spredixcan_genes_tissues_variance.loc[_gene_id]
# # expected value obtained by sum of PCA eigenvalues on this gene's predicted expression
# assert np.sum(list(x.values())) == pytest.approx(1.2326202607409493)

# %%
# # testing
# spredixcan_genes_tissues_variance.loc["ENSG00000000419"]

# %%
# # add to spredixcan_genes_models
# spredixcan_genes_models = spredixcan_genes_models.join(
#     spredixcan_genes_tissues_variance.rename("tissues_variances")
# )

# %%
# spredixcan_genes_models.shape

# %%
# spredixcan_genes_models.head()

# %% [markdown]
# ### Count number of SNPs predictors used across tissue models

# %%
spredixcan_genes_sum_of_n_snps_used = (
    spredixcan_dfs.groupby("gene_id")["n_snps_used"].sum().rename("n_snps_used_sum")
)

# %%
spredixcan_genes_sum_of_n_snps_used

# %%
# add sum of snps used to spredixcan_genes_models
spredixcan_genes_models = spredixcan_genes_models.join(
    spredixcan_genes_sum_of_n_snps_used
)

# %%
spredixcan_genes_models.shape

# %%
spredixcan_genes_models.head()

# %% [markdown]
# ### Count number of SNPs predictors in models across tissue models

# %%
spredixcan_genes_sum_of_n_snps_in_model = (
    spredixcan_dfs.groupby("gene_id")["n_snps_in_model"]
    .sum()
    .rename("n_snps_in_model_sum")
)

# %%
spredixcan_genes_sum_of_n_snps_in_model

# %%
# add sum of snps in model to spredixcan_genes_models
spredixcan_genes_models = spredixcan_genes_models.join(
    spredixcan_genes_sum_of_n_snps_in_model
)

# %%
spredixcan_genes_models.shape

# %%
spredixcan_genes_models.head()

# %% [markdown]
# ### Summarize prediction models for each gene

# %%
# def _summarize_gene_models(gene_id):
#     """
#     For a given gene ID, it returns a dataframe with predictor SNPs in rows and tissues in columns, where
#     values are the weights of SNPs in those tissues.
#     It can contain NaNs.
#     """
#     gene_obj = spredixcan_gene_obj[gene_id]
#     gene_tissues = spredixcan_genes_models.loc[gene_id, "tissue"]

#     gene_models = {}
#     gene_unique_snps = set()
#     for t in gene_tissues:
#         gene_model = gene_obj.get_prediction_weights(tissue=t, model_type=EQTL_MODEL)
#         gene_models[t] = gene_model

#         gene_unique_snps.update(set(gene_model.index))

#     df = pd.DataFrame(
#         data=np.nan, index=list(gene_unique_snps), columns=list(gene_tissues)
#     )

#     for t in df.columns:
#         for snp in df.index:
#             gene_model = gene_models[t]

#             if snp in gene_model.index:
#                 df.loc[snp, t] = gene_model.loc[snp]

#     return df

# %%
# # testing
# spredixcan_gene_obj["ENSG00000000419"].get_prediction_weights(
#     tissue="Brain_Hypothalamus", model_type=EQTL_MODEL
# )

# %%
# spredixcan_gene_obj["ENSG00000000419"].get_prediction_weights(
#     tissue="Brain_Substantia_nigra", model_type=EQTL_MODEL
# )

# %%
# # testing
# _gene_id = "ENSG00000000419"

# _gene_model = _summarize_gene_models(_gene_id)
# assert (
#     _gene_model.loc["chr20_50862947_C_T_b38", "Brain_Hypothalamus"].round(5) == 0.43138
# )
# assert pd.isnull(_gene_model.loc["chr20_50957480_C_T_b38", "Brain_Hypothalamus"])

# assert pd.isnull(_gene_model.loc["chr20_50862947_C_T_b38", "Brain_Substantia_nigra"])
# assert (
#     _gene_model.loc["chr20_50957480_C_T_b38", "Brain_Substantia_nigra"].round(5)
#     == -0.1468
# )

# %%
# gene_models = {}

# for gene_id in spredixcan_genes_models.index:
#     gene_models[gene_id] = _summarize_gene_models(gene_id)

# %%
# # testing
# _gene_id = "ENSG00000000419"

# _gene_model = gene_models[_gene_id]
# assert (
#     _gene_model.loc["chr20_50862947_C_T_b38", "Brain_Hypothalamus"].round(5) == 0.43138
# )
# assert pd.isnull(_gene_model.loc["chr20_50957480_C_T_b38", "Brain_Hypothalamus"])

# assert pd.isnull(_gene_model.loc["chr20_50862947_C_T_b38", "Brain_Substantia_nigra"])
# assert (
#     _gene_model.loc["chr20_50957480_C_T_b38", "Brain_Substantia_nigra"].round(5)
#     == -0.1468
# )

# %%
# # save
# import gzip

# with gzip.GzipFile(OUTPUT_DIR_BASE / "gene_tissues_models.pkl.gz", "w") as f:
#     pickle.dump(gene_models, f)

# %%
# # testing saved file
# with gzip.GzipFile(OUTPUT_DIR_BASE / "gene_tissues_models.pkl.gz", "r") as f:
#     _tmp = pickle.load(f)

# %%
# assert len(gene_models) == len(_tmp)
# assert gene_models["ENSG00000000419"].equals(_tmp["ENSG00000000419"])

# %% [markdown]
# ### Count number of _unique_ SNPs predictors used and available across tissue models

# %%
# def _count_unique_snps(gene_id):
#     """
#     For a gene_id, it counts unique SNPs in all models and their intersection with GWAS SNPs (therefore, used by S-PrediXcan).
#     """
#     gene_tissues = spredixcan_genes_models.loc[gene_id, "tissue"]

#     gene_unique_snps = set()
#     for t in gene_tissues:
#         t_snps = set(gene_models[gene_id].index)
#         gene_unique_snps.update(t_snps)

#     gene_unique_snps_in_gwas = gwas_variants_ids_set.intersection(gene_unique_snps)

#     return pd.Series(
#         {
#             "unique_n_snps_in_model": len(gene_unique_snps),
#             "unique_n_snps_used": len(gene_unique_snps_in_gwas),
#         }
#     )

# %%
# # testing
# spredixcan_genes_models[spredixcan_genes_models["n_snps_used_sum"] == 2].head()

# %%
# # case with two snps, not repeated across tissues
# _gene_id = "ENSG00000000419"
# display(
#     spredixcan_gene_obj[_gene_id].get_prediction_weights(
#         tissue="Brain_Hypothalamus", model_type=EQTL_MODEL
#     )
# )
# display(
#     spredixcan_gene_obj[_gene_id].get_prediction_weights(
#         tissue="Brain_Substantia_nigra", model_type=EQTL_MODEL
#     )
# )

# %%
# _tmp = _count_unique_snps(_gene_id)
# assert _tmp.shape[0] == 2
# assert _tmp["unique_n_snps_in_model"] == 2
# assert _tmp["unique_n_snps_used"] == 2

# %%
# # get unique snps for all genes
# spredixcan_genes_unique_n_snps = spredixcan_genes_models.groupby("gene_id").apply(
#     lambda x: _count_unique_snps(x.name)
# )

# %%
# spredixcan_genes_unique_n_snps.head()

# %%
# assert (
#     spredixcan_genes_unique_n_snps["unique_n_snps_in_model"]
#     >= spredixcan_genes_unique_n_snps["unique_n_snps_used"]
# ).all()

# %%
# # add unique snps to spredixcan_genes_models
# spredixcan_genes_models = spredixcan_genes_models.join(spredixcan_genes_unique_n_snps)

# %%
# spredixcan_genes_models.shape

# %%
# spredixcan_genes_models.head()

# %% [markdown]
# ### Save

# %%
# this is important, other scripts depend on gene_name to be unique
assert spredixcan_genes_models["gene_name"].is_unique

# %%
assert not spredixcan_genes_models.isna().any(None)

# %%
spredixcan_genes_models.to_pickle(OUTPUT_DIR_BASE / "gene_tissues.pkl")

# %%
