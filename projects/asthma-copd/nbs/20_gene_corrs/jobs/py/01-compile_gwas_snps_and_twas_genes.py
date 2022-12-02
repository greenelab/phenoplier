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
# This notebook compiles information about the GWAS and TWAS for a particular cohort. For example, the set of GWAS variants, variance of predicted expression of genes, etc.
#
# It has specicfic parameters for papermill (see under `Settings` below).
#
# This notebook should not be directly run. It is used by other notebooks.

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

OUTPUT_DIR_BASE = (Path(OUTPUT_DIR_BASE) / "gene_corrs" / COHORT_NAME).resolve()

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
assert not spredixcan_genes_models.isna().any(axis=None)

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
# ### Save

# %%
# this is important, other scripts depend on gene_name to be unique
assert spredixcan_genes_models["gene_name"].is_unique

# %%
assert not spredixcan_genes_models.isna().any(axis=None)

# %%
spredixcan_genes_models.to_pickle(OUTPUT_DIR_BASE / "gene_tissues.pkl")

# %%
