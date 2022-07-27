# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
#     formats: ipynb,py//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
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
# **TODO:** update
#
# This notebook computes predicted expression correlations between all genes in the MultiPLIER models.
#
# It also has a parameter set for papermill to run on a single chromosome to run in parallel (see under `Settings` below).
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
from tqdm import tqdm

import conf
from entity import Gene

# %% [markdown] tags=[]
# # Settings

# %% tags=["parameters"]
# a cohort name (it could be something like UK_BIOBANK, etc)
COHORT_NAME = None

# a string with a path pointing to an imputed GWAS
GWAS_FILE = None

# a string with a path pointing where S-PrediXcan results (tissue-specific are located
SPREDIXCAN_FOLDER = None

# an f-string with one placeholder {tissue}
SPREDIXCAN_FILE_PATTERN = None

# a string with a path pointing to an S-MultiXcan result
SMULTIXCAN_FILE = None

# predictions models such as MASHR or ELASTIC_NET
EQTL_MODEL = None

# %%
assert COHORT_NAME is not None and len(COHORT_NAME) > 0, "A cohort name must be given"

COHORT_NAME = COHORT_NAME.lower()
display(f"Cohort name: {COHORT_NAME}")

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
assert SPREDIXCAN_FOLDER.exists(), "S-PrediXcan folder does not exist"

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
assert SMULTIXCAN_FILE.exists(), "S-MultiXcan result file does not exist"

display(f"S-MultiXcan file path: {str(SMULTIXCAN_FILE)}")

# %%
assert (
    EQTL_MODEL is not None and len(EQTL_MODEL) > 0
), "A prediction/eQTL model must be given"

display(f"eQTL model: {EQTL_MODEL}")

# %%
OUTPUT_DIR_BASE = conf.RESULTS["GLS"] / "gene_corrs" / "cohorts" / COHORT_NAME
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
), "GWAS file must be final imputed one with column 'panel_variant_id'"
# FIXME: add other needed columns here

# %%
gwas_data = pd.read_csv(
    GWAS_FILE,
    sep="\t",
    usecols=["panel_variant_id", "pvalue", "zscore", "imputation_status"],
)

# %%
gwas_data.shape

# %%
gwas_data.head()

# %%
gwas_data["imputation_status"].unique()

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
assert gwas_data["panel_variant_id"].is_unique

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
        f, usecols=["gene", "zscore", "pvalue", "n_snps_used", "n_snps_in_model"]
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

# %% [markdown] tags=[]
# ### Count number of tissues available per gene

# %%
spredixcan_genes_n_models = spredixcan_dfs.groupby("gene_id")["tissue"].nunique()

# %%
spredixcan_genes_n_models

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
# testing
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
# ### Count number of _unique_ SNPs predictors used and available across tissue models

# %%
spredixcan_gene_obj = {
    gene_id: Gene(ensembl_id=gene_id) for gene_id in spredixcan_genes_models.index
}

# %%
len(spredixcan_gene_obj)


# %%
def _count_unique_snps(gene_id):
    """
    For a gene_id, it counts unique SNPs in all models and their intersection with GWAS SNPs (therefore, used by S-PrediXcan).
    """
    gene_obj = spredixcan_gene_obj[gene_id]
    gene_tissues = spredixcan_genes_models.loc[gene_id, "tissue"]

    gene_unique_snps = set()
    for t in gene_tissues:
        t_snps = set(
            gene_obj.get_prediction_weights(tissue=t, model_type=EQTL_MODEL).index
        )
        gene_unique_snps.update(t_snps)

    gene_unique_snps_in_gwas = gwas_variants_ids_set.intersection(gene_unique_snps)

    return pd.Series(
        {
            "unique_n_snps_in_model": len(gene_unique_snps),
            "unique_n_snps_used": len(gene_unique_snps_in_gwas),
        }
    )


# %%
# testing
spredixcan_genes_models[spredixcan_genes_models["n_snps_used_sum"] == 2].head()

# %%
# case with two snps, not repeated across tissues
_gene_id = "ENSG00000000419"
display(
    spredixcan_gene_obj[_gene_id].get_prediction_weights(
        tissue="Brain_Hypothalamus", model_type=EQTL_MODEL
    )
)
display(
    spredixcan_gene_obj[_gene_id].get_prediction_weights(
        tissue="Brain_Substantia_nigra", model_type=EQTL_MODEL
    )
)

# %%
_tmp = _count_unique_snps(_gene_id)
assert _tmp.shape[0] == 2
assert _tmp["unique_n_snps_in_model"] == 2
assert _tmp["unique_n_snps_used"] == 2

# %%
# case with two snps, repeated across tissues
_gene_id = "ENSG00000007202"
display(
    spredixcan_gene_obj[_gene_id].get_prediction_weights(
        tissue="Adipose_Subcutaneous", model_type=EQTL_MODEL
    )
)
display(
    spredixcan_gene_obj[_gene_id].get_prediction_weights(
        tissue="Esophagus_Mucosa", model_type=EQTL_MODEL
    )
)

# %%
"chr17_28638994_C_T_b38" in gwas_variants_ids_set

# %%
_tmp = _count_unique_snps(_gene_id)
assert _tmp.shape[0] == 2
assert _tmp["unique_n_snps_in_model"] == 1
assert _tmp["unique_n_snps_used"] == 1

# %%
# get unique snps for all genes
spredixcan_genes_unique_n_snps = spredixcan_genes_models.groupby("gene_id").apply(
    lambda x: _count_unique_snps(x.name)
)

# %%
spredixcan_genes_unique_n_snps.head()

# %%
assert (
    spredixcan_genes_unique_n_snps["unique_n_snps_in_model"]
    >= spredixcan_genes_unique_n_snps["unique_n_snps_used"]
).all()

# %%
# add unique snps to spredixcan_genes_models
spredixcan_genes_models = spredixcan_genes_models.join(spredixcan_genes_unique_n_snps)

# %%
spredixcan_genes_models.shape

# %%
spredixcan_genes_models.head()

# %% [markdown]
# ### Save

# %%
spredixcan_genes_models.to_pickle(OUTPUT_DIR_BASE / "gene_tissues.pkl")

# %% [markdown] tags=[]
# ## S-MultiXcan results

# %%
# TODO: something that could be interesting to do is to compare `n_indep` with the number of independent components I get
smultixcan_results = pd.read_csv(
    SMULTIXCAN_FILE, sep="\t", usecols=["gene", "gene_name", "pvalue", "n"]
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

# %%
# testing
_tmp_smultixcan_results_n_models = (
    smultixcan_results.set_index("gene_id")["n"].astype(int).rename("tissue")
)

assert spredixcan_genes_n_models.shape[0] == _tmp_smultixcan_results_n_models.shape[0]
assert spredixcan_genes_n_models.equals(
    _tmp_smultixcan_results_n_models.loc[spredixcan_genes_n_models.index]
)

# %% [markdown]
# ### Remove duplicated gene names

# %%
smultixcan_results["gene_name"].is_unique

# %%
# list duplicated gene names
_smultixcan_duplicated_gene_names = smultixcan_results[
    smultixcan_results["gene_name"].duplicated(keep=False)
]
display(_smultixcan_duplicated_gene_names)

# %%
# TODO: my strategy below to handle duplicated gene names is to keep the first one
#  it might be better to have another strategy, maybe keeping the most significant

# %%
smultixcan_results = smultixcan_results.drop_duplicates(
    subset=["gene_name"], keep="first"
)
display(smultixcan_results.shape)

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

# %%
assert smultixcan_results[smultixcan_results["gene_name"].isin(common_genes)].shape[
    0
] == len(common_genes)

# %% [markdown]
# ### Save

# %%
with open(OUTPUT_DIR_BASE / "common_genes.pkl", "wb") as handle:
    pickle.dump(common_genes, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %% [markdown] tags=[]
# # Get gene objects

# %% tags=[]
multiplier_gene_obj = {
    gene_name: Gene(name=gene_name)
    for gene_name in common_genes
    if gene_name in Gene.GENE_NAME_TO_ID_MAP
}

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

# %% tags=[]
genes_info.shape

# %% tags=[]
genes_info.head()

# %% [markdown]
# ## Save

# %%
genes_info.to_pickle(OUTPUT_DIR_BASE / "genes_info.pkl")

# %%
