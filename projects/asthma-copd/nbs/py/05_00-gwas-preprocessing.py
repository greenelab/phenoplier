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

# %% [markdown]
# # Description

# %% [markdown]
# TODO

# %% [markdown]
# # Modules

# %%
import pandas as pd

import conf

# %% [markdown]
# # Settings

# %% [markdown]
# # Paths

# %%
INPUT_GWAS_DIR = conf.GWAS_DIR / "asthma-copd"
display(INPUT_GWAS_DIR)
assert INPUT_GWAS_DIR.exists()

# %% [markdown]
# # List GWAS dir

# %%
list(INPUT_GWAS_DIR.iterdir())

# %% [markdown]
# # Read ACO GWAS

# %% [markdown]
# These GWAS results were generated using PLINK 2.0

# %%
gwas_data = pd.read_csv(INPUT_GWAS_DIR / "GWAS_ACO_GLM_SNPs_info0.7.txt", sep="\t")

# %%
gwas_data.shape

# %%
gwas_data.head()

# %% [markdown]
#

# %% [markdown]
# See here for PLINK formats: https://www.cog-genomics.org/plink/2.0/formats
#
# Some notes on columns:
# * `A1` is the counted allele (effect allele in PrediXcan terms).
# * `OBS_CT` is the sample size
# * `L95` and `U95` is the confidence interval for `OR`.

# %% [markdown]
# ## Checks

# %% [markdown]
# ### Stats

# %%
_tmp_p_stats = gwas_data["P"].describe()
display(_tmp_p_stats)

assert _tmp_p_stats["min"] > 0.00
assert _tmp_p_stats["max"] <= 1.00

# %%
gwas_data["Z_STAT"].describe()

# %% [markdown]
# ### Sample size

# %%
gwas_data["OBS_CT"].unique()

# %% [markdown]
# Sample size column has a single value.

# %% [markdown]
# ### Number of chromosomes

# %%
assert gwas_data["#CHROM"].unique().shape[0] == 22

# %% [markdown]
# ### Repeated chr/pos rows

# %%
duplicated_rows = gwas_data[gwas_data.duplicated(subset=["#CHROM", "POS"], keep=False)]
display(duplicated_rows)

# %%
duplicated_rows["P"].describe()

# %%
duplicated_rows.sort_values(by="P").head(10)

# %% [markdown]
# In cases like `rs3828794` (chr 6), REF=G, ALT=C, A1=G, should not be taken as REF/A1, since it would be G/G.
#
# Here we need to change those rows and make sure we have the correct alleles.

# %% [markdown]
# ### Check multiallelic variants

# %%
_tmp = gwas_data[gwas_data["REF"].str.contains(",")]
display(_tmp)

assert _tmp.shape[0] == 0

# %%
_tmp = gwas_data[gwas_data["ALT"].str.contains(",")]
display(_tmp)

assert _tmp.shape[0] == 0

# %%
_tmp = gwas_data[gwas_data["A1"].str.contains(",")]
display(_tmp)

assert _tmp.shape[0] == 0

# %% [markdown]
# ### Check A1 (effect allele)

# %%
gwas_data[gwas_data["ALT"] != gwas_data["A1"]]

# %% [markdown]
# Looks like A1 is not always equal to ALT, so I need to adjust that

# %% [markdown]
# # Harmonization, etc

# %% [markdown]
# In this page (https://predictdb.org/post/2021/07/21/gtex-v8-models-on-eqtl-and-sqtl/) they recommend two links:
# * https://github.com/hakyimlab/MetaXcan/wiki/Best-practices-for-integrating-GWAS-and-GTEX-v8-transcriptome-prediction-models
# * Tutorial: https://github.com/hakyimlab/MetaXcan/wiki/Tutorial:-GTEx-v8-MASH-models-integration-with-a-Coronary-Artery-Disease-GWAS

# %% [markdown]
# # Save GWAS in common format

# %%

# %%

# %% [markdown]
# BELOW SHOULD GO TO ANOTHER NOTEBOOK

# %% [markdown]
# # Run S-PrediXcan

# %% [markdown]
# Take a look here: https://github.com/hakyimlab/phenomexcan/blob/master/scripts/000_spredixcan/jobs/02_phenotypes_processing/main

# %% [markdown]
# ```
#         python ${METAXCAN_DIR}/software/MetaXcan.py \
#             --model_db_path ${METAXCAN_GTEX_V8_MODELS_DIR}/${METAXCAN_GTEX_V8_FILES_PREFIX}${tissue}.db \
#             --covariance ${METAXCAN_GTEX_V8_COVARS_DIR}/${METAXCAN_GTEX_V8_FILES_PREFIX}${tissue}.txt.gz \
#             --gwas_folder ${WORK_DIR} \
#             --gwas_file_pattern "final.*tsv" \
#             --separator $'\t' \
#             --non_effect_allele_column "ref" \
#             --effect_allele_column "alt" \
#             --snp_column  "${METAXCAN_RUN_SNP_COLUMN}" \
#             --beta_column "beta" \
#             --se_column "se" \
#             ${METAXCAN_RUN_EXTRA_OPTIONS} --output_file ${RESULT_PHENO_DIR}/${CSV_FILENAME} >> ${RESULT_PHENO_DIR}/${LOG_FILENAME} 2>&1
#
# ```

# %%
