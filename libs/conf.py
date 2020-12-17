"""
Gets user settings (from settings.py module) and create the final configuration values.
All the rest of the code reads configuration values from this module.
This file IS NOT intended to be modified by the user.
"""
import os
import tempfile
from multiprocessing import cpu_count
from pathlib import Path

import settings

# IMPORTANT: for variables or dictionary keys pointing to a directory, add the _DIR
# suffix to make sure the directory is created during setup.

#
# PhenoPLIER, general file structure
#
ROOT_DIR = os.environ.get("PHENOPLIER_ROOT_DIR")
if ROOT_DIR is None and hasattr(settings, "ROOT_DIR"):
    ROOT_DIR = settings.ROOT_DIR

if ROOT_DIR is None:
    ROOT_DIR = str(Path(tempfile.gettempdir(), "phenoplier").resolve())

# DATA_DIR stores input data
DATA_DIR = Path(ROOT_DIR, "data").resolve()

# RESULTS_DIR stores newly generated data
RESULTS_DIR = Path(ROOT_DIR, "results").resolve()

#
# General
#
GENERAL = {}
GENERAL["BIOMART_GENES_INFO_FILE"] = Path(
    DATA_DIR, "biomart_genes_hg38.csv.gz"
).resolve()

GENERAL["LOG_CONFIG_FILE"] = Path(
    Path(__file__).resolve().parent, "log_config.yaml"
).resolve()

# CPU usage
options = [settings.N_JOBS, int(cpu_count() / 2)]
GENERAL["N_JOBS"] = next(opt for opt in options if opt is not None)

options = [settings.N_JOBS_HIGH, cpu_count()]
GENERAL["N_JOBS_HIGH"] = next(opt for opt in options if opt is not None)

GENERAL["TERM_ID_LABEL_FILE"] = Path(DATA_DIR, "term_id_labels.tsv.gz").resolve()

GENERAL["TERM_ID_XREFS_FILE"] = Path(DATA_DIR, "term_id_xrefs.tsv.gz").resolve()

GENERAL["EFO_ONTOLOGY_OBO_FILE"] = Path(DATA_DIR, "efo.obo").resolve()

#
# Results
#
RESULTS = {}
RESULTS["BASE_DIR"] = RESULTS_DIR
RESULTS["PROJECTIONS_DIR"] = Path(RESULTS["BASE_DIR"], "projections").resolve()

RESULTS["DATA_TRANSFORMATIONS_DIR"] = Path(
    RESULTS["BASE_DIR"], "data_transformations"
).resolve()

RESULTS["CLUSTERING_DIR"] = Path(RESULTS["BASE_DIR"], "clustering").resolve()
# RESULTS["CLUSTERING_TRAITS_DATA_DIR"] = Path(
#     RESULTS["CLUSTERING_DIR"],
#     "traits_data"
# ).resolve()

#
# Manuscript
#
MANUSCRIPT = {}
MANUSCRIPT["BASE_DIR"] = os.environ.get(
    "PHENOPLIER_MANUSCRIPT_DIR", settings.MANUSCRIPT_DIR
)
if MANUSCRIPT["BASE_DIR"] is not None:
    MANUSCRIPT["FIGURES_DIR"] = Path(
        MANUSCRIPT["BASE_DIR"], "content", "images"
    ).resolve()

#
# recount2
#
RECOUNT2 = {}
RECOUNT2["BASE_DIR"] = Path(DATA_DIR, "recount2").resolve()
RECOUNT2["PREPROCESSED_GENE_EXPRESSION_FILE"] = Path(
    RECOUNT2["BASE_DIR"], "recount_data_prep_PLIER.RDS"
).resolve()

#
# UK Biobank paths
#
UK_BIOBANK = {}
UK_BIOBANK["BASE_DIR"] = Path(DATA_DIR, "uk_biobank").resolve()
UK_BIOBANK["CODINGS_DIR"] = Path(UK_BIOBANK["BASE_DIR"], "codings").resolve()
UK_BIOBANK["CODING_3_FILE"] = Path(UK_BIOBANK["CODINGS_DIR"], "coding3.tsv").resolve()
UK_BIOBANK["CODING_6_FILE"] = Path(UK_BIOBANK["CODINGS_DIR"], "coding6.tsv").resolve()
UK_BIOBANK["UKBCODE_TO_EFO_MAP_FILE"] = Path(
    UK_BIOBANK["BASE_DIR"], "UK_Biobank_master_file.tsv"
).resolve()

#
# MultiPLIER
#
MULTIPLIER = {}
MULTIPLIER["BASE_DIR"] = Path(DATA_DIR, "multiplier").resolve()
MULTIPLIER["MODEL_SUMMARY_FILE"] = Path(
    MULTIPLIER["BASE_DIR"], "multiplier_model_summary.pkl"
).resolve()
MULTIPLIER["MODEL_Z_MATRIX_FILE"] = Path(
    MULTIPLIER["BASE_DIR"], "multiplier_model_z.pkl"
).resolve()
MULTIPLIER["MODEL_B_MATRIX_FILE"] = Path(
    MULTIPLIER["BASE_DIR"], "multiplier_model_b.pkl"
).resolve()
MULTIPLIER["MODEL_U_MATRIX_FILE"] = Path(
    MULTIPLIER["BASE_DIR"], "multiplier_model_u.pkl"
).resolve()
MULTIPLIER["MODEL_U_AUC_MATRIX_FILE"] = Path(
    MULTIPLIER["BASE_DIR"], "multiplier_model_u_auc.pkl"
).resolve()
MULTIPLIER["MODEL_METADATA_FILE"] = Path(
    MULTIPLIER["BASE_DIR"], "multiplier_model_metadata.pkl"
).resolve()
MULTIPLIER["RECOUNT2_MODEL_FILE"] = Path(
    MULTIPLIER["BASE_DIR"], "recount_PLIER_model.RDS"
).resolve()

#
# PhenomeXcan
#
PHENOMEXCAN = {}
PHENOMEXCAN["BASE_DIR"] = Path(DATA_DIR, "phenomexcan").resolve()

# genes metadata and mappings
PHENOMEXCAN["GENES_METADATA_DIR"] = Path(
    PHENOMEXCAN["BASE_DIR"], "genes_metadata"
).resolve()
PHENOMEXCAN["GENE_MAP_ID_TO_NAME"] = Path(
    PHENOMEXCAN["GENES_METADATA_DIR"],
    "genes_mapping_id_to_name.pkl",
).resolve()
PHENOMEXCAN["GENE_MAP_NAME_TO_ID"] = Path(
    PHENOMEXCAN["GENES_METADATA_DIR"],
    "genes_mapping_name_to_id.pkl",
).resolve()
PHENOMEXCAN["TRAITS_FULLCODE_TO_EFO_MAP_FILE"] = Path(
    PHENOMEXCAN["BASE_DIR"], "phenomexcan_traits_fullcode_to_efo.tsv"
).resolve()

# gene association results
PHENOMEXCAN["GENE_ASSOC_DIR"] = Path(PHENOMEXCAN["BASE_DIR"], "gene_assoc").resolve()
PHENOMEXCAN["SMULTIXCAN_MASHR_ZSCORES_FILE"] = Path(
    PHENOMEXCAN["GENE_ASSOC_DIR"], "smultixcan-mashr-zscores.pkl"
).resolve()
PHENOMEXCAN["SMULTIXCAN_EFO_PARTIAL_MASHR_ZSCORES_FILE"] = Path(
    PHENOMEXCAN["GENE_ASSOC_DIR"], "smultixcan-efo_partial-mashr-zscores.pkl"
).resolve()
PHENOMEXCAN["SMULTIXCAN_MASHR_PVALUES_FILE"] = Path(
    PHENOMEXCAN["GENE_ASSOC_DIR"], "smultixcan-mashr-pvalues.pkl"
).resolve()
PHENOMEXCAN["FASTENLOC_TORUS_RCP_FILE"] = Path(
    PHENOMEXCAN["GENE_ASSOC_DIR"], "fastenloc-torus-rcp.pkl"
).resolve()

# GWAS info
PHENOMEXCAN["RAPID_GWAS_PHENO_INFO_FILE"] = Path(
    PHENOMEXCAN["BASE_DIR"], "phenotypes.both_sexes.tsv.gz"
).resolve()
PHENOMEXCAN["RAPID_GWAS_DATA_DICT_FILE"] = Path(
    PHENOMEXCAN["BASE_DIR"], "UKB_Data_Dictionary_Showcase.tsv"
).resolve()
PHENOMEXCAN["GTEX_GWAS_PHENO_INFO_FILE"] = Path(
    PHENOMEXCAN["BASE_DIR"], "gtex_gwas_phenotypes_metadata.tsv"
).resolve()
