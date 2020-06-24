"""
General settings for the projects; includes mainly paths to data.
"""
import os
from multiprocessing import cpu_count
from os.path import join

# General settings
N_JOBS = int(cpu_count() / 2)
N_JOBS_HIGH = cpu_count()  # for low-computational tasks (IO, etc)

#
BASE_SETTINGS = {}
base_dir_default = "/media/miltondp/Elements/projects/k99/base/"
BASE_SETTINGS["BASE_DIR"] = os.environ.get("PHENOPLIER_BASE_DIR", base_dir_default)
BASE_SETTINGS["DATA_DIR"] = join(BASE_SETTINGS["BASE_DIR"], "data")
BASE_SETTINGS["RESULTS_DIR"] = join(BASE_SETTINGS["BASE_DIR"], "results")

manuscript_dir_default = "/home/miltondp/projects/labs/greenelab/phenoplier_manuscript"
BASE_SETTINGS["MANUSCRIPT_DIR"] = os.environ.get(
    "PHENOPLIER_MANUSCRIPT_DIR", manuscript_dir_default
)
BASE_SETTINGS["MANUSCRIPT_FIGURES_DIR"] = join(
    BASE_SETTINGS["MANUSCRIPT_DIR"], "content", "images"
)

BASE_SETTINGS["UK_BIOBANK_DIR"] = join(BASE_SETTINGS["DATA_DIR"], "uk_biobank")
BASE_SETTINGS["UK_BIOBANK/CODINGS_DIR"] = join(
    BASE_SETTINGS["UK_BIOBANK_DIR"], "codings"
)

# MultiPLIER
MULTIPLIER_SETTINGS = {}
multiplier_base_dir_default = "/media/miltondp/Elements/projects/multiplier"
MULTIPLIER_SETTINGS["BASE_DIR"] = os.environ.get(
    "MULTIPLIER_BASE_DIR", multiplier_base_dir_default
)
MULTIPLIER_SETTINGS["RECOUNT2_DATA_DIR"] = join(
    MULTIPLIER_SETTINGS["BASE_DIR"], "recount2_PLIER_data"
)
MULTIPLIER_SETTINGS["RECOUNT2_FULL_MODEL_FILE"] = join(
    MULTIPLIER_SETTINGS["RECOUNT2_DATA_DIR"], "recount_PLIER_model.RDS"
)
MULTIPLIER_SETTINGS["RECOUNT2_PREP_GENE_EXP_FILE"] = join(
    MULTIPLIER_SETTINGS["RECOUNT2_DATA_DIR"], "recount_data_prep_PLIER.RDS"
)
MULTIPLIER_SETTINGS["RECOUNT2_RPKM_EXP_FILE"] = join(
    MULTIPLIER_SETTINGS["RECOUNT2_DATA_DIR"], "recount_rpkm.RDS"
)
MULTIPLIER_SETTINGS["RESULTS_DIR"] = join(MULTIPLIER_SETTINGS["BASE_DIR"], "results")

# PhenomeXcan
PHENOMEXCAN_SETTINGS = {}
phenomexcan_base_dir_default = "/media/miltondp/Elements/projects/phenomexcan/base"
PHENOMEXCAN_SETTINGS["BASE_DIR"] = os.environ.get(
    "PHENOMEXCAN_BASE_DIR", phenomexcan_base_dir_default
)
PHENOMEXCAN_SETTINGS["DATA_DIR"] = join(PHENOMEXCAN_SETTINGS["BASE_DIR"], "data")
PHENOMEXCAN_SETTINGS["GENE_ASSOC_DIR"] = join(
    PHENOMEXCAN_SETTINGS["BASE_DIR"], "gene_assoc"
)

# Genes info
PHENOMEXCAN_SETTINGS["BIOMART_GENES_INFO_FILE"] = join(
    PHENOMEXCAN_SETTINGS["DATA_DIR"], "biomart_genes_hg38.csv.gz"
)
PHENOMEXCAN_SETTINGS["GENES_METADATA_DIR"] = join(
    PHENOMEXCAN_SETTINGS["DATA_DIR"], "genes_metadata"
)

# GWAS info
PHENOMEXCAN_SETTINGS["RAPID_GWAS_PHENO_INFO_FILE"] = join(
    PHENOMEXCAN_SETTINGS["DATA_DIR"], "phenotypes.both_sexes.tsv.gz"
)
PHENOMEXCAN_SETTINGS["RAPID_GWAS_DATA_DICT_FILE"] = join(
    PHENOMEXCAN_SETTINGS["DATA_DIR"], "UKB_Data_Dictionary_Showcase.tsv"
)
PHENOMEXCAN_SETTINGS["GTEX_GWAS_PHENO_INFO_FILE"] = join(
    PHENOMEXCAN_SETTINGS["DATA_DIR"], "gtex_gwas_phenotypes_metadata.tsv"
)

# LVs analysis of MultiPLIER and PhenomeXcan integration
LVS_MULTIPLIER_PHENOMEXCAN_SETTINGS = {}
LVS_MULTIPLIER_PHENOMEXCAN_SETTINGS["BASE_DIR"] = join(
    BASE_SETTINGS["RESULTS_DIR"], "lvs_x_phenomexcan"
)
LVS_MULTIPLIER_PHENOMEXCAN_SETTINGS["LVS_PROJECTIONS"] = join(
    LVS_MULTIPLIER_PHENOMEXCAN_SETTINGS["BASE_DIR"], "projections"
)
LVS_MULTIPLIER_PHENOMEXCAN_SETTINGS["CLUSTERING_DIR"] = join(
    LVS_MULTIPLIER_PHENOMEXCAN_SETTINGS["BASE_DIR"], "clustering"
)
