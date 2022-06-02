"""
Gets user settings (from settings.py module) and create the final configuration values.
All the rest of the code reads configuration values from this module.
This file IS NOT intended to be modified by the user.
"""
import os
import tempfile
from pathlib import Path

import settings

# IMPORTANT: for variables or dictionary keys pointing to a directory,
# add the _DIR suffix to make sure the directory is created during setup.

#
# PhenoPLIER, general file structure
#
ROOT_DIR = os.environ.get("PHENOPLIER_ROOT_DIR")
if ROOT_DIR is None and hasattr(settings, "ROOT_DIR"):
    ROOT_DIR = settings.ROOT_DIR

if ROOT_DIR is None:
    ROOT_DIR = str(Path(tempfile.gettempdir(), "phenoplier").resolve())

# CODE_DIR points to the base directory where the code is
CODE_DIR = Path(__file__).parent.parent.resolve()

# DATA_DIR stores input data
DATA_DIR = Path(ROOT_DIR, "data").resolve()

# RESULTS_DIR stores newly generated data
RESULTS_DIR = Path(ROOT_DIR, "results").resolve()

# SOFTWARE_DIR stores third-party applications
SOFTWARE_DIR = Path(ROOT_DIR, "software").resolve()
CONDA_ENVS_DIR = Path(SOFTWARE_DIR, "conda_envs").resolve()


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
options = [
    m
    if (m := os.environ.get("PHENOPLIER_N_JOBS")) is not None and m.strip() != ""
    else None,
    getattr(settings, "N_JOBS", None),
    1,
]
GENERAL["N_JOBS"] = next(int(opt) for opt in options if opt is not None)

options = [
    m
    if (m := os.environ.get("PHENOPLIER_N_JOBS_HIGH")) is not None and m.strip() != ""
    else None,
    getattr(settings, "N_JOBS_HIGH", None),
    GENERAL["N_JOBS"],
]
GENERAL["N_JOBS_HIGH"] = next(int(opt) for opt in options if opt is not None)

GENERAL["TERM_ID_LABEL_FILE"] = Path(DATA_DIR, "term_id_labels.tsv.gz").resolve()

GENERAL["TERM_ID_XREFS_FILE"] = Path(DATA_DIR, "term_id_xrefs.tsv.gz").resolve()

# Experimental Factor Ontology (EFO)
GENERAL["EFO_ONTOLOGY_OBO_FILE"] = Path(DATA_DIR, "efo.obo").resolve()

# Liftover chain files
GENERAL["LIFTOVER"] = {}
GENERAL["LIFTOVER"]["BASE_DIR"] = Path(DATA_DIR, "liftover").resolve()
GENERAL["LIFTOVER"]["CHAINS_DIR"] = Path(
    GENERAL["LIFTOVER"]["BASE_DIR"], "chains"
).resolve()
GENERAL["LIFTOVER"]["HG19_TO_HG38"] = Path(
    GENERAL["LIFTOVER"]["CHAINS_DIR"], "hg19ToHg38.over.chain.gz"
).resolve()

# LD regions
GENERAL["EUR_LD_REGIONS_FILE"] = Path(DATA_DIR, "eur_ld.bed.gz").resolve()


#
# Software
#

# PLINK
PLINK = {}
PLINK["BASE_DIR"] = Path(SOFTWARE_DIR, "plink").resolve()
PLINK["EXECUTABLE_VERSION_1_9"] = Path(PLINK["BASE_DIR"], "plink")
PLINK["EXECUTABLE_VERSION_2"] = Path(PLINK["BASE_DIR"], "plink2")

# https://github.com/hakyimlab/summary-gwas-imputation
GWAS_IMPUTATION = {}
GWAS_IMPUTATION["BASE_DIR"] = Path(SOFTWARE_DIR, "summary-gwas-imputation").resolve()
GWAS_IMPUTATION["CONDA_ENV"] = Path(CONDA_ENVS_DIR, "summary_gwas_imputation").resolve()

# MetaXcan family of methods: https://github.com/hakyimlab/MetaXcan
METAXCAN = {}
METAXCAN["BASE_DIR"] = Path(SOFTWARE_DIR, "metaxcan").resolve()
METAXCAN["CONDA_ENV"] = Path(CONDA_ENVS_DIR, "metaxcan").resolve()


#
# Results
#
RESULTS = {}
RESULTS["BASE_DIR"] = RESULTS_DIR
RESULTS["PROJECTIONS_DIR"] = Path(RESULTS["BASE_DIR"], "projections").resolve()
RESULTS["DRUG_DISEASE_ANALYSES"] = Path(
    RESULTS["BASE_DIR"], "drug_disease_analyses"
).resolve()

RESULTS["DATA_TRANSFORMATIONS_DIR"] = Path(
    RESULTS["BASE_DIR"], "data_transformations"
).resolve()

RESULTS["CLUSTERING_DIR"] = Path(RESULTS["BASE_DIR"], "clustering").resolve()
RESULTS["CLUSTERING_TESTS_DIR"] = Path(RESULTS["CLUSTERING_DIR"], "tests").resolve()
RESULTS["CLUSTERING_RUNS_DIR"] = Path(RESULTS["CLUSTERING_DIR"], "runs").resolve()

RESULTS["CLUSTERING_INTERPRETATION"] = {}
RESULTS["CLUSTERING_INTERPRETATION"]["BASE_DIR"] = Path(
    RESULTS["CLUSTERING_DIR"],
    "interpretation",
).resolve()
RESULTS["CLUSTERING_INTERPRETATION"]["CLUSTERS_STATS"] = Path(
    Path(__file__).parent.parent,
    "nbs",
    "14_cluster_interpretation",
).resolve()

RESULTS["CRISPR_ANALYSES"] = {}
RESULTS["CRISPR_ANALYSES"]["BASE_DIR"] = Path(
    RESULTS["BASE_DIR"], "crispr_analyses"
).resolve()

RESULTS["GLS"] = Path(RESULTS["BASE_DIR"], "gls").resolve()
RESULTS["GLS_NULL_SIMS"] = Path(RESULTS["GLS"], "null_sims").resolve()


#
# Manuscript
#
MANUSCRIPT = {}
MANUSCRIPT["BASE_DIR"] = os.environ.get(
    "PHENOPLIER_MANUSCRIPT_DIR", settings.MANUSCRIPT_DIR
)
if MANUSCRIPT["BASE_DIR"] is not None:
    MANUSCRIPT["CONTENT_DIR"] = Path(MANUSCRIPT["BASE_DIR"], "content").resolve()
    MANUSCRIPT["FIGURES_DIR"] = Path(MANUSCRIPT["CONTENT_DIR"], "images").resolve()

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
MULTIPLIER["BANCHEREAU_MCPCOUNTER_NEUTROPHIL_FILE"] = Path(
    MULTIPLIER["BASE_DIR"], "Banchereau_MCPcounter_neutrophil_LV.tsv"
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
PHENOMEXCAN["SPREDIXCAN_MASHR_ZSCORES_FOLDER"] = Path(
    PHENOMEXCAN["GENE_ASSOC_DIR"], "spredixcan"
).resolve()
PHENOMEXCAN["SMULTIXCAN_MASHR_ZSCORES_FILE"] = Path(
    PHENOMEXCAN["GENE_ASSOC_DIR"], "smultixcan-mashr-zscores.pkl"
).resolve()
PHENOMEXCAN["SMULTIXCAN_EFO_PARTIAL_MASHR_ZSCORES_FILE"] = Path(
    PHENOMEXCAN["GENE_ASSOC_DIR"], "smultixcan-efo_partial-mashr-zscores.pkl"
).resolve()
PHENOMEXCAN["SMULTIXCAN_EFO_PARTIAL_MASHR_PVALUES_FILE"] = Path(
    PHENOMEXCAN["GENE_ASSOC_DIR"], "smultixcan-efo_partial-mashr-pvalues.pkl"
).resolve()
PHENOMEXCAN["SMULTIXCAN_MASHR_PVALUES_FILE"] = Path(
    PHENOMEXCAN["GENE_ASSOC_DIR"], "smultixcan-mashr-pvalues.pkl"
).resolve()
PHENOMEXCAN["FASTENLOC_TORUS_RCP_FILE"] = Path(
    PHENOMEXCAN["GENE_ASSOC_DIR"], "fastenloc-torus-rcp.pkl"
).resolve()
PHENOMEXCAN["FASTENLOC_EFO_PARTIAL_TORUS_RCP_FILE"] = Path(
    PHENOMEXCAN["GENE_ASSOC_DIR"], "fastenloc-efo_partial-torus-rcp.pkl"
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

# ld blocks
PHENOMEXCAN["LD_BLOCKS"] = {}
PHENOMEXCAN["LD_BLOCKS"]["BASE_DIR"] = Path(
    PHENOMEXCAN["BASE_DIR"], "ld_blocks"
).resolve()
PHENOMEXCAN["LD_BLOCKS"]["1000G_GENOTYPE_DIR"] = Path(
    PHENOMEXCAN["LD_BLOCKS"]["BASE_DIR"], "reference_panel_1000G"
).resolve()
PHENOMEXCAN["LD_BLOCKS"]["SNPS_COVARIANCE_FILE"] = Path(
    PHENOMEXCAN["LD_BLOCKS"]["BASE_DIR"],
    "mashr_snps_chr_blocks_cov.h5",
).resolve()
PHENOMEXCAN["LD_BLOCKS"]["GENE_IDS_CORR_AVG"] = Path(
    PHENOMEXCAN["LD_BLOCKS"]["BASE_DIR"],
    "multiplier_genes-pred_expression_corr_avg.pkl",
).resolve()
PHENOMEXCAN["LD_BLOCKS"]["GENE_NAMES_CORR_AVG"] = Path(
    PHENOMEXCAN["LD_BLOCKS"]["BASE_DIR"],
    "multiplier_genes-pred_expression_corr_avg-gene_names.pkl",
).resolve()

# predictions models
PHENOMEXCAN["PREDICTION_MODELS"] = {}
PHENOMEXCAN["PREDICTION_MODELS"]["BASE_DIR"] = Path(
    PHENOMEXCAN["BASE_DIR"], "prediction_models"
).resolve()
PHENOMEXCAN["PREDICTION_MODELS"]["MASHR"] = Path(
    PHENOMEXCAN["PREDICTION_MODELS"]["BASE_DIR"], "mashr"
).resolve()
PHENOMEXCAN["PREDICTION_MODELS"]["MASHR_PREFIX"] = "mashr_"
PHENOMEXCAN["PREDICTION_MODELS"]["MASHR_TISSUES"] = " ".join(
    tissue_file.name.split(PHENOMEXCAN["PREDICTION_MODELS"]["MASHR_PREFIX"])[1].split(
        ".db"
    )[0]
    for tissue_file in PHENOMEXCAN["PREDICTION_MODELS"]["MASHR"].glob("*.db")
)
PHENOMEXCAN["PREDICTION_MODELS"]["MASHR_SMULTIXCAN_COV_FILE"] = Path(
    PHENOMEXCAN["PREDICTION_MODELS"]["BASE_DIR"],
    "gtex_v8_expression_mashr_snp_covariance.txt.gz",
).resolve()


#
# Hetionet
#
HETIONET_BASE_DIR = Path(DATA_DIR, "hetionet").resolve()

PHARMACOTHERAPYDB = {}
PHARMACOTHERAPYDB["BASE_DIR"] = Path(
    HETIONET_BASE_DIR, "pharmacotherapydb-v1.0"
).resolve()
PHARMACOTHERAPYDB["INDICATIONS_FILE"] = Path(
    PHARMACOTHERAPYDB["BASE_DIR"], "indications.tsv"
).resolve()

LINCS = {}
LINCS["BASE_DIR"] = Path(HETIONET_BASE_DIR, "lincs-v2.0").resolve()
LINCS["CONSENSUS_SIGNATURES_FILE"] = Path(
    LINCS["BASE_DIR"], "consensi-drugbank.tsv.bz2"
).resolve()


#
# eMERGE
#
EMERGE = {}
EMERGE["BASE_DIR"] = Path(DATA_DIR, "emerge").resolve()
EMERGE["PHECODE_DESC_FILE"] = Path(
    EMERGE["BASE_DIR"], "phecode_definitions1.2.csv"
).resolve()
EMERGE["DESC_FILE_WITH_SAMPLE_SIZE"] = Path(
    EMERGE["BASE_DIR"],
    "eMERGE_III_PMBB_GSA_v2_2020_phecode_AFR_EUR_cc50_counts_w_dictionary.txt",
).resolve()
EMERGE["GENE_ASSOC_DIR"] = Path(EMERGE["BASE_DIR"], "gene_assoc").resolve()
EMERGE["SMULTIXCAN_MASHR_ZSCORES_FILE"] = Path(
    EMERGE["GENE_ASSOC_DIR"], "emerge-smultixcan-mashr-zscores.pkl"
).resolve()


#
# CRISPR screening
#
CRISPR = {}
CRISPR["BASE_DIR"] = Path(DATA_DIR, "crispr_screen").resolve()
CRISPR["LIPIDS_GENE_SETS_FILE"] = Path(CRISPR["BASE_DIR"], "lipid_DEG.csv").resolve()


#
# PharmacotherapyDB
#
HETIONET_BASE_DIR = Path(DATA_DIR, "hetionet").resolve()

PHARMACOTHERAPYDB = {}
PHARMACOTHERAPYDB["BASE_DIR"] = Path(
    HETIONET_BASE_DIR, "pharmacotherapydb-v1.0"
).resolve()
PHARMACOTHERAPYDB["INDICATIONS_FILE"] = Path(
    PHARMACOTHERAPYDB["BASE_DIR"], "indications.tsv"
).resolve()

LINCS = {}
LINCS["BASE_DIR"] = Path(HETIONET_BASE_DIR, "lincs-v2.0").resolve()
LINCS["CONSENSUS_SIGNATURES_FILE"] = Path(
    LINCS["BASE_DIR"], "consensi-drugbank.tsv.bz2"
).resolve()


#
# 1000 Genomes
#
A1000G = {}

A1000G["BASE_DIR"] = Path(DATA_DIR, "1000g").resolve()

# genotypes
A1000G["GENOTYPES_DIR"] = Path(A1000G["BASE_DIR"], "genotypes").resolve()


#
# External paths (outside ROOT_DIR)
#
EXTERNAL = {}

# GTEx v8
EXTERNAL["GTEX_V8_DIR"] = os.environ.get("PHENOPLIER_GTEX_V8_DIR")
if EXTERNAL["GTEX_V8_DIR"] is None and hasattr(settings, "GTEX_V8_DIR"):
    EXTERNAL["GTEX_V8_DIR"] = settings.GTEX_V8_DIR
if EXTERNAL["GTEX_V8_DIR"] is not None:
    EXTERNAL["GTEX_V8_DIR"] = Path(EXTERNAL["GTEX_V8_DIR"]).resolve()


if __name__ == "__main__":
    # if this script is run, then it exports the configuration as environment
    # variables (for bash/R, etc)
    from pathlib import PurePath

    def print_conf(conf_dict):
        for var_name, var_value in conf_dict.items():
            if var_value is None:
                continue

            if isinstance(var_value, (str, int, PurePath)):
                print(f'export PHENOPLIER_{var_name}="{str(var_value)}"')
            elif isinstance(var_value, dict):
                new_dict = {f"{var_name}_{k}": v for k, v in var_value.items()}
                print_conf(new_dict)
            else:
                raise ValueError(f"Configuration type not understood: {var_name}")

    local_variables = {
        k: v for k, v in locals().items() if not k.startswith("__") and k == k.upper()
    }

    print_conf(local_variables)
