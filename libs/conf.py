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
    "20_cluster_interpretation",
).resolve()

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
PHENOMEXCAN["SPREDIXCAN_MASHR_ZSCORES_FOLDER"] = Path(
    PHENOMEXCAN["GENE_ASSOC_DIR"], "spredixcan"
).resolve()
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
