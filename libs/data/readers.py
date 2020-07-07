"""
Specifies functions to read different files used in the project.
"""
import pandas as pd

import conf


#
# Phenotypes metadata
#
def read_phenomexcan_rapid_gwas_pheno_info_file():
    return pd.read_csv(
        conf.PHENOMEXCAN["RAPID_GWAS_PHENO_INFO_FILE"], sep="\t", index_col="phenotype",
    )


def read_phenomexcan_rapid_gwas_data_dict():
    return pd.read_csv(
        conf.PHENOMEXCAN["RAPID_GWAS_DATA_DICT_FILE"], sep="\t", index_col="FieldID",
    )


def read_phenomexcan_gtex_gwas_pheno_info():
    return pd.read_csv(
        conf.PHENOMEXCAN["GTEX_GWAS_PHENO_INFO_FILE"], sep="\t", index_col="Tag"
    )


#
# UK Biobank codings files
#
def read_uk_biobank_codings(coding_number):
    """Returns functions to read coding files for UK Biobank fields.

    Differently than the other read_* functions, this one returns functions instead
    of data.
    """
    return lambda: pd.read_csv(
        conf.UK_BIOBANK[f"CODING_{coding_number}_FILE"], sep="\t"
    )


#
# Genes
#
def read_genes_biomart_data():
    return pd.read_csv(
        conf.GENERAL["BIOMART_GENES_INFO_FILE"], index_col="ensembl_gene_id"
    )


def read_gene_map_id_to_name():
    return pd.read_pickle(conf.PHENOMEXCAN["GENE_MAP_ID_TO_NAME"])


def read_gene_map_name_to_id():
    return pd.read_pickle(conf.PHENOMEXCAN["GENE_MAP_NAME_TO_ID"])


#
# This dictionary specifies as the value the function that knows how to read the
# file given in the key.
#
DATA_READERS = {
    conf.PHENOMEXCAN[
        "RAPID_GWAS_PHENO_INFO_FILE"
    ]: read_phenomexcan_rapid_gwas_pheno_info_file,
    conf.PHENOMEXCAN[
        "RAPID_GWAS_DATA_DICT_FILE"
    ]: read_phenomexcan_rapid_gwas_data_dict,
    conf.PHENOMEXCAN[
        "GTEX_GWAS_PHENO_INFO_FILE"
    ]: read_phenomexcan_gtex_gwas_pheno_info,
    conf.UK_BIOBANK["CODING_3_FILE"]: read_uk_biobank_codings(3),
    conf.UK_BIOBANK["CODING_6_FILE"]: read_uk_biobank_codings(6),
    conf.GENERAL["BIOMART_GENES_INFO_FILE"]: read_genes_biomart_data,
    conf.PHENOMEXCAN["GENE_MAP_ID_TO_NAME"]: read_gene_map_id_to_name,
    conf.PHENOMEXCAN["GENE_MAP_NAME_TO_ID"]: read_gene_map_name_to_id,
}
