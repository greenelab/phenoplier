"""
Provides access to frequently used data directly.
"""
from os.path import join

import pandas as pd

import settings as conf
from utils import load_pickle


# Phenotypes metadata
RAPID_GWAS_PHENO_INFO = pd.read_csv(conf.PHENOMEXCAN_SETTINGS['RAPID_GWAS_PHENO_INFO_FILE'], sep='\t', index_col='phenotype')
RAPID_GWAS_DATA_DICT = pd.read_csv(conf.PHENOMEXCAN_SETTINGS['RAPID_GWAS_DATA_DICT_FILE'], sep='\t', index_col='FieldID')

GTEX_GWAS_PHENO_INFO = pd.read_csv(conf.PHENOMEXCAN_SETTINGS['GTEX_GWAS_PHENO_INFO_FILE'], sep='\t', index_col='Tag')

# UK Biobank codings files.
UK_BIOBANK_CODINGS = {}
UK_BIOBANK_CODINGS[3] = pd.read_csv(join(conf.BASE_SETTINGS['UK_BIOBANK/CODINGS_DIR'], 'coding3.tsv'), sep='\t')
UK_BIOBANK_CODINGS[6] = pd.read_csv(join(conf.BASE_SETTINGS['UK_BIOBANK/CODINGS_DIR'], 'coding6.tsv'), sep='\t')

# Genes metadata
BIOMART_GENES = pd.read_csv(conf.PHENOMEXCAN_SETTINGS['BIOMART_GENES_INFO_FILE'], index_col='ensembl_gene_id')
# GENES_MAPPINGS = load_pickle(join(conf.PHENOMEXCAN_SETTINGS['GENES_METADATA_DIR'], 'genes_mappings.pkl'))
GENE_ID_TO_NAME_MAP = load_pickle(join(conf.PHENOMEXCAN_SETTINGS['GENES_METADATA_DIR'], 'genes_mapping_simplified-0.pkl'))
GENE_NAME_TO_ID_MAP = load_pickle(join(conf.PHENOMEXCAN_SETTINGS['GENES_METADATA_DIR'], 'genes_mapping_simplified-1.pkl'))
