#!/bin/bash

# Global parameters
SOFTWARE_DIR="/home/miltondp/projects/labs/greenelab/phenoplier/nbs/15_gsa_gls/20-null_simulations/10_gwas_harmonization/_software/summary-gwas-imputation"
DATA_DIR="/home/miltondp/projects/labs/greenelab/phenoplier/nbs/15_gsa_gls/20-null_simulations/10_gwas_harmonization/_data"
GWAS_DIR="/home/miltondp/projects/labs/greenelab/phenoplier/base/data/1000g/genotypes/gwas"
A1000G_REFERENCE_DATA="/home/miltondp/projects/labs/greenelab/phenoplier/base/data/phenomexcan/ld_blocks/reference_panel_1000G"

INPUT_GWAS_FILE="random_pheno0.glm.linear"

OUTPUT_HARMONIZED_DIR="outputs/harmonized"
INPUT_HARMONIZED_FILE="${INPUT_GWAS_FILE}.harmo.txt"
INPUT_HARMONIZED_FILE_PREFIX=${INPUT_HARMONIZED_FILE%.txt}

OUTPUT_IMPUTED_DIR="outputs/imputed"

OUTPUT_DIR="outputs/final"
mkdir -p ${OUTPUT_DIR}


python3 ${SOFTWARE_DIR}/src/gwas_summary_imputation_postprocess.py \
    -gwas_file ${OUTPUT_HARMONIZED_DIR}/${INPUT_HARMONIZED_FILE} \
    -folder ${OUTPUT_IMPUTED_DIR} \
    -pattern ${INPUT_HARMONIZED_FILE_PREFIX}.* \
    -parsimony 7 \
    -output ${OUTPUT_DIR}/${INPUT_GWAS_FILE}.imputed.txt.gz

