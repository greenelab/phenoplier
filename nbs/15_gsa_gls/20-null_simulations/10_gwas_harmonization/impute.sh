#!/bin/bash

# Global parameters
SOFTWARE_DIR="/home/miltondp/projects/labs/greenelab/phenoplier/nbs/15_gsa_gls/20-null_simulations/10_gwas_harmonization/_software/summary-gwas-imputation"
DATA_DIR="/home/miltondp/projects/labs/greenelab/phenoplier/nbs/15_gsa_gls/20-null_simulations/10_gwas_harmonization/_data"
GWAS_DIR="/home/miltondp/projects/labs/greenelab/phenoplier/base/data/1000g/genotypes/gwas"
A1000G_REFERENCE_DATA="/home/miltondp/projects/labs/greenelab/phenoplier/base/data/phenomexcan/ld_blocks/reference_panel_1000G"

INPUT_DIR="outputs/harmonized"

OUTPUT_DIR="outputs/imputed"
mkdir -p ${OUTPUT_DIR}

INPUT_FILE="random_pheno0.glm.linear.harmo.txt"
CHROMOSOME=$1
N_BATCHES=$2
BATCH_ID=$3

INPUT_FILE_PREFIX=${INPUT_FILE%.txt}

python3 ${SOFTWARE_DIR}/src/gwas_summary_imputation.py \
    -by_region_file ${DATA_DIR}/eur_ld.bed.gz \
    -gwas_file ${INPUT_DIR}/${INPUT_FILE} \
    -parquet_genotype ${A1000G_REFERENCE_DATA}/chr${CHROMOSOME}.variants.parquet \
    -parquet_genotype_metadata ${A1000G_REFERENCE_DATA}/variant_metadata.parquet \
    -window 100000 \
    -parsimony 7 \
    -chromosome ${CHROMOSOME} \
    -regularization 0.1 \
    -frequency_filter 0.01 \
    -sub_batches ${N_BATCHES} \
    -sub_batch ${BATCH_ID} \
    --standardise_dosages \
    -output ${OUTPUT_DIR}/${INPUT_FILE_PREFIX}.imp.chr${CHROMOSOME}.batch${BATCH_ID}_${N_BATCHES}.txt

