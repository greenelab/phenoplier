#!/bin/bash

# Removes sex chromosomes from genotype data.

N_JOBS="${PHENOPLIER_GENERAL_N_JOBS}"
INPUT_DIR="${PHENOPLIER_A1000G_GENOTYPES_DIR}"
SUBSETS_DIR="${PHENOPLIER_A1000G_GENOTYPES_DIR}/subsets"
PLINK2="${PHENOPLIER_PLINK_EXECUTABLE_VERSION_2}"
PLINK19="${PHENOPLIER_PLINK_EXECUTABLE_VERSION_1_9}"

# keep snps in chr 1 to 22 only
$PLINK19 --bfile ${SUBSETS_DIR}/all_phase3.1.1 \
    --threads ${N_JOBS} \
    --allow-extra-chr \
    --chr 1-22 \
    --make-bed \
    --out ${SUBSETS_DIR}/all_phase3.2
