#!/bin/bash

# Removes variants with a specific proportion of missing values (--geno parameter).

N_JOBS="${PHENOPLIER_GENERAL_N_JOBS}"
INPUT_DIR="${PHENOPLIER_A1000G_GENOTYPES_DIR}"
SUBSETS_DIR="${PHENOPLIER_A1000G_GENOTYPES_DIR}/subsets"
PLINK2="${PHENOPLIER_PLINK_EXECUTABLE_VERSION_2}"
PLINK19="${PHENOPLIER_PLINK_EXECUTABLE_VERSION_1_9}"

# delete variants with missingness > 0.01
$PLINK19 --bfile ${SUBSETS_DIR}/all_phase3 \
    --threads ${N_JOBS} \
    --allow-extra-chr \
    --geno 0.01 \
    --make-bed \
    --out ${SUBSETS_DIR}/all_phase3.1
