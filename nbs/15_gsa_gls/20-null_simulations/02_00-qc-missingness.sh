#!/bin/bash

N_JOBS="${PHENOPLIER_GENERAL_N_JOBS}"
INPUT_DIR="${PHENOPLIER_A1000G_GENOTYPES_DIR}"
SUBSETS_DIR="${PHENOPLIER_A1000G_GENOTYPES_DIR}/subsets"
PLINK2="${PHENOPLIER_PLINK_EXECUTABLE_VERSION_2}"
PLINK19="${PHENOPLIER_PLINK_EXECUTABLE_VERSION_1_9}"

# calculate missingness
$PLINK19 --bfile ${SUBSETS_DIR}/all_phase3 \
    --threads ${N_JOBS} \
    --allow-extra-chr \
    --missing \
    --out ${SUBSETS_DIR}/all_phase3.missing
