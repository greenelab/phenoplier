#!/bin/bash

N_JOBS="${PHENOPLIER_GENERAL_N_JOBS}"
INPUT_DIR="${PHENOPLIER_A1000G_GENOTYPES_DIR}"
SUBSETS_DIR="${PHENOPLIER_A1000G_GENOTYPES_DIR}/subsets"
PLINK2="${PHENOPLIER_PLINK_EXECUTABLE_VERSION_2}"
PLINK19="${PHENOPLIER_PLINK_EXECUTABLE_VERSION_1_9}"

# remove related individuals (file deg2_... was downloaded from plink webpage)
$PLINK2 --bfile ${SUBSETS_DIR}/all_phase3.5 \
    --threads ${N_JOBS} \
    --allow-extra-chr \
    --remove ${INPUT_DIR}/deg2_phase3.king.cutoff.out.id \
    --make-bed \
    --out ${SUBSETS_DIR}/all_phase3.6
