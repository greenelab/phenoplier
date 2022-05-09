#!/bin/bash

# It computes the Hardy-Weinberg equilibrium (HWE) exact test statistics on variants. This will be used to remove variants
# later.

N_JOBS="${PHENOPLIER_GENERAL_N_JOBS}"
INPUT_DIR="${PHENOPLIER_A1000G_GENOTYPES_DIR}"
SUBSETS_DIR="${PHENOPLIER_A1000G_GENOTYPES_DIR}/subsets"
PLINK2="${PHENOPLIER_PLINK_EXECUTABLE_VERSION_2}"
PLINK19="${PHENOPLIER_PLINK_EXECUTABLE_VERSION_1_9}"

# hardy
$PLINK19 --bfile ${SUBSETS_DIR}/all_phase3.3 \
    --threads ${N_JOBS} \
    --hardy \
    --out ${SUBSETS_DIR}/all_phase3.3
