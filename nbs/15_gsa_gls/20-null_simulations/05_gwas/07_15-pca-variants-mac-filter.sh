#!/bin/bash

# Remove rare variants

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
N_JOBS="${PHENOPLIER_GENERAL_N_JOBS}"
INPUT_DIR="${PHENOPLIER_A1000G_GENOTYPES_DIR}"
SUBSETS_DIR="${PHENOPLIER_A1000G_GENOTYPES_DIR}/subsets"
PLINK2="${PHENOPLIER_PLINK_EXECUTABLE_VERSION_2}"
PLINK19="${PHENOPLIER_PLINK_EXECUTABLE_VERSION_1_9}"

# mac (minimum allele count)
$PLINK19 --bfile ${SUBSETS_DIR}/all_phase3.7 \
    --threads ${N_JOBS} \
    --mac 20 \
    --make-bed \
    --out ${SUBSETS_DIR}/all_phase3.8
