#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
N_JOBS="${PHENOPLIER_GENERAL_N_JOBS}"
INPUT_DIR="${PHENOPLIER_A1000G_GENOTYPES_DIR}"
SUBSETS_DIR="${PHENOPLIER_A1000G_GENOTYPES_DIR}/subsets"
PLINK2="${PHENOPLIER_PLINK_EXECUTABLE_VERSION_2}"
PLINK19="${PHENOPLIER_PLINK_EXECUTABLE_VERSION_1_9}"

# independent snps (not correlated)
$PLINK19 --bfile ${SUBSETS_DIR}/all_phase3.4 \
    --exclude range ${SCRIPT_DIR}/inversion.txt \
    --indep-pairwise 50 5 0.2 \
    --out ${SUBSETS_DIR}/all_phase3.4.indepSNP

# heterozygosity
$PLINK19 --bfile ${SUBSETS_DIR}/all_phase3.4 \
    --extract ${SUBSETS_DIR}/all_phase3.4.indepSNP.prune.in \
    --het \
    --out ${SUBSETS_DIR}/all_phase3.4.indepSNP.R_check
