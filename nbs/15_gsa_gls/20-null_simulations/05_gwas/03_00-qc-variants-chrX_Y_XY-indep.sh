#!/bin/bash

# Computes variants correlations on sex chromosomes.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
N_JOBS="${PHENOPLIER_GENERAL_N_JOBS}"
INPUT_DIR="${PHENOPLIER_A1000G_GENOTYPES_DIR}"
SUBSETS_DIR="${PHENOPLIER_A1000G_GENOTYPES_DIR}/subsets"
PLINK2="${PHENOPLIER_PLINK_EXECUTABLE_VERSION_2}"
PLINK19="${PHENOPLIER_PLINK_EXECUTABLE_VERSION_1_9}"

# independent snps (not correlated)
$PLINK19 --bfile ${SUBSETS_DIR}/all_phase3.1 \
    --threads ${N_JOBS} \
    --allow-extra-chr \
    --chr X,Y,XY \
    --indep-pairphase 20000 2000 0.5 \
    --out ${SUBSETS_DIR}/all_phase3.1.chrX.indepSNP
