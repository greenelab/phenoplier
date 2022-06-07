#!/bin/bash

# Remove original GWAS files

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
N_JOBS="${PHENOPLIER_GENERAL_N_JOBS}"
INPUT_DIR="${PHENOPLIER_A1000G_GENOTYPES_DIR}"
SUBSETS_DIR="${PHENOPLIER_A1000G_GENOTYPES_DIR}/subsets"
PLINK2="${PHENOPLIER_PLINK_EXECUTABLE_VERSION_2}"
PLINK19="${PHENOPLIER_PLINK_EXECUTABLE_VERSION_1_9}"
GWAS_DIR="${PHENOPLIER_RESULTS_GLS_NULL_SIMS}/gwas"

# remove original files
echo "IMPORTANT: this step is dangerous. Make sure the previous steps worked before removing the original GWAS files"
sleep 10

# rm ${GWAS_DIR}/*.glm.linear
