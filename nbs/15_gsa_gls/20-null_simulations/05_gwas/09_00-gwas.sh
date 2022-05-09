#!/bin/bash

# Computes GWAS. In this case, I use PLINK2 because it can compute GWAS in parallel and it's much faster.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
N_JOBS="${PHENOPLIER_GENERAL_N_JOBS}"
INPUT_DIR="${PHENOPLIER_A1000G_GENOTYPES_DIR}"
SUBSETS_DIR="${PHENOPLIER_A1000G_GENOTYPES_DIR}/subsets"
PLINK2="${PHENOPLIER_PLINK_EXECUTABLE_VERSION_2}"
PLINK19="${PHENOPLIER_PLINK_EXECUTABLE_VERSION_1_9}"
GWAS_DIR="${PHENOPLIER_RESULTS_GLS_NULL_SIMS}/gwas"

# gwas
mkdir -p ${GWAS_DIR}

$PLINK2 --bfile ${SUBSETS_DIR}/all_phase3.8 \
    --threads ${N_JOBS} \
    --glm hide-covar sex \
    --ci 0.95 \
    --covar ${SUBSETS_DIR}/all_phase3.7.pca_covar.eigenvec \
    --pheno ${SUBSETS_DIR}/all_phase3.8.random_pheno.txt \
    --out ${GWAS_DIR}/random
