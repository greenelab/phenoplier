#!/bin/bash

# Remove unneeded columns from GWAS and compress files

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
N_JOBS="${PHENOPLIER_GENERAL_N_JOBS}"
INPUT_DIR="${PHENOPLIER_A1000G_GENOTYPES_DIR}"
SUBSETS_DIR="${PHENOPLIER_A1000G_GENOTYPES_DIR}/subsets"
PLINK2="${PHENOPLIER_PLINK_EXECUTABLE_VERSION_2}"
PLINK19="${PHENOPLIER_PLINK_EXECUTABLE_VERSION_1_9}"
GWAS_DIR="${PHENOPLIER_RESULTS_GLS_NULL_SIMS}/gwas"

# remove unneeded columns
echo "Removing columns and compressing"
removecols() {
   FILE=$1
   awk -F'\t' 'BEGIN {OFS = FS} {print $1,$2,$3,$4,$6,$9,$10,$12}' ${FILE} | gzip > ${FILE}.tsv.gz
}
export -f removecols
parallel -j${N_JOBS} removecols {} ::: \
  ${GWAS_DIR}/*.glm.linear
