#!/bin/bash

# It removes samples with high heterozygosity. In this case, code is commented out since no such samples were found
# in this data.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
N_JOBS="${PHENOPLIER_GENERAL_N_JOBS}"
INPUT_DIR="${PHENOPLIER_A1000G_GENOTYPES_DIR}"
SUBSETS_DIR="${PHENOPLIER_A1000G_GENOTYPES_DIR}/subsets"
PLINK2="${PHENOPLIER_PLINK_EXECUTABLE_VERSION_2}"
PLINK19="${PHENOPLIER_PLINK_EXECUTABLE_VERSION_1_9}"

# remove samples with high heterozygosity
# in this data there are no samples to remove (see notebook)

# sed 's/"// g' ${SUBSETS_DIR}/all_phase3.4.fail-het-qc.txt | awk '{print$1, $2}'> ${SUBSETS_DIR}/all_phase3.4.het_fail_ind.txt

# $PLINK19 --bfile ${SUBSETS_DIR}/all_phase3.4 \
#     --threads ${N_JOBS} \
#     --remove ${SUBSETS_DIR}/all_phase3.4.het_fail_ind.txt \
#     --make-bed \
#     --out ${SUBSETS_DIR}/all_phase3.5
