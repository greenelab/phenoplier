#!/bin/bash
# +

# -
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
N_JOBS="${PHENOPLIER_GENERAL_N_JOBS}"
INPUT_DIR="${PHENOPLIER_A1000G_GENOTYPES_DIR}"
SUBSETS_DIR="${PHENOPLIER_A1000G_GENOTYPES_DIR}/subsets"
PLINK2="${PHENOPLIER_PLINK_EXECUTABLE_VERSION_2}"
PLINK19="${PHENOPLIER_PLINK_EXECUTABLE_VERSION_1_9}"

# keep europeans only
$PLINK19 --bfile ${SUBSETS_DIR}/all_phase3.6 \
    --threads ${N_JOBS} \
    --keep ${SUBSETS_DIR}/all_phase3.6.pca.eur \
    --make-bed \
    --out ${SUBSETS_DIR}/all_phase3.7
