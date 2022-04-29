#!/bin/bash

N_JOBS="${PHENOPLIER_GENERAL_N_JOBS}"
INPUT_DIR="${PHENOPLIER_A1000G_GENOTYPES_DIR}"
SUBSETS_DIR="${PHENOPLIER_A1000G_GENOTYPES_DIR}/subsets"
PLINK2="${PHENOPLIER_PLINK_EXECUTABLE_VERSION_2}"
PLINK19="${PHENOPLIER_PLINK_EXECUTABLE_VERSION_1_9}"

mkdir -p ${SUBSETS_DIR}

# decompress genotype data
$PLINK2 --zst-decompress ${INPUT_DIR}/all_phase3.pgen.zst > ${INPUT_DIR}/all_phase3.pgen

# convert to plink 1 binary format
$PLINK2 --pfile ${INPUT_DIR}/all_phase3 vzs \
    --threads ${N_JOBS} \
    --max-alleles 2 \
    --make-bed \
    --out ${SUBSETS_DIR}/all_phase3
