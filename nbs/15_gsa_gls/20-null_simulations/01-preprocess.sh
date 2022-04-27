#!/bin/bash

INPUT_DIR="${PHENOPLIER_A1000G_GENOTYPES_DIR}"
SUBSETS_DIR="${PHENOPLIER_A1000G_GENOTYPES_DIR}/subsets"
PLINK2="${PHENOPLIER_PLINK2_EXECUTABLE}"

mkdir -p ${SUBSETS_DIR}

# decompress genotype data
$PLINK2 --zst-decompress ${INPUT_DIR}/all_phase3.pgen.zst > ${INPUT_DIR}/all_phase3.pgen
$PLINK2 --pfile ${INPUT_DIR}/all_phase3 vzs --max-alleles 2 --make-bed --out ${SUBSETS_DIR}/all_phase3
