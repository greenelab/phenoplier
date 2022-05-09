#!/bin/bash

# It uses the correlations computed previously on the sex-chromosomes to check sex discrepancy on a set of
# independent SNPs.

N_JOBS="${PHENOPLIER_GENERAL_N_JOBS}"
INPUT_DIR="${PHENOPLIER_A1000G_GENOTYPES_DIR}"
SUBSETS_DIR="${PHENOPLIER_A1000G_GENOTYPES_DIR}/subsets"
PLINK2="${PHENOPLIER_PLINK_EXECUTABLE_VERSION_2}"
PLINK19="${PHENOPLIER_PLINK_EXECUTABLE_VERSION_1_9}"

# --split-x
$PLINK19 --bfile ${SUBSETS_DIR}/all_phase3.1 \
    --threads ${N_JOBS} \
    --allow-extra-chr \
    --split-x b37 'no-fail' \
    --make-bed \
    --out ${SUBSETS_DIR}/all_phase3.1.split_x

# sex discrepancy (--check-sex)
$PLINK19 --bfile ${SUBSETS_DIR}/all_phase3.1.split_x \
    --threads ${N_JOBS} \
    --extract ${SUBSETS_DIR}/all_phase3.1.chrX.indepSNP.prune.in \
    --allow-extra-chr \
    --check-sex \
    --out ${SUBSETS_DIR}/all_phase3.1.split_x.sexcheck
