#!/bin/bash
# +

# -
N_JOBS="${PHENOPLIER_GENERAL_N_JOBS}"
INPUT_DIR="${PHENOPLIER_A1000G_GENOTYPES_DIR}"
SUBSETS_DIR="${PHENOPLIER_A1000G_GENOTYPES_DIR}/subsets"
PLINK2="${PHENOPLIER_PLINK_EXECUTABLE_VERSION_2}"
PLINK19="${PHENOPLIER_PLINK_EXECUTABLE_VERSION_1_9}"

# sex discrepancy (--check-sex)
#
# run it again with specific parameter to --check-sex according
# to the histogram in the previous notebook
$PLINK19 --bfile ${SUBSETS_DIR}/all_phase3.1.split_x \
    --threads ${N_JOBS} \
    --extract ${SUBSETS_DIR}/all_phase3.1.chrX.indepSNP.prune.in \
    --allow-extra-chr \
    --check-sex 0.20 0.80 \
    --out ${SUBSETS_DIR}/all_phase3.1.split_x.sexcheck

grep "PROBLEM" ${SUBSETS_DIR}/all_phase3.1.split_x.sexcheck.sexcheck | \
    awk '{print$1,$2}'> ${SUBSETS_DIR}/all_phase3.1.split_x.sexcheck.sex_discrepancy.txt

$PLINK19 --bfile ${SUBSETS_DIR}/all_phase3.1 \
    --threads ${N_JOBS} \
    --allow-extra-chr \
    --remove ${SUBSETS_DIR}/all_phase3.1.split_x.sexcheck.sex_discrepancy.txt \
    --make-bed \
    --out ${SUBSETS_DIR}/all_phase3.1.1
