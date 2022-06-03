#!/bin/bash
# BSUB -J variant_selection
# BSUB -cwd _tmp/
# BSUB -oo variant_selection%I.%J.out
# BSUB -eo variant_selection%I.%J.error
# -#BSUB -u miltondp@gmail.com
# -#BSUB -N
# BSUB -n 1
# BSUB -R "rusage[mem=5GB]"
# BSUB -M 5GB
# BSUB -W 1:00

# Code taken and adapted from:
#  https://github.com/hakyimlab/summary-gwas-imputation/wiki/Reference-Data-Set-Compilation

# make sure we use the number of CPUs specified
export PHENOPLIER_N_JOBS=1
export NUMBA_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPEN_BLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

GTEX_V8_DIR=${PHENOPLIER_EXTERNAL_GTEX_V8_DIR}
CODE_DIR=${PHENOPLIER_CODE_DIR}/nbs/15_gsa_gls/03_gtex_v8

PYTHON_EXECUTABLE="${PHENOPLIER_GWAS_IMPUTATION_CONDA_ENV}/bin/python"
if [ ! -f ${PYTHON_EXECUTABLE} ]; then
    >&2 echo "The python executable does not exist: ${PYTHON_EXECUTABLE}"
    exit 1
fi

${PYTHON_EXECUTABLE} ${PHENOPLIER_GWAS_IMPUTATION_BASE_DIR}/src/get_reference_metadata.py \
  -genotype ${GTEX_V8_DIR}/generated/gtex_v8_eur_filtered.txt.gz \
  -annotation ${GTEX_V8_DIR}/analysis_supplement/GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze.lookup_table.txt.gz \
  -filter MAF 0.01 \
  -filter TOP_CHR_POS_BY_FREQ \
  -output ${GTEX_V8_DIR}/generated/gtex_v8_eur_filtered_maf0.01_monoallelic_variants.txt.gz

