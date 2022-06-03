#!/bin/bash
# BSUB -J genotype_compilation
# BSUB -cwd _tmp/
# BSUB -oo genotype_compilation%I.%J.out
# BSUB -eo genotype_compilation%I.%J.error
# -#BSUB -u miltondp@gmail.com
# -#BSUB -N
# BSUB -n 1
# BSUB -R "rusage[mem=25GB]"
# BSUB -M 25GB
# BSUB -W 5:00

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
OUTPUT_DIR=${PHENOPLIER_EXTERNAL_GTEX_V8_DIR}/generated/genotype
mkdir -p ${OUTPUT_DIR}
CODE_DIR=${PHENOPLIER_CODE_DIR}/nbs/15_gsa_gls/03_gtex_v8

PYTHON_EXECUTABLE="${PHENOPLIER_GWAS_IMPUTATION_CONDA_ENV}/bin/python"
if [ ! -f ${PYTHON_EXECUTABLE} ]; then
    >&2 echo "The python executable does not exist: ${PYTHON_EXECUTABLE}"
    exit 1
fi

${PYTHON_EXECUTABLE} ${PHENOPLIER_GWAS_IMPUTATION_BASE_DIR}/src/model_training_genotype_to_parquet.py \
  -input_genotype_file ${GTEX_V8_DIR}/generated/gtex_v8_eur_filtered.txt.gz \
  -snp_annotation_file ${GTEX_V8_DIR}/generated/gtex_v8_eur_filtered_maf0.01_monoallelic_variants.txt.gz METADATA \
  -parsimony 9 \
  --impute_to_mean \
  --split_by_chromosome \
  --only_in_key \
  -output_prefix ${OUTPUT_DIR}/gtex_v8_eur_filtered_maf0.01_monoallelic_variants

