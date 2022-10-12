#!/bin/bash
# BSUB -J random_pheno${pheno_id}
# BSUB -cwd _tmp/gls_phenoplier
# BSUB -oo random_pheno${pheno_id}.%J.out
# BSUB -eo random_pheno${pheno_id}.%J.error
# -#BSUB -u miltondp@gmail.com
# -#BSUB -N
# BSUB -n 1
# BSUB -R "rusage[mem=3GB]"
# BSUB -M 3GB
# BSUB -W 0:30
# BSUB -R 'select[hname!=lambda25]'

# IMPORTANT: this is not a ready-for-submission script, it's a template.
#   see README.md to know how to generate the actual job scripts.

# make sure we use the number of CPUs specified
export n_jobs=1
export PHENOPLIER_N_JOBS=${n_jobs}
export NUMBA_NUM_THREADS=${n_jobs}
export MKL_NUM_THREADS=${n_jobs}
export OPEN_BLAS_NUM_THREADS=${n_jobs}
export NUMEXPR_NUM_THREADS=${n_jobs}
export OMP_NUM_THREADS=${n_jobs}

# Settings
COHORT_NAME="1000g_eur"
REFERENCE_PANEL="1000g"
USING_COVARS="multixcan_gtex_v8/covars"

# Paths
CODE_DIR=${PHENOPLIER_CODE_DIR}/nbs/15_gsa_gls/20-null_simulations/20_gls_phenoplier
INPUT_SMULTIXCAN_DIR="${PHENOPLIER_RESULTS_GLS_NULL_SIMS}/twas/multixcan_gtex_v8"

OUTPUT_DIR="${PHENOPLIER_RESULTS_GLS_NULL_SIMS}/phenoplier/${COHORT_NAME}/${USING_COVARS}/gls-${REFERENCE_PANEL}_mashr-full_corr"
mkdir -p ${OUTPUT_DIR}

# Gene correlation matrix
GENE_CORR_FILE="${PHENOPLIER_RESULTS_GLS}/gene_corrs/cohorts/${COHORT_NAME}/${REFERENCE_PANEL}/mashr/gene_corrs-symbols.pkl"

bash ${CODE_DIR}/01_gls_phenoplier.sh \
  --input-file ${INPUT_SMULTIXCAN_DIR}/random.pheno${pheno_id}-gtex_v8-mashr-multixcan.txt \
  --gene-corr-file ${GENE_CORR_FILE} \
  --output-file ${OUTPUT_DIR}/random.pheno${pheno_id}-gls_phenoplier.tsv.gz

