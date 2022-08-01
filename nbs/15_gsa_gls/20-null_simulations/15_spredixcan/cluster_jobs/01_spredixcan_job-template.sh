#!/bin/bash
# BSUB -J random_pheno${pheno_id}-${tissue}
# BSUB -cwd _tmp/spredixcan
# BSUB -oo random_pheno${pheno_id}-${tissue}.%J.out
# BSUB -eo random_pheno${pheno_id}-${tissue}.%J.error
# -#BSUB -u miltondp@gmail.com
# -#BSUB -N
# BSUB -n 1
# BSUB -R "rusage[mem=5GB]"
# BSUB -M 5GB
# BSUB -W 0:15

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

CODE_DIR=${PHENOPLIER_CODE_DIR}/nbs/15_gsa_gls/20-null_simulations/15_spredixcan
FINAL_IMPUTED_GWAS_DIR="${PHENOPLIER_RESULTS_GLS_NULL_SIMS}/final_imputed_gwas"
OUTPUT_DIR="${PHENOPLIER_RESULTS_GLS_NULL_SIMS}/twas/spredixcan"

bash ${CODE_DIR}/01_spredixcan.sh \
  --gwas-dir ${FINAL_IMPUTED_GWAS_DIR} \
  --phenotype-name "random.pheno${pheno_id}" \
  --tissue "${tissue}" \
  --output-dir ${OUTPUT_DIR}
