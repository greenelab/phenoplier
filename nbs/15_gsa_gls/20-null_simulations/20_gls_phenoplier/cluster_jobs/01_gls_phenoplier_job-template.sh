#!/bin/bash
# BSUB -J random_pheno${pheno_id}-${batch_id}
# BSUB -cwd _tmp/gls_phenoplier
# BSUB -oo random_pheno${pheno_id}.%J.out
# BSUB -eo random_pheno${pheno_id}.%J.error
# -#BSUB -u miltondp@gmail.com
# -#BSUB -N
# BSUB -n 2
# BSUB -R "rusage[mem=3GB]"
# BSUB -M 3GB
# BSUB -W 1:00

# IMPORTANT: this is not a ready-for-submission script, it's a template.
#   see README.md to know how to generate the actual job scripts.

# make sure we use the number of CPUs specified
export n_jobs=2
export CM_N_JOBS=${n_jobs}
export NUMBA_NUM_THREADS=${n_jobs}
export MKL_NUM_THREADS=${n_jobs}
export OPEN_BLAS_NUM_THREADS=${n_jobs}
export NUMEXPR_NUM_THREADS=${n_jobs}
export OMP_NUM_THREADS=${n_jobs}

CODE_DIR=${PHENOPLIER_CODE_DIR}/nbs/15_gsa_gls/20-null_simulations/20_gls_phenoplier
INPUT_SMULTIXCAN_DIR="${PHENOPLIER_RESULTS_GLS_NULL_SIMS}/twas/smultixcan"
OUTPUT_DIR="${PHENOPLIER_RESULTS_GLS_NULL_SIMS}/phenoplier/gls"

mkdir -p ${OUTPUT_DIR}

bash ${CODE_DIR}/01_gls_phenoplier.sh \
  --input-file ${INPUT_SMULTIXCAN_DIR}/random.pheno${pheno_id}-gtex_v8-mashr-smultixcan.txt \
  --batch-id ${batch_id} \
  --batch-n-splits ${batch_n_splits} \
  --output-file ${OUTPUT_DIR}/random.pheno${pheno_id}-batch${batch_id}_${batch_n_splits}-gls_phenomexcan.tsv.gz

