#!/bin/bash
# BSUB -J random_pheno${pheno_id}
# BSUB -cwd _tmp/gls_phenoplier_ols
# BSUB -oo random_pheno${pheno_id}.%J.out
# BSUB -eo random_pheno${pheno_id}.%J.error
# -#BSUB -u miltondp@gmail.com
# -#BSUB -N
# BSUB -n 1
# BSUB -R "rusage[mem=1GB]"
# BSUB -M 1GB
# BSUB -W 0:10

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

CODE_DIR=${PHENOPLIER_CODE_DIR}/nbs/15_gsa_gls/20-null_simulations/20_gls_phenoplier
INPUT_SMULTIXCAN_DIR="${PHENOPLIER_RESULTS_GLS_NULL_SIMS}/twas/smultixcan"

# Custom gene corr matrix
# GENE_CORR_FILE="/project/ritchie20/projects/phenoplier/base/data/tmp/gene_corrs/gene_corrs-default_identity_matrix.pkl"
OUTPUT_DIR="${PHENOPLIER_RESULTS_GLS_NULL_SIMS}/phenoplier/gls-debug_use_ols"

# # GTEx v8 / MASHR / max gene expression across tissues
# GENE_CORR_FILE="${PHENOPLIER_PHENOMEXCAN_LD_BLOCKS_GENE_CORRS_DIR}/gtex_v8/mashr/multiplier_genes-pred_expression_corr_avg-max-gene_symbols.pkl"
# OUTPUT_DIR="${PHENOPLIER_RESULTS_GLS_NULL_SIMS}/phenoplier/gls-gtex-mashr-max_gene_expr"

mkdir -p ${OUTPUT_DIR}

bash ${CODE_DIR}/01_gls_phenoplier.sh \
  --input-file ${INPUT_SMULTIXCAN_DIR}/random.pheno${pheno_id}-gtex_v8-mashr-smultixcan.txt \
  --debug-use-ols 1 \
  --output-file ${OUTPUT_DIR}/random.pheno${pheno_id}-gls_phenoplier.tsv.gz

