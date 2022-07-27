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

# 1000G / MASHR
GENE_CORR_FILE="${PHENOPLIER_RESULTS_GLS}/gene_corrs/cohorts/1000g_eur/1000g/mashr/all_genes/gene_corrs-symbols.pkl"
#GENE_CORR_FILE="${PHENOPLIER_RESULTS_GLS}/gene_corrs/cohorts/1000g_eur/1000g/mashr/within_distance/gene_corrs-symbols.pkl"
OUTPUT_DIR="${PHENOPLIER_RESULTS_GLS_NULL_SIMS}/phenoplier/gls-1000g_mashr-full_corr"

mkdir -p ${OUTPUT_DIR}

bash ${CODE_DIR}/01_gls_phenoplier.sh \
  --input-file ${INPUT_SMULTIXCAN_DIR}/random.pheno${pheno_id}-gtex_v8-mashr-smultixcan.txt \
  --gene-corr-file ${GENE_CORR_FILE} \
  --covars "all" \
  --output-file ${OUTPUT_DIR}/random.pheno${pheno_id}-gls_phenoplier.tsv.gz

