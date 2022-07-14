#!/bin/bash
# BSUB -J ${pheno_code}
# BSUB -cwd _tmp/gls_phenoplier_phenomexcan
# BSUB -oo ${pheno_code}.%J.out
# BSUB -eo ${pheno_code}.%J.error
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

CODE_DIR=${PHENOPLIER_CODE_DIR}/nbs/15_gsa_gls/30-real_phenotypes
#INPUT_SMULTIXCAN_DIR="${PHENOPLIER_RESULTS_GLS_NULL_SIMS}/twas/smultixcan"

# 1000G / MASHR
GENE_CORR_FILE="${PHENOPLIER_PHENOMEXCAN_LD_BLOCKS_GENE_CORRS_DIR}/1000g/mashr/multiplier_genes-gene_correlations-gene_symbols.pkl"
OUTPUT_DIR="${PHENOPLIER_RESULTS_GLS_PHENOMEXCAN}/gls-1000g_mashr-full_corr"

mkdir -p ${OUTPUT_DIR}

bash ${CODE_DIR}/01_gls_phenoplier.sh \
  --input-file ${pheno_filepath} \
  --gene-corr-file ${GENE_CORR_FILE} \
  --output-file ${OUTPUT_DIR}/${pheno_code}-gls_phenoplier.tsv.gz

