#!/bin/bash
#BSUB -J random_pheno[1-100]
#BSUB -cwd _tmp/harmonization
#BSUB -oo random_pheno%I.%J.out
#BSUB -eo random_pheno%I.%J.error
#-#BSUB -u miltondp@gmail.com
#-#BSUB -N
#BSUB -n 1
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 8GB
#BSUB -W 0:15

# make sure we use the number of CPUs specified
export CM_N_JOBS=1
export NUMBA_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPEN_BLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

CODE_DIR=${PHENOPLIER_CODE_DIR}/nbs/15_gsa_gls/20-null_simulations/10_gwas_harmonization
GWAS_DIR="${PHENOPLIER_RESULTS_GLS_NULL_SIMS}/gwas"
OUTPUT_DIR="${PHENOPLIER_RESULTS_GLS_NULL_SIMS}/harmonized_gwas"

GWAS_JOBINDEX=`expr $LSB_JOBINDEX - 1`

bash ${CODE_DIR}/01_harmonize.sh \
  --input-gwas-file ${GWAS_DIR}/random.pheno${GWAS_JOBINDEX}.glm.linear \
  --liftover-chain-file /project/ritchie20/projects/phenoplier/base/data/liftover/chains/hg19ToHg38.over.chain.gz \
  --output-dir ${OUTPUT_DIR}

