# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] tags=[]
# # Description

# %% tags=[]

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
import os
from pathlib import Path

# %% [markdown] tags=[]
# # Settings

# %% tags=["parameters"]
PHENOPLIER_NOTEBOOK_FILEPATH = None
PHENOPLIER_NOTEBOOK_DIR = os.getcwd()

# %% tags=[]
if PHENOPLIER_NOTEBOOK_FILEPATH is not None:
    PHENOPLIER_NOTEBOOK_DIR = str(Path(PHENOPLIER_NOTEBOOK_FILEPATH).parent)

display(PHENOPLIER_NOTEBOOK_DIR)

# %% [markdown] tags=[]
# # Run

# %% tags=[]
# %env PHENOPLIER_NOTEBOOK_DIR=$PHENOPLIER_NOTEBOOK_DIR

# %% tags=[] language="bash"
# run_job () {
#   # read trait information
#   IFS=',' read -r pheno_id desc file sample_size n_cases <<< "$1"
#
#   # CODE_RELATIVE_DIR="$1"
#   CODE_DIR="${PHENOPLIER_NOTEBOOK_DIR}"
#
#   # GWAS_DIR=${PHENOPLIER_PROJECTS_ASTHMA_COPD_RESULTS_DIR}/final_imputed_gwas
#   INPUT_FILENAME=${file%.*}
#   GENE_CORR_FILE="${PHENOPLIER_PROJECTS_ASTHMA_COPD_RESULTS_DIR}/gls_phenoplier/gene_corrs/${pheno_id}/gene_corrs-symbols.per_lv"
#
#   SMULTIXCAN_DIR="${PHENOPLIER_PROJECTS_ASTHMA_COPD_RESULTS_DIR}/twas/smultixcan"
#   SMULTIXCAN_FILE="${SMULTIXCAN_DIR}/${INPUT_FILENAME}-gtex_v8-mashr-smultixcan.txt"
#
#   OUTPUT_DIR="${PHENOPLIER_PROJECTS_ASTHMA_COPD_RESULTS_DIR}/gls_phenoplier/gls"
#   mkdir -p "${OUTPUT_DIR}"
#
#   OUTPUT_FILENAME_BASE="${INPUT_FILENAME}-gls_phenoplier"
#
#   LOGS_DIR="${CODE_DIR}/jobs_output"
#   mkdir -p "${LOGS_DIR}"
#
#   # make sure we are not also parallelizing within numpy, etc
#   export NUMBA_NUM_THREADS=1
#   export MKL_NUM_THREADS=1
#   export OPEN_BLAS_NUM_THREADS=1
#   export NUMEXPR_NUM_THREADS=1
#   export OMP_NUM_THREADS=1
#
#   echo "Running for $pheno_id"
#   echo "Saving results in ${OUTPUT_DIR}/${OUTPUT_FILENAME_BASE}.tsv.gz"
#   echo "Saving logs in ${LOGS_DIR}/${OUTPUT_FILENAME_BASE}.log"
#
#   bash "${PHENOPLIER_CODE_DIR}/scripts/gls_phenoplier.sh" \
#     --input-file "${SMULTIXCAN_FILE}" \
#     --gene-corr-file "${GENE_CORR_FILE}" \
#     --covars "gene_size gene_size_log gene_density gene_density_log" \
#     --debug-use-sub-gene-corr 1 \
#     --output-file "${OUTPUT_DIR}/${OUTPUT_FILENAME_BASE}.tsv.gz" > "${LOGS_DIR}/${OUTPUT_FILENAME_BASE}.log" 2>&1
#
#   echo
# }
#
# # export function so GNU Parallel can see it
# export -f run_job
#
# # generate a list of run_job calls for GNU Parallel
# while IFS= read -r line; do
#     echo run_job "${line}"
# done < <(tail -n "+2" "${PHENOPLIER_PROJECTS_ASTHMA_COPD_DATA_DIR}/traits_info.csv") |
#     parallel -k --lb --halt 2 -j${PHENOPLIER_GENERAL_N_JOBS}

# %% tags=[]
