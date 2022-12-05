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

# %% [markdown] tags=[]
# It computes gene-gene correlation for all traits in the project. For that, it runs another notebook (in the `jobs/` subfolder) for each chromosome/trait pair. Later, these submatrices will be combined to generate a single gene-gene correlation matrix.

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
import os
from pathlib import Path

# %% [markdown] tags=[]
# # Settings

# %% [markdown] tags=[]
# Apparently, there is no easy way to get the parent directory of
# a notebook in Jupyter, so here I get that information either from
# the parameter sent by `nbs/run_nbs.sh` (if called from command-line) or
# from `os.getcwd()` (if called from browser).

# %% tags=["parameters"]
PHENOPLIER_NOTEBOOK_FILEPATH = None
PHENOPLIER_NOTEBOOK_DIR = os.getcwd()

# %% tags=["injected-parameters"]
# Parameters
PHENOPLIER_NOTEBOOK_FILEPATH = (
    "projects/asthma-copd/nbs/20_gene_corrs/05-compute_gene_correlations.ipynb"
)


# %% tags=[]
if PHENOPLIER_NOTEBOOK_FILEPATH is not None:
    PHENOPLIER_NOTEBOOK_DIR = str(Path(PHENOPLIER_NOTEBOOK_FILEPATH).parent)

display(PHENOPLIER_NOTEBOOK_DIR)

# %% [markdown] tags=[]
# # Run

# %% tags=[] magic_args="-s \"$PHENOPLIER_NOTEBOOK_DIR\"" language="bash"
# set -euo pipefail
# IFS=$'\n\t'
#
# # read the notebook directory parameter and remove $1
# export PHENOPLIER_NOTEBOOK_DIR="$1"
# shift
#
# run_job () {
#     # run_job is a standard function name that performs a particular job
#     # depending on the context. It will be called by GNU Parallel below.
#
#     # read trait information
#     # the first parameter to this function is a string with values separated by
#     # commas (,). So here I split those into different variables.
#     IFS=',' read -r pheno_id file sample_size n_cases chromosome <<< "$1"
#
#     INPUT_FILENAME=${file%.*}
#
#     # get input GWAS file, there should be a single file
#     # here I make sure that there are no other files in the folder that
#     # match this phenotype/trait filename prefix
#     GWAS_DIR="${PHENOPLIER_PROJECTS_ASTHMA_COPD_RESULTS_DIR}/final_imputed_gwas"
#     N_GWAS_FILES=$(ls ${GWAS_DIR}/${INPUT_FILENAME}*.txt.gz | wc -l)
#     if [ "${N_GWAS_FILES}" != "1" ]; then
#         echo "ERROR: found ${N_GWAS_FILES} GWAS files instead of one"
#         exit 1
#     fi
#     INPUT_GWAS_FILEPATH=$(ls ${GWAS_DIR}/${INPUT_FILENAME}*.txt.gz)
#
#     SPREDIXCAN_DIR=${PHENOPLIER_PROJECTS_ASTHMA_COPD_RESULTS_DIR}/twas/spredixcan
#     SMULTIXCAN_DIR=${PHENOPLIER_PROJECTS_ASTHMA_COPD_RESULTS_DIR}/twas/smultixcan
#     OUTPUT_DIR=${PHENOPLIER_PROJECTS_ASTHMA_COPD_RESULTS_DIR}/gls_phenoplier
#
#     # make sure we are not also parallelizing within numpy, etc
#     export NUMBA_NUM_THREADS=1
#     export MKL_NUM_THREADS=1
#     export OPEN_BLAS_NUM_THREADS=1
#     export NUMEXPR_NUM_THREADS=1
#     export OMP_NUM_THREADS=1
#
#     cd "${PHENOPLIER_CODE_DIR}"
#
#     NOTEBOOK_OUTPUT_FOLDER="output/${pheno_id,,}"
#     FULL_NOTEBOOK_OUTPUT_FOLDER="${PHENOPLIER_NOTEBOOK_DIR}/${NOTEBOOK_OUTPUT_FOLDER}"
#     mkdir -p "$FULL_NOTEBOOK_OUTPUT_FOLDER"
#
#     echo "Running for $pheno_id and chromosome $chromosome"
#
#     bash nbs/run_nbs.sh \
#         "${PHENOPLIER_NOTEBOOK_DIR}/jobs/05-gene_expr_correlations.ipynb" \
#         "../${NOTEBOOK_OUTPUT_FOLDER}/05-gene_expr_correlations-chr${chromosome}.run.ipynb" \
#         -p COHORT_NAME "$pheno_id" \
#         -p CHROMOSOME "$chromosome" \
#         -p OUTPUT_DIR_BASE "$OUTPUT_DIR" \
#     &>/dev/null
# }
#
# # export function so GNU Parallel can see it
# export -f run_job
#
# # generate a list of run_job calls for GNU Parallel
# while IFS= read -r line; do
#     for chromosome in {1..22}; do
#         echo run_job "$line,$chromosome"
#     done
# done < <(tail -n "+2" "${PHENOPLIER_PROJECTS_ASTHMA_COPD_TRAITS_INFO_FILE}") |
#     parallel -k --lb --halt 2 -j${PHENOPLIER_GENERAL_N_JOBS}

# %% tags=[]
