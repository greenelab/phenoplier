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
# It runs the GLS model (regression) of PhenoPLIER on a set of traits.

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
    "projects/asthma-copd/nbs/30_gls_phenoplier/01-run_gls_phenoplier.ipynb"
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
# export PHENOPLIER_NOTEBOOK_DIR="${PHENOPLIER_CODE_DIR}/$1"
# shift
#
# run_job () {
#     # run_job is a standard function name that performs a particular job
#     # depending on the context. It will be called by GNU Parallel below.
#     #
#     # The implementation here runs the GLS model of PhenoPLIER on a trait.
#
#     # read trait information
#     # the first parameter to this function is a string with values separated by
#     # commas (,). So here I split those into different variables.
#     IFS=',' read -r pheno_id file sample_size n_cases <<< "$1"
#
#     INPUT_FILENAME=${file%.*}
#     GENE_CORR_FILE="${PHENOPLIER_PROJECTS_ASTHMA_COPD_RESULTS_DIR}/gls_phenoplier/gene_corrs/${pheno_id}/gene_corrs-symbols.per_lv"
#
#     SMULTIXCAN_DIR="${PHENOPLIER_PROJECTS_ASTHMA_COPD_RESULTS_DIR}/twas/smultixcan"
#     SMULTIXCAN_FILE="${SMULTIXCAN_DIR}/${INPUT_FILENAME}-gtex_v8-mashr-smultixcan.txt"
#
#     OUTPUT_DIR="${PHENOPLIER_PROJECTS_ASTHMA_COPD_RESULTS_DIR}/gls_phenoplier/gls"
#     mkdir -p "${OUTPUT_DIR}"
#
#     OUTPUT_FILENAME_BASE="${INPUT_FILENAME}-gls_phenoplier"
#
#     LOG_FILE="${OUTPUT_DIR}/${OUTPUT_FILENAME_BASE}.log"
#
#     # make sure we are not also parallelizing within numpy, etc
#     export NUMBA_NUM_THREADS=1
#     export MKL_NUM_THREADS=1
#     export OPEN_BLAS_NUM_THREADS=1
#     export NUMEXPR_NUM_THREADS=1
#     export OMP_NUM_THREADS=1
#
#     echo "Running for $pheno_id"
#     echo "Saving results in ${OUTPUT_DIR}/${OUTPUT_FILENAME_BASE}.tsv.gz"
#
#     bash "${PHENOPLIER_CODE_DIR}/scripts/phenoplier_gls.sh" \
#         --input-file "${SMULTIXCAN_FILE}" \
#         --gene-corr-file "${GENE_CORR_FILE}" \
#         --covars "gene_size gene_size_log gene_density gene_density_log" \
#         --output-file "${OUTPUT_DIR}/${OUTPUT_FILENAME_BASE}.tsv.gz" \
#     >"${LOG_FILE}" 2>&1
#
#     # print errors here in the notebook
#     cat "${LOG_FILE}" | grep -iE "warning|error"
#
#     echo
# }
#
# # export function so GNU Parallel can see it
# export -f run_job
#
# # generate a list of run_job calls for GNU Parallel
# # here I read a file with information about traits (one trait per line)
# while IFS= read -r line; do
#     echo run_job "${line}"
# done < <(tail -n "+2" "${PHENOPLIER_PROJECTS_ASTHMA_COPD_TRAITS_INFO_FILE}") |
#     parallel -k --group --halt 2 -j${PHENOPLIER_GENERAL_N_JOBS}

# %% tags=[]
