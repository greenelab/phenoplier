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
# **TODO: UPDATE**
#
# It read all gene-gene correlation matrices across chromosomes, performs some tests and saves a final, singla gene-gene correlation matrix.

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
import os
from glob import glob
from pathlib import Path

from utils import read_log_file_and_check_line_exists
import conf

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
    "projects/asthma-copd/nbs/15_twas/05-run_smultixcan.ipynb"
)


# %% tags=[]
if PHENOPLIER_NOTEBOOK_FILEPATH is not None:
    PHENOPLIER_NOTEBOOK_DIR = str(Path(PHENOPLIER_NOTEBOOK_FILEPATH).parent)

display(PHENOPLIER_NOTEBOOK_DIR)

# %% tags=[]
SPREDIXCAN_DIR = conf.PROJECTS["ASTHMA_COPD"]["RESULTS_DIR"] / "twas" / "spredixcan"
display(SPREDIXCAN_DIR)
assert SPREDIXCAN_DIR.exists()

SPREDIXCAN_DIR_STR = str(SPREDIXCAN_DIR)
display(SPREDIXCAN_DIR_STR)

# %% tags=[]
OUTPUT_DIR = conf.PROJECTS["ASTHMA_COPD"]["RESULTS_DIR"] / "twas" / "smultixcan"
display(OUTPUT_DIR)

OUTPUT_DIR_STR = str(OUTPUT_DIR)
display(OUTPUT_DIR_STR)

# %% [markdown] tags=[]
# # Run

# %% tags=[] magic_args="-s \"$PHENOPLIER_NOTEBOOK_DIR\" \"$SPREDIXCAN_DIR_STR\" \"$OUTPUT_DIR_STR\"" language="bash"
# set -euo pipefail
# # IFS=$'\n\t'
#
# # read the notebook directory parameter and remove $1
# export PHENOPLIER_NOTEBOOK_DIR="${PHENOPLIER_CODE_DIR}/$1"
# shift
#
# # read S-PrediXcan input dir
# export SPREDIXCAN_DIR="$1"
# shift
#
# # read output dir
# export OUTPUT_DIR="$1"
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
#
#     # get input GWAS file, there should be a single file
#     # here I make sure that there are no other files in the folder that
#     # match this phenotype/trait filename prefix
#     GWAS_DIR="${PHENOPLIER_PROJECTS_ASTHMA_COPD_RESULTS_DIR}/final_imputed_gwas"
#     N_GWAS_FILES=$(ls ${GWAS_DIR}/${INPUT_FILENAME}* | wc -l)
#     if [ "${N_GWAS_FILES}" != "1" ]; then
#         echo "ERROR: found ${N_GWAS_FILES} GWAS files instead of one"
#         exit 1
#     fi
#     INPUT_GWAS_FILEPATH=$(ls ${GWAS_DIR}/${INPUT_FILENAME}*)
#
#     # OUTPUT_DIR="${PHENOPLIER_PROJECTS_ASTHMA_COPD_RESULTS_DIR}/twas/spredixcan"
#     mkdir -p "${OUTPUT_DIR}"
#
#     # make sure we are not also parallelizing within numpy, etc
#     export NUMBA_NUM_THREADS=1
#     export MKL_NUM_THREADS=1
#     export OPEN_BLAS_NUM_THREADS=1
#     export NUMEXPR_NUM_THREADS=1
#     export OMP_NUM_THREADS=1
#
#     echo "Running for $pheno_id"
#     echo "Saving results in ${OUTPUT_DIR}"
#
#     bash "${PHENOPLIER_CODE_DIR}/scripts/smultixcan.sh" \
#         --input-gwas-file "${INPUT_GWAS_FILEPATH}" \
#         --spredixcan-folder "${SPREDIXCAN_DIR}" \
#         --phenotype-name "${INPUT_FILENAME}" \
#         --output-dir "${OUTPUT_DIR}" \
#     | grep -iE "warning|error"
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

# %% [markdown] tags=[]
# # Perform some checks in output and log files

# %% tags=[]
assert OUTPUT_DIR.exists()

# %% tags=[]
log_files = OUTPUT_DIR.glob("*.log")

# %% tags=[]
for f in log_files:
    read_log_file_and_check_line_exists(
        f,
        [
            "INFO - Ran multi tissue in",
        ],
    )
