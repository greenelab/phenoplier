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
# It runs the GWAS harmonization step from the script in: https://github.com/hakyimlab/summary-gwas-imputation

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
    "projects/asthma-copd/nbs/10_gwas_harmonization/01-run_harmonization.ipynb"
)


# %% tags=[]
if PHENOPLIER_NOTEBOOK_FILEPATH is not None:
    PHENOPLIER_NOTEBOOK_DIR = str(Path(PHENOPLIER_NOTEBOOK_FILEPATH).parent)

display(PHENOPLIER_NOTEBOOK_DIR)

# %% tags=[]
OUTPUT_DIR = conf.PROJECTS["ASTHMA_COPD"]["RESULTS_DIR"] / "harmonized_gwas"
display(OUTPUT_DIR)

OUTPUT_DIR_STR = str(OUTPUT_DIR)
display(OUTPUT_DIR_STR)

# %% [markdown] tags=[]
# # Run

# %% tags=[] magic_args="-s \"$PHENOPLIER_NOTEBOOK_DIR\" \"$OUTPUT_DIR_STR\"" language="bash"
# set -euo pipefail
# # IFS=$'\n\t'
#
# # read the notebook directory parameter and remove $1
# export PHENOPLIER_NOTEBOOK_DIR="${PHENOPLIER_CODE_DIR}/$1"
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
#     LIFTOVER_CHAIN_FILE_PATH="${PHENOPLIER_GENERAL_LIFTOVER_HG19_TO_HG38}"
#
#     # get input GWAS file, there should be a single file
#     # here I make sure that there are no other files in the folder that
#     # match this phenotype/trait filename prefix
#     GWAS_DIR="${PHENOPLIER_PROJECTS_ASTHMA_COPD_DATA_DIR}/gwas"
#     N_GWAS_FILES=$(ls ${GWAS_DIR}/${INPUT_FILENAME}* | wc -l)
#     if [ "${N_GWAS_FILES}" != "1" ]; then
#         echo "ERROR: found ${N_GWAS_FILES} GWAS files instead of one"
#         exit 1
#     fi
#     INPUT_GWAS_FILEPATH=$(ls ${GWAS_DIR}/${INPUT_FILENAME}*)
#
#     mkdir -p "${OUTPUT_DIR}"
#
#     OUTPUT_FILENAME_BASE="${INPUT_FILENAME}-harmonized"
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
#     echo "Saving results in ${OUTPUT_DIR}"
#
#     bash "${PHENOPLIER_CODE_DIR}/scripts/harmonize.sh" \
#         --input-gwas-file "${INPUT_GWAS_FILEPATH}" \
#         --samples-n-cases ${n_cases} \
#         --liftover-chain-file "${LIFTOVER_CHAIN_FILE_PATH}" \
#         --output-dir "${OUTPUT_DIR}" \
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
#     parallel -k --lb --halt 2 -j${PHENOPLIER_GENERAL_N_JOBS}

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
            "INFO - Finished converting GWAS in",
        ],
    )
