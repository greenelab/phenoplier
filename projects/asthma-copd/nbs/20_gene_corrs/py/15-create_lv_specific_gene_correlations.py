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
# It generates LV-specific gene correlations matrices that will be used in the GLS model. The aim of these LV-specific matrices is to speed up the GLS model when computing an LV-trait association.

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
import os
from pathlib import Path

from utils import chunker

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
PHENOPLIER_NOTEBOOK_FILEPATH = "projects/asthma-copd/nbs/20_gene_corrs/15-create_lv_specific_gene_correlations.ipynb"


# %% tags=[]
if PHENOPLIER_NOTEBOOK_FILEPATH is not None:
    PHENOPLIER_NOTEBOOK_DIR = str(Path(PHENOPLIER_NOTEBOOK_FILEPATH).parent)

display(PHENOPLIER_NOTEBOOK_DIR)

# %% [markdown] tags=[]
# # Generate list of LVs to parallelize

# %% tags=[]
CHUNK_SIZE = 50

# %% tags=[]
# generate a list of chunks with CHUNK_SIZE LVs in each chunk
chunks = list(chunker([lv_code for lv_code in range(1, 987 + 1)], CHUNK_SIZE))

# for each chunk, generate the range
chunks = [f"{min(lvs)}-{max(lvs)}" for lvs in chunks]
display(chunks[:5])

# now join all chunks and separate them by a comma
LV_CHUNKS = ",".join(chunks)

# %% tags=[]
LV_CHUNKS

# %% [markdown] tags=[]
# # Run

# %% tags=[] magic_args="-s \"$PHENOPLIER_NOTEBOOK_DIR\" \"$LV_CHUNKS\"" language="bash"
# set -euo pipefail
# IFS=$'\n\t'
#
# # read the notebook directory parameter and remove $1
# export PHENOPLIER_NOTEBOOK_DIR="$1"
# shift
#
# # read LVs chunks
# export LV_CHUNKS="$1"
# shift
#
# run_job () {
#     # run_job is a standard function name that performs a particular job
#     # depending on the context. It will be called by GNU Parallel below.
#
#     # read trait information
#     # the first parameter to this function is a string with values separated by
#     # commas (,). So here I split those into different variables.
#     IFS=',' read -r pheno_id file sample_size n_cases lv_range <<< "$1"
#
#     LV_CODE="LV${lv_code}"
#
#     # LV_PERC has to match whatever is used in the GLS models: the top
#     # (LV_PERC * 100) percent of genes in an LV
#     LV_PERC=0.01
#
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
#     echo "Running for $pheno_id and range $lv_range"
#
#     bash nbs/run_nbs.sh \
#         "${PHENOPLIER_NOTEBOOK_DIR}/jobs/15-create_corr_mat_per_lv.ipynb" \
#         "../${NOTEBOOK_OUTPUT_FOLDER}/15-create_corr_mat_per_lv-${lv_range}.run.ipynb" \
#         -p COHORT_NAME "$pheno_id" \
#         -p LV_RANGE "$lv_range" \
#         -p LV_PERCENTILE "$LV_PERC" \
#         -p OUTPUT_DIR_BASE "$OUTPUT_DIR" \
#     &>/dev/null
# }
#
# # export function so GNU Parallel can see it
# export -f run_job
#
# IFS=',' read -ra LV_RANGES <<< "$LV_CHUNKS"
#
# # generate a list of run_job calls for GNU Parallel
# while IFS= read -r line; do
#     for lv_range in "${LV_RANGES[@]}"; do
#         echo run_job "$line,$lv_range"
#     done
# done < <(tail -n "+2" "${PHENOPLIER_PROJECTS_ASTHMA_COPD_TRAITS_INFO_FILE}") |
#     parallel -k --lb --halt 2 -j${PHENOPLIER_GENERAL_N_JOBS}

# %% tags=[]
