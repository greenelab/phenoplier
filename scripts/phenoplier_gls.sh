#!/bin/bash
set -euo pipefail
# IFS=$'\n\t'

# This scripts is a shortcut to run gls_cli.py. It provides a simpler interface
# with common and default parameter values.

# Runs GLS PhenoPLIER

# read arguments
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input-file)
            INPUT_FILE="$2"
            shift # past argument
            shift # past value
            ;;
        -g|--gene-corr-file)
            GENE_CORR_FILE="$2"
            shift # past argument
            shift # past value
            ;;
        --lv-list)
            LV_LIST="$2"
            shift # past argument
            shift # past value
            ;;
        -b|--batch-id)
            BATCH_ID="$2"
            shift # past argument
            shift # past value
            ;;
        -n|--batch-n-splits)
            BATCH_N_SPLITS="$2"
            shift # past argument
            shift # past value
            ;;
        -o|--output-file)
            OUTPUT_FILE="$2"
            shift # past argument
            shift # past value
            ;;
        --covars)
            USE_COVARS="$2"
            shift # past argument
            shift # past value
            ;;
        -*|--*)
            echo "Unknown option $1"
            exit 1
            ;;
        *)
            POSITIONAL_ARGS+=("$1") # save positional arg
            shift # past argument
;;
    esac
done

set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters


#
# check arguments
#
if [ -z "${INPUT_FILE}" ]; then
    >&2 echo "Error, --input-file <value> not provided"
    exit 1
fi

if [ -z "${GENE_CORR_FILE}" ]; then
    >&2 echo "Error, --gene-corr-file <value> must be provided"
    exit 1
fi

if [ -z "${OUTPUT_FILE}" ]; then
    >&2 echo "Error, --output-file <value> not provided"
    exit 1
fi


#
# Global PhenoPLIER environmental variables
#

# make sure we have environment variables with configuration
if [ -z "${PHENOPLIER_ROOT_DIR}" ] || [ -z "${PHENOPLIER_METAXCAN_BASE_DIR}" ]; then
    >&2 echo "PhenoPLIER configuration was not loaded"
    exit 1
fi

if ! python -c "from gls import GLSPhenoplier"; then
    >&2 echo "Conda environment for PhenoPLIER is not not activated"
    exit 1
fi

COVARS_ARGS=""
if [ ! -z "${USE_COVARS}" ]; then
    COVARS_ARGS="--covars ${USE_COVARS}"
fi

BATCH_ARGS=""
if [ ! -z "${BATCH_ID:-}" ] && [ ! -z "${BATCH_N_SPLITS:-}" ]; then
    BATCH_ARGS="--batch-id ${BATCH_ID} --batch-n-splits ${BATCH_N_SPLITS}"
elif [ ! -z "${BATCH_ID:-}" ] || [ ! -z "${BATCH_N_SPLITS:-}" ]; then
    echo "Wrong arguments"
    exit 1
fi

LV_LIST_ARGS=""
if [ ! -z "${LV_LIST:-}" ]; then
    BATCH_ARGS="--lv-list ${LV_LIST}"
fi

python ${PHENOPLIER_CODE_DIR}/libs/gls_cli.py \
    -i "${INPUT_FILE}" \
    --duplicated-genes-action "keep-first" \
    --gene-corr-file "${GENE_CORR_FILE}" \
    --debug-use-sub-gene-corr \
    -o "${OUTPUT_FILE}" ${BATCH_ARGS} ${LV_LIST_ARGS} ${COVARS_ARGS}
