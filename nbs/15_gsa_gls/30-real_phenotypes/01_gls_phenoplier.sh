#!/bin/bash

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
    --debug-use-ols)
      DEBUG_USE_OLS="$2"
      shift # past argument
      shift # past value
      ;;
    --debug-use-sub-gene-corr)
      DEBUG_USE_SUB_CORR="$2"
      shift # past argument
      shift # past value
      ;;
    --covars)
      USE_COVARS="$2"
      shift # past argument
      shift # past value
      ;;
    --cohort-name)
      COHORT_NAME="$2"
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

if [ -z "${DEBUG_USE_OLS}" ] && [ -z "${GENE_CORR_FILE}" ]; then
    >&2 echo "Error, either --debug-use-ols or --gene-corr-file <value> must be provided"
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

GENE_CORRS_ARGS=""
if [ ! -z "${GENE_CORR_FILE}" ]; then
  GENE_CORRS_ARGS="--gene-corr-file ${GENE_CORR_FILE}"
elif [ ! -z "${DEBUG_USE_OLS}" ]; then
  GENE_CORRS_ARGS="--debug-use-ols"
else
  echo "Wrong arguments"
  exit 1
fi

COVARS_ARGS=""
if [ ! -z "${USE_COVARS}" ]; then
  COVARS_ARGS="--covars ${USE_COVARS}"
fi

COHORT_ARGS=""
if [ ! -z "${COHORT_NAME}" ]; then
  # FIXME: hardcoded
  COHORT_METADATA_DIR="${PHENOPLIER_RESULTS_GLS}/gene_corrs/cohorts/${COHORT_NAME}/gtex_v8/mashr/"
  COHORT_ARGS="--cohort-metadata-dir ${COHORT_METADATA_DIR}"
fi

if [ ! -z "${DEBUG_USE_SUB_CORR}" ]; then
  GENE_CORRS_ARGS="${GENE_CORRS_ARGS} --debug-use-sub-gene-corr"
fi

BATCH_ARGS=""
if [ ! -z "${BATCH_ID}" ] && [ ! -z "${BATCH_N_SPLITS}" ]; then
  BATCH_ARGS="--batch-id ${BATCH_ID} --batch-n-splits ${BATCH_N_SPLITS}"
elif [ ! -z "${BATCH_ID}" ] || [ ! -z "${BATCH_N_SPLITS}" ]; then
  echo "Wrong arguments"
  exit 1
fi

LV_LIST_ARGS=""
if [ ! -z "${LV_LIST}" ]; then
  BATCH_ARGS="--lv-list ${LV_LIST}"
fi

python ${PHENOPLIER_CODE_DIR}/libs/gls_cli.py \
    -i ${INPUT_FILE} \
    --duplicated-genes-action keep-first \
    ${GENE_CORRS_ARGS} \
    -o ${OUTPUT_FILE} ${BATCH_ARGS} ${LV_LIST_ARGS} ${COVARS_ARGS} ${COHORT_ARGS}

