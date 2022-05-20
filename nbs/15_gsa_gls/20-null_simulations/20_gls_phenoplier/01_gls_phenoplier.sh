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

if [ -z "${BATCH_ID}" ]; then
    >&2 echo "Error, --batch-id <value> not provided"
    exit 1
fi

if [ -z "${BATCH_N_SPLITS}" ]; then
    >&2 echo "Error, --batch-n-splits <value> not provided"
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

# Create output directory
#mkdir -p ${OUTPUT_DIR}
#OUTPUT_FILENAME_BASE="${PHENOTYPE_NAME}-gtex_v8-mashr-smultixcan"

python ${PHENOPLIER_CODE_DIR}/libs/gls_cli.py \
    -i ${INPUT_FILE} \
    -o ${OUTPUT_FILE} \
    --duplicated-genes-action keep-first \
    --batch-id ${BATCH_ID} \
    --batch-n-splits ${BATCH_N_SPLITS}

