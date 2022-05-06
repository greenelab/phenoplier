#!/bin/bash

# Runs S-MultiXcan.

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -i|--input-gwas-file)
      INPUT_GWAS_FILE="$2"
      shift # past argument
      shift # past value
      ;;
    -t|--spredixcan-folder)
      SPREDIXCAN_FOLDER="$2"
      shift # past argument
      shift # past value
      ;;
    -n|--phenotype-name)
      PHENOTYPE_NAME="$2"
      shift # past argument
      shift # past value
      ;;
    -o|--output-dir)
      OUTPUT_DIR="$2"
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
if [ -z "${INPUT_GWAS_FILE}" ]; then
    >&2 echo "Error, --input-gwas-file <value> not provided"
    exit 1
fi

if [ -z "${SPREDIXCAN_FOLDER}" ]; then
    >&2 echo "Error, --spredixcan-folder <value> not provided"
    exit 1
fi

if [ -z "${PHENOTYPE_NAME}" ]; then
    >&2 echo "Error, --phenotype-name <value> not provided"
    exit 1
fi

if [ -z "${OUTPUT_DIR}" ]; then
    >&2 echo "Error, --output-dir <value> not provided"
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

PYTHON_EXECUTABLE="${PHENOPLIER_METAXCAN_CONDA_ENV}/bin/python"
if [ ! -f ${PYTHON_EXECUTABLE} ]; then
    >&2 echo "The python executable does not exist: ${PYTHON_EXECUTABLE}"
    exit 1
fi

# Create output directory
mkdir -p ${OUTPUT_DIR}
OUTPUT_FILENAME_BASE="${PHENOTYPE_NAME}-gtex_v8-mashr-smultixcan"

${PYTHON_EXECUTABLE} ${PHENOPLIER_METAXCAN_BASE_DIR}/software/SMulTiXcan.py \
    --models_folder ${PHENOPLIER_PHENOMEXCAN_PREDICTION_MODELS_MASHR} \
    --models_name_pattern "${PHENOPLIER_PHENOMEXCAN_PREDICTION_MODELS_MASHR_PREFIX}(.*).db" \
    --snp_covariance ${PHENOPLIER_PHENOMEXCAN_PREDICTION_MODELS_MASHR_SMULTIXCAN_COV_FILE} \
    --metaxcan_folder ${SPREDIXCAN_FOLDER} \
    --metaxcan_filter "${PHENOTYPE_NAME}\-.*csv" \
    --metaxcan_file_name_parse_pattern '(.*)\-gtex_v8\-mashr\-(.*).csv' \
    --gwas_file ${INPUT_GWAS_FILE} \
    --snp_column "panel_variant_id" \
    --effect_allele_column "effect_allele" \
    --non_effect_allele_column "non_effect_allele" \
    --zscore_column "zscore" \
    --keep_non_rsid \
    --model_db_snp_key varID \
    --cutoff_condition_number 30 \
    --verbosity 7 \
    --throw \
    --output ${OUTPUT_DIR}/${OUTPUT_FILENAME_BASE}.txt >> ${OUTPUT_DIR}/${OUTPUT_FILENAME_BASE}.log 2>&1
