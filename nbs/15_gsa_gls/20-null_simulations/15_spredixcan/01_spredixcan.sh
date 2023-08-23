#!/bin/bash

# Runs S-PrediXcan.

# read arguments
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -d|--gwas-dir)
      INPUT_GWAS_DIR="$2"
      shift # past argument
      shift # past value
      ;;
    -n|--phenotype-name)
      PHENOTYPE_NAME="$2"
      shift # past argument
      shift # past value
      ;;
    -t|--tissue)
      TISSUE="$2"
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
if [ -z "${INPUT_GWAS_DIR}" ]; then
    >&2 echo "Error, --gwas-dir <value> not provided"
    exit 1
fi

if [ -z "${PHENOTYPE_NAME}" ]; then
    >&2 echo "Error, --phenotype-name <value> not provided"
    exit 1
fi

if [ -z "${TISSUE}" ]; then
    >&2 echo "Error, --tissue <value> not provided"
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
OUTPUT_FILENAME_BASE="${PHENOTYPE_NAME}-gtex_v8-mashr-${TISSUE}"

# FIXME: in the future:
#  * use parameter --gwas-file instead of --gwas_folder and --gwas_file_pattern
#  * add --throw (it's suggested in the documentation, but does not seem necessary)
${PYTHON_EXECUTABLE} ${PHENOPLIER_METAXCAN_BASE_DIR}/software/SPrediXcan.py \
    --model_db_path ${PHENOPLIER_PHENOMEXCAN_PREDICTION_MODELS_MASHR}/${PHENOPLIER_PHENOMEXCAN_PREDICTION_MODELS_MASHR_PREFIX}${TISSUE}.db \
    --covariance ${PHENOPLIER_PHENOMEXCAN_PREDICTION_MODELS_MASHR}/${PHENOPLIER_PHENOMEXCAN_PREDICTION_MODELS_MASHR_PREFIX}${TISSUE}.txt.gz \
    --gwas_folder ${INPUT_GWAS_DIR} \
    --gwas_file_pattern "${PHENOTYPE_NAME}.glm.*.txt.gz" \
    --separator $'\t' \
    --non_effect_allele_column "non_effect_allele" \
    --effect_allele_column "effect_allele" \
    --snp_column  "panel_variant_id" \
    --zscore_column "zscore" \
    --keep_non_rsid --additional_output --model_db_snp_key varID \
    --output_file ${OUTPUT_DIR}/${OUTPUT_FILENAME_BASE}.csv 2>&1 | tee -a ${OUTPUT_DIR}/${OUTPUT_FILENAME_BASE}.log

