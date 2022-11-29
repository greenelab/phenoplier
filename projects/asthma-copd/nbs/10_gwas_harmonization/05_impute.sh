#!/bin/bash

# Runs the imputation step of the pipeline here: https://github.com/hakyimlab/summary-gwas-imputation

# read arguments
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -i|--input-gwas-file)
      INPUT_GWAS_FILE="$2"
      shift # past argument
      shift # past value
      ;;
    -c|--chromosome)
      CHROMOSOME="$2"
      shift # past argument
      shift # past value
      ;;
    -n|--n-batches)
      N_BATCHES="$2"
      shift # past argument
      shift # past value
      ;;
    -m|--batch-id)
      BATCH_ID="$2"
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

if [ -z "${CHROMOSOME}" ]; then
    >&2 echo "Error, --chromosome <value> not provided"
    exit 1
fi

if [ -z "${N_BATCHES}" ]; then
    >&2 echo "Error, --n-batches <value> not provided"
    exit 1
fi

if [ -z "${BATCH_ID}" ]; then
    >&2 echo "Error, --batch-id <value> not provided"
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
if [ -z "${PHENOPLIER_ROOT_DIR}" ] || [ -z "${PHENOPLIER_GWAS_IMPUTATION_BASE_DIR}" ]; then
    >&2 echo "PhenoPLIER configuration was not loaded"
    exit 1
fi

PYTHON_EXECUTABLE="${PHENOPLIER_GWAS_IMPUTATION_CONDA_ENV}/bin/python"
if [ ! -f ${PYTHON_EXECUTABLE} ]; then
    >&2 echo "The python executable does not exist: ${PYTHON_EXECUTABLE}"
    exit 1
fi

A1000G_VARIANTS_METADATA_FILE="${PHENOPLIER_PHENOMEXCAN_LD_BLOCKS_1000G_GENOTYPE_DIR}/variant_metadata.parquet"
if [ ! -f ${A1000G_VARIANTS_METADATA_FILE} ]; then
    >&2 echo "The 1000 Genomes variants metadata file does not exist: ${A1000G_VARIANTS_METADATA_FILE}"
    exit 1
fi


# Create output directory
mkdir -p ${OUTPUT_DIR}

INPUT_GWAS_FILENAME=$(basename ${INPUT_GWAS_FILE})
OUTPUT_FILENAME_PREFIX=${INPUT_GWAS_FILENAME%.*}-imputed

${PYTHON_EXECUTABLE} ${PHENOPLIER_GWAS_IMPUTATION_BASE_DIR}/src/gwas_summary_imputation.py \
    -by_region_file ${PHENOPLIER_GENERAL_EUR_LD_REGIONS_FILE} \
    -gwas_file ${INPUT_GWAS_FILE} \
    -parquet_genotype ${PHENOPLIER_PHENOMEXCAN_LD_BLOCKS_1000G_GENOTYPE_DIR}/chr${CHROMOSOME}.variants.parquet \
    -parquet_genotype_metadata ${A1000G_VARIANTS_METADATA_FILE} \
    -window 100000 \
    -parsimony 7 \
    -chromosome ${CHROMOSOME} \
    -regularization 0.1 \
    -frequency_filter 0.01 \
    -sub_batches ${N_BATCHES} \
    -sub_batch ${BATCH_ID} \
    --standardise_dosages \
    -output ${OUTPUT_DIR}/${OUTPUT_FILENAME_PREFIX}-chr${CHROMOSOME}-batch${BATCH_ID}_${N_BATCHES}.txt

