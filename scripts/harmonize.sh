#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# Runs the harmonization step of the pipeline here: https://github.com/hakyimlab/summary-gwas-imputation

# read arguments
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -i|--input-gwas-file)
      INPUT_GWAS_FILE="$2"
      shift # past argument
      shift # past value
      ;;
    -l|--liftover-chain-file)
      LIFTOVER_CHAIN_FILE="$2"
      shift # past argument
      shift # past value
      ;;
    -o|--output-dir)
      OUTPUT_DIR="$2"
      shift # past argument
      shift # past value
      ;;
    -s|--samples-n-cases)
      SAMPLES_N_CASES="$2"
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

if [ -z "${OUTPUT_DIR}" ]; then
    >&2 echo "Error, --output-dir <value> not provided"
    exit 1
fi

if [ -z "${SAMPLES_N_CASES}" ]; then
    >&2 echo "Error, --samples-n-cases <value> not provided"
    exit 1
fi

# liftover argument (optional)
LIFTOVER_ARG=""
if [ ! -z "${LIFTOVER_CHAIN_FILE}" ]; then
    LIFTOVER_ARG="-liftover ${LIFTOVER_CHAIN_FILE}"
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

A1000G_VARIANTS_METADATA_FILE="${PHENOPLIER_PHENOMEXCAN_LD_BLOCKS_1000G_GENOTYPE_DIR}/variant_metadata.txt.gz"
if [ ! -f ${A1000G_VARIANTS_METADATA_FILE} ]; then
    >&2 echo "The 1000 Genomes variants metadata file does not exist: ${A1000G_VARIANTS_METADATA_FILE}"
    exit 1
fi


# Create output directory
mkdir -p ${OUTPUT_DIR}

INPUT_GWAS_FILENAME=$(basename ${INPUT_GWAS_FILE})
OUTPUT_FILENAME=${INPUT_GWAS_FILENAME%.*}

${PYTHON_EXECUTABLE} ${PHENOPLIER_GWAS_IMPUTATION_BASE_DIR}/src/gwas_parsing.py \
    -gwas_file ${INPUT_GWAS_FILE} \
    -separator $'\t' \
    -snp_reference_metadata ${A1000G_VARIANTS_METADATA_FILE} METADATA \
    --chromosome_format \
    -output_column_map "#CHROM" chromosome \
    -output_column_map ID variant_id \
    -output_column_map A1 effect_allele \
    -output_column_map REF non_effect_allele \
    -output_column_map OR or \
    -output_column_map "LOG(OR)_SE" standard_error \
    -output_column_map P pvalue \
    -output_column_map POS position \
    -output_column_map OBS_CT sample_size \
    --insert_value n_cases ${SAMPLES_N_CASES} \
    -output_order variant_id panel_variant_id chromosome position effect_allele non_effect_allele frequency pvalue zscore effect_size standard_error sample_size n_cases \
    ${LIFTOVER_ARG} -output ${OUTPUT_DIR}/${OUTPUT_FILENAME}-harmonized.txt

