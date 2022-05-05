#!/bin/bash

# Runs the imputation step.

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -i|--input-gwas-file)
      INPUT_GWAS_FILE="$2"
      shift # past argument
      shift # past value
      ;;
    -f|--imputed-gwas-folder)
      IMPUTED_GWAS_FOLDER="$2"
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
    echo "Error, --input-gwas-file <value> not provided"
    exit 1
fi

if [ -z "${IMPUTED_GWAS_FOLDER}" ]; then
    echo "Error, --imputed-gwas-folder <value> not provided"
    exit 1
fi

if [ -z "${PHENOTYPE_NAME}" ]; then
    echo "Error, --phenotype-name <value> not provided"
    exit 1
fi

if [ -z "${OUTPUT_DIR}" ]; then
    echo "Error, --output-dir <value> not provided"
    exit 1
fi


#
# Global PhenoPLIER environmental variables
#

# make sure we have environment variables with configuration
if [ -z "${PHENOPLIER_ROOT_DIR}" ] || [ -z "${PHENOPLIER_GWAS_IMPUTATION_BASE_DIR}" ]; then
    echo "PhenoPLIER configuration was not loaded"
    exit 1
fi

PYTHON_EXECUTABLE="${PHENOPLIER_GWAS_IMPUTATION_CONDA_ENV}/bin/python"
if [ ! -f ${PYTHON_EXECUTABLE} ]; then
    echo "The python executable does not exist: ${PYTHON_EXECUTABLE}"
    exit 1
fi


# make sure the the input gwas file is for the phenotype name given
INPUT_GWAS_FILENAME=$(basename ${INPUT_GWAS_FILE})
if [[ ! "${INPUT_GWAS_FILENAME}" == *"${PHENOTYPE_NAME}"* ]]; then
  echo "Phenotype name given (${PHENOTYPE_NAME}) is not present in input GWAS file name (${INPUT_GWAS_FILENAME})."
  exit 1
fi

# make sure the number of files with imputed variants is 22 * 10 = 220
N_FILES_IMP_VARIANTS=`ls -dq ${IMPUTED_GWAS_FOLDER}/${PHENOTYPE_NAME}* | wc -l`
if [ "${N_FILES_IMP_VARIANTS}" -ne "220" ]; then
  echo "Number of expected files with imputed variants (220) does not match expected value (${N_FILES_IMP_VARIANTS}). Check your phenotype name."
  exit 1
fi

# Create output directory
mkdir -p ${OUTPUT_DIR}

${PYTHON_EXECUTABLE} ${PHENOPLIER_GWAS_IMPUTATION_BASE_DIR}/src/gwas_summary_imputation_postprocess.py \
    -gwas_file ${INPUT_GWAS_FILE} \
    -folder ${IMPUTED_GWAS_FOLDER} \
    -pattern ${PHENOTYPE_NAME}.* \
    -parsimony 7 \
    -output ${OUTPUT_DIR}/${PHENOTYPE_NAME}-imputed.txt.gz

