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

# Global parameters
SOFTWARE_DIR="/home/miltondp/projects/labs/greenelab/phenoplier/nbs/15_gsa_gls/20-null_simulations/10_gwas_harmonization/_software/summary-gwas-imputation"
DATA_DIR="/home/miltondp/projects/labs/greenelab/phenoplier/nbs/15_gsa_gls/20-null_simulations/10_gwas_harmonization/_data"
#GWAS_DIR="/home/miltondp/projects/labs/greenelab/phenoplier/base/data/1000g/genotypes/gwas"
A1000G_REFERENCE_DATA_DIR="/home/miltondp/projects/labs/greenelab/phenoplier/base/data/phenomexcan/ld_blocks/reference_panel_1000G"
A1000G_VARIANTS_METADATA_FILE="${A1000G_REFERENCE_DATA_DIR}/variant_metadata.parquet"

PYTHON_EXECUTABLE="${PHENOPLIER_GWAS_IMPUTATION_CONDA_ENV}/bin/python"
if [ ! -f ${PYTHON_EXECUTABLE} ]; then
    echo "The python executable does not exist: ${PYTHON_EXECUTABLE}"
    exit 1
fi

A1000G_VARIANTS_METADATA_FILE="${PHENOPLIER_PHENOMEXCAN_LD_BLOCKS_1000G_GENOTYPE_DIR}/variant_metadata.txt.gz"
if [ ! -f ${A1000G_VARIANTS_METADATA_FILE} ]; then
    echo "The 1000 Genomes variants metadata file does not exist: ${A1000G_VARIANTS_METADATA_FILE}"
    exit 1
fi


# Create output directory
mkdir -p ${OUTPUT_DIR}

#INPUT_FILE="random_pheno0.glm.linear.harmo.txt"
#CHROMOSOME=$1
#N_BATCHES=$2
#BATCH_ID=$3

#INPUT_FILE_PREFIX=${INPUT_FILE%.txt}

INPUT_GWAS_FILENAME=$(basename ${INPUT_GWAS_FILE})
OUTPUT_FILENAME_PREFIX="${INPUT_GWAS_FILENAME%.*}-imputed"

${PYTHON_EXECUTABLE} ${PHENOPLIER_GWAS_IMPUTATION_BASE_DIR}/src/gwas_summary_imputation.py \
    -by_region_file ${DATA_DIR}/eur_ld.bed.gz \
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

