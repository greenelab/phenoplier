#!/bin/bash

# Runs the harmonization step for all GWAS on random phenotypes.

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -i|--input-gwas-file)
      INPUT_GWAS_FILE="$2"
      shift # past argument
      shift # past value
      ;;
    -v|--variants-metadata)
      VARIANTS_METADATA_FILE="$2"
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


SOFTWARE_DIR="/home/miltondp/projects/labs/greenelab/phenoplier/nbs/15_gsa_gls/20-null_simulations/10_gwas_harmonization/_software/summary-gwas-imputation"
#DATA_DIR="/home/miltondp/projects/labs/greenelab/phenoplier/nbs/15_gsa_gls/20-null_simulations/10_gwas_harmonization/_data"
#GWAS_DIR="/home/miltondp/projects/labs/greenelab/phenoplier/base/data/1000g/genotypes/gwas"
#A1000G_REFERENCE_DATA="/home/miltondp/projects/labs/greenelab/phenoplier/base/data/phenomexcan/ld_blocks/reference_panel_1000G"

mkdir -p ${OUTPUT_DIR}

LIFTOVER_ARG=""
if [ ! -z "${LIFTOVER_CHAIN_FILE}" ]; then
    LIFTOVER_ARG="-liftover ${LIFTOVER_CHAIN_FILE}"
fi

OUTPUT_FILENAME=${INPUT_GWAS_FILE%.*}
echo $OUTPUT_FILENAME

python3 ${SOFTWARE_DIR}/src/gwas_parsing.py \
    -gwas_file ${INPUT_GWAS_FILE} \
    -snp_reference_metadata ${VARIANTS_METADATA_FILE} METADATA \
    --chromosome_format \
    -output_column_map ID variant_id \
    -output_column_map A1 effect_allele \
    -output_column_map REF non_effect_allele \
    -output_column_map T_STAT zscore \
    -output_column_map BETA effect_size \
    -output_column_map SE standard_error \
    -output_column_map P pvalue \
    -output_column_map "#CHROM" chromosome \
    -output_column_map POS position \
    -output_column_map OBS_CT sample_size \
    -output_order variant_id panel_variant_id chromosome position effect_allele non_effect_allele frequency pvalue zscore effect_size standard_error sample_size n_cases \
    ${LIFTOVER_ARG} -output ${OUTPUT_DIR}/${OUTPUT_FILENAME}-harmonized.txt

