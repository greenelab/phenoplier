#!/bin/bash

SOFTWARE_DIR="/home/miltondp/projects/labs/greenelab/phenoplier/nbs/15_gsa_gls/20-null_simulations/10_gwas_harmonization/_software/summary-gwas-imputation"
DATA_DIR="/home/miltondp/projects/labs/greenelab/phenoplier/nbs/15_gsa_gls/20-null_simulations/10_gwas_harmonization/_data"
GWAS_DIR="/home/miltondp/projects/labs/greenelab/phenoplier/base/data/1000g/genotypes/gwas"
A1000G_REFERENCE_DATA="/home/miltondp/projects/labs/greenelab/phenoplier/base/data/phenomexcan/ld_blocks/reference_panel_1000G"

OUTPUT_DIR="outputs/harmonized"
mkdir -p ${OUTPUT_DIR}

python3 ${SOFTWARE_DIR}/src/gwas_parsing.py \
    -gwas_file ${GWAS_DIR}/random.pheno0.glm.linear \
    -snp_reference_metadata ${A1000G_REFERENCE_DATA}/variant_metadata.txt.gz METADATA \
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
    -liftover ${DATA_DIR}/hg19ToHg38.over.chain.gz \
    -output ${OUTPUT_DIR}/random_pheno0.glm.linear.harmo.txt

