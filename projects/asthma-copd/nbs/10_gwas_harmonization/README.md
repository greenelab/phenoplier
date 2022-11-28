# Overview

This folder has the scripts to run the harmonization and imputation process across all GWAS (see `../05_gwas`).
It uses a standard pipeline for this task: https://github.com/hakyimlab/summary-gwas-imputation 


# Setup

## Desktop computer

Set the executor to bash:
```bash
. projects/asthma-copd/scripts/env.sh
```

For this, it's convenient to use Docker by running the specified command between single quotes:

```bash
bash scripts/run_docker_dev.sh '[COMMAND]'
```

## Download the necessary data

TODO: add GWAS download of asthma-only, copd-only, acos

```bash
python environment/scripts/setup_data.py \
  --actions \
    download_1000g_genotype_data \
    download_liftover_hg19tohg38_chain \
    download_eur_ld_regions \
    download_setup_summary_gwas_imputation
```


# Run

```bash
# export code dir
export CODE_DIR='${PHENOPLIER_CODE_DIR}'/projects/asthma-copd/nbs/10_gwas_harmonization
```

## Harmonization

```bash
export GWAS_DIR='${PHENOPLIER_PROJECTS_ASTHMA_COPD_DATA_DIR}'/gwas
export OUTPUT_DIR='${PHENOPLIER_PROJECTS_ASTHMA_COPD_RESULTS_DIR}'/harmonized_gwas
export LIFTOVER_CHAIN_FILE_PATH='${PHENOPLIER_GENERAL_LIFTOVER_HG19_TO_HG38}'

#
# Asthma-only
#
export GWAS_FILE_NAME="GWAS_Asthma_only_GLM_SNPs_info0.7.txt"
export GWAS_SAMPLES_N_CASES=19217

bash scripts/run_docker_dev.sh \
  bash ${CODE_DIR}/01_harmonize.sh \
    --input-gwas-file ${GWAS_DIR}/${GWAS_FILE_NAME} \
    --samples-n-cases ${GWAS_SAMPLES_N_CASES} \
    --liftover-chain-file ${LIFTOVER_CHAIN_FILE_PATH} \
    --output-dir ${OUTPUT_DIR}

#
# COPD-only
#
export GWAS_FILE_NAME="GWAS_COPD_only_GLM_SNPs_info0.7.txt"
export GWAS_SAMPLES_N_CASES=13055

bash scripts/run_docker_dev.sh \
  bash ${CODE_DIR}/01_harmonize.sh \
    --input-gwas-file ${GWAS_DIR}/${GWAS_FILE_NAME} \
    --samples-n-cases ${GWAS_SAMPLES_N_CASES} \
    --liftover-chain-file ${LIFTOVER_CHAIN_FILE_PATH} \
    --output-dir ${OUTPUT_DIR}

#
# ACOS
#
export GWAS_FILE_NAME="GWAS_ACO_GLM_SNPs_info0.7.txt"
export GWAS_SAMPLES_N_CASES=7035

bash scripts/run_docker_dev.sh \
  bash ${CODE_DIR}/01_harmonize.sh \
    --input-gwas-file ${GWAS_DIR}/${GWAS_FILE_NAME} \
    --samples-n-cases ${GWAS_SAMPLES_N_CASES} \
    --liftover-chain-file ${LIFTOVER_CHAIN_FILE_PATH} \
    --output-dir ${OUTPUT_DIR}
```
