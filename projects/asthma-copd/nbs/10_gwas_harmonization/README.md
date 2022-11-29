# Overview

This folder has the scripts to run the harmonization and imputation process across all GWAS (see `../05_gwas`).
It uses a standard pipeline for this task: https://github.com/hakyimlab/summary-gwas-imputation 


# Setup

## Desktop computer

```bash
. projects/asthma-copd/scripts/env.sh

# export PHENOPLIER_JOBS_EXECUTOR="bash"
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


# traits info file
export TRAITS_INFO_FILE='${PHENOPLIER_PROJECTS_ASTHMA_COPD_DATA_DIR}'/traits_info.csv
```

## Harmonization

```bash
run_job () {
  IFS=',' read -r id desc file sample_size n_cases <<< "$1"
  
  export CODE_DIR=${PHENOPLIER_CODE_DIR}/projects/asthma-copd/nbs/10_gwas_harmonization
  
  export GWAS_DIR=${PHENOPLIER_PROJECTS_ASTHMA_COPD_DATA_DIR}/gwas
  export OUTPUT_DIR=${PHENOPLIER_PROJECTS_ASTHMA_COPD_RESULTS_DIR}/harmonized_gwas
  export LIFTOVER_CHAIN_FILE_PATH=${PHENOPLIER_GENERAL_LIFTOVER_HG19_TO_HG38}
  
  bash ${CODE_DIR}/01_harmonize.sh \
    --input-gwas-file ${GWAS_DIR}/${file} \
    --samples-n-cases ${n_cases} \
    --liftover-chain-file ${LIFTOVER_CHAIN_FILE_PATH} \
    --output-dir ${OUTPUT_DIR}
}

export -f run_job

# (optional) export function definition so it's included in the Docker container
export PHENOPLIER_BASH_FUNCTIONS_CODE="$(declare -f run_job)"

bash scripts/run_docker_dev.sh \
  'parallel -k --lb --halt 2 --skip-first-line -j${PHENOPLIER_GENERAL_N_JOBS} run_job < ${PHENOPLIER_PROJECTS_ASTHMA_COPD_DATA_DIR}/traits_info.csv'
```

## Imputation

```bash
run_job () {
  IFS=',' read -r id desc file sample_size n_cases chromosome batch_id <<< "$1"
  
  export CODE_DIR=${PHENOPLIER_CODE_DIR}/projects/asthma-copd/nbs/10_gwas_harmonization
  
  export GWAS_DIR=${PHENOPLIER_PROJECTS_ASTHMA_COPD_RESULTS_DIR}/harmonized_gwas
  export INPUT_FILENAME=${file%.*}-harmonized.txt
  export OUTPUT_DIR=${PHENOPLIER_PROJECTS_ASTHMA_COPD_RESULTS_DIR}/imputed_gwas
  
  export NUMBA_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPEN_BLAS_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1
  export OMP_NUM_THREADS=1
  
  bash ${CODE_DIR}/05_impute.sh \
    --input-gwas-file ${GWAS_DIR}/${INPUT_FILENAME} \
    --chromosome ${chromosome} \
    --n-batches 10 \
    --batch-id ${batch_id} \
    --output-dir ${OUTPUT_DIR}
}

export -f run_job

# (optional) export function definition so it's included in the Docker container
export PHENOPLIER_BASH_FUNCTIONS_CODE="$(declare -f run_job)"


bash scripts/run_docker_dev.sh \
'while IFS= read -r line; do
  for chromosome in {1..22}; do
    for batch_id in {0..9}; do
      echo "run_job $line,$chromosome,$batch_id"
    done
  done
done < <(tail -n "+2" ${PHENOPLIER_PROJECTS_ASTHMA_COPD_DATA_DIR}/traits_info.csv) | parallel -k --lb --halt 2 -j${PHENOPLIER_GENERAL_N_JOBS}'
```
