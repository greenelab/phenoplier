# Overview

This folder has the scripts to run the harmonization and imputation process across all GWAS (see `../05_gwas`).
It uses a standard pipeline for this task: https://github.com/hakyimlab/summary-gwas-imputation 


# Setup

## Desktop computer

```bash
. projects/asthma-copd/scripts/env.sh
```


# Run

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
