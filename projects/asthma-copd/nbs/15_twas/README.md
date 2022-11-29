# Overview

This folder has the scripts to run the PrediXcan family of methods on imputed GWAS (`../10_gwas_harmonization`).


# Setup

## Desktop computer

```bash
. projects/asthma-copd/scripts/env.sh
```

## Download the necessary data

```bash
bash scripts/run_docker_dev.sh \
  python environment/scripts/setup_data.py \
    --actions \
      download_setup_metaxcan \
      download_predixcan_mashr_prediction_models \
      download_mashr_expression_smultixcan_snp_covariance
```


# Run

## S-PrediXcan

```bash
run_job () {
  IFS=',' read -r id desc file sample_size n_cases tissue <<< "$1"
  
  export CODE_DIR=${PHENOPLIER_CODE_DIR}/projects/asthma-copd/nbs/15_twas
  
  export GWAS_DIR=${PHENOPLIER_PROJECTS_ASTHMA_COPD_RESULTS_DIR}/final_imputed_gwas
  export INPUT_FILENAME=${file%.*}
  export OUTPUT_DIR=${PHENOPLIER_PROJECTS_ASTHMA_COPD_RESULTS_DIR}/twas/spredixcan
  
  export NUMBA_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPEN_BLAS_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1
  export OMP_NUM_THREADS=1
  
  # get input GWAS file
  N_GWAS_FILES=$(ls ${GWAS_DIR}/${INPUT_FILENAME}* | wc -l)
  if [ "${N_GWAS_FILES}" != "1" ]; then
    echo "ERROR: found ${N_GWAS_FILES} GWAS files instead of one"
    exit 1
  fi
  INPUT_GWAS_FILEPATH=$(ls ${GWAS_DIR}/${INPUT_FILENAME}*)

  bash ${CODE_DIR}/01_spredixcan.sh \
    --input-gwas-file "${INPUT_GWAS_FILEPATH}" \
    --phenotype-name "${INPUT_FILENAME}" \
    --tissue "${tissue}" \
    --output-dir "${OUTPUT_DIR}"
}

export -f run_job

# (optional) export function definition so it's included in the Docker container
export PHENOPLIER_BASH_FUNCTIONS_CODE="$(declare -f run_job)"


bash scripts/run_docker_dev.sh \
'while IFS= read -r line; do
  for tissue in ${PHENOPLIER_PHENOMEXCAN_PREDICTION_MODELS_MASHR_TISSUES}; do
    echo "run_job $line,$tissue"
  done
done < <(tail -n "+2" ${PHENOPLIER_PROJECTS_ASTHMA_COPD_DATA_DIR}/traits_info.csv) | parallel -k --lb --halt 2 -j${PHENOPLIER_GENERAL_N_JOBS}'
```

## S-MultiXcan

```bash
run_job () {
  IFS=',' read -r id desc file sample_size n_cases <<< "$1"
  
  export CODE_DIR=${PHENOPLIER_CODE_DIR}/projects/asthma-copd/nbs/15_twas
  
  export GWAS_DIR=${PHENOPLIER_PROJECTS_ASTHMA_COPD_RESULTS_DIR}/final_imputed_gwas
  export INPUT_FILENAME=${file%.*}
  export SPREDIXCAN_DIR=${PHENOPLIER_PROJECTS_ASTHMA_COPD_RESULTS_DIR}/twas/spredixcan
  export OUTPUT_DIR=${PHENOPLIER_PROJECTS_ASTHMA_COPD_RESULTS_DIR}/twas/smultixcan
  
  export NUMBA_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPEN_BLAS_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1
  export OMP_NUM_THREADS=1
  
  # get input GWAS file
  N_GWAS_FILES=$(ls ${GWAS_DIR}/${INPUT_FILENAME}* | wc -l)
  if [ "${N_GWAS_FILES}" != "1" ]; then
    echo "ERROR: found ${N_GWAS_FILES} GWAS files instead of one"
    exit 1
  fi
  INPUT_GWAS_FILEPATH=$(ls ${GWAS_DIR}/${INPUT_FILENAME}*)
  
  bash ${CODE_DIR}/05_smultixcan.sh \
    --input-gwas-file "${INPUT_GWAS_FILEPATH}" \
    --spredixcan-folder "${SPREDIXCAN_DIR}" \
    --phenotype-name "${INPUT_FILENAME}" \
    --output-dir "${OUTPUT_DIR}"
}

export -f run_job

# (optional) export function definition so it's included in the Docker container
export PHENOPLIER_BASH_FUNCTIONS_CODE="$(declare -f run_job)"


bash scripts/run_docker_dev.sh \
'while IFS= read -r line; do
    echo "run_job $line"
done < <(tail -n "+2" ${PHENOPLIER_PROJECTS_ASTHMA_COPD_DATA_DIR}/traits_info.csv) | parallel -k --lb --halt 2 -j${PHENOPLIER_GENERAL_N_JOBS}'
```


# QQ plots

Notebook `15-twas-qqplot.ipynb` checks that the distribution of pvalues is as expected.
