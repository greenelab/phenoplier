# Overview

This folder has the scripts to compile GWAS/TWAS information and compute gene-gene correlations.


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
      download_gene_map_id_to_name \
      download_gene_map_name_to_id \
      download_biomart_genes_hg38 \
      download_multiplier_model_z_pkl \
      download_snps_covariance_gtex_mashr
```


# Run

## Compile GWAS/TWAS information

```bash
run_job () {
  # read trait information
  IFS=',' read -r id desc file sample_size n_cases <<< "$1"
  
  export CODE_RELATIVE_DIR="projects/asthma-copd/nbs/20_gene_corrs"
  export CODE_DIR=${PHENOPLIER_CODE_DIR}/${CODE_RELATIVE_DIR}
  
  export GWAS_DIR=${PHENOPLIER_PROJECTS_ASTHMA_COPD_RESULTS_DIR}/final_imputed_gwas
  export INPUT_FILENAME=${file%.*}
  export SPREDIXCAN_DIR=${PHENOPLIER_PROJECTS_ASTHMA_COPD_RESULTS_DIR}/twas/spredixcan
  export SMULTIXCAN_DIR=${PHENOPLIER_PROJECTS_ASTHMA_COPD_RESULTS_DIR}/twas/smultixcan
  export OUTPUT_DIR=${PHENOPLIER_PROJECTS_ASTHMA_COPD_RESULTS_DIR}/gls_phenoplier
  
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
  
  cohort_name="$id"
  gwas_file="$INPUT_GWAS_FILEPATH"
  spredixcan_folder="$SPREDIXCAN_DIR"
  spredixcan_file_pattern="${INPUT_FILENAME}-gtex_v8-mashr-{tissue}.csv"
  smultixcan_file="${SMULTIXCAN_DIR}/${INPUT_FILENAME}-gtex_v8-mashr-smultixcan.txt"

  cd ${PHENOPLIER_CODE_DIR}

  notebook_output_folder="gene_corrs/${cohort_name,,}"
  full_notebook_output_folder="${CODE_RELATIVE_DIR}/${notebook_output_folder}"
  mkdir -p $full_notebook_output_folder

  bash nbs/run_nbs.sh \
    "${CODE_RELATIVE_DIR}/07-compile_gwas_snps_and_twas_genes.ipynb" \
    "${notebook_output_folder}/07-compile_gwas_snps_and_twas_genes.run.ipynb" \
    -p COHORT_NAME "$cohort_name" \
    -p GWAS_FILE "$gwas_file" \
    -p SPREDIXCAN_FOLDER "$spredixcan_folder" \
    -p SPREDIXCAN_FILE_PATTERN "$spredixcan_file_pattern" \
    -p SMULTIXCAN_FILE "$smultixcan_file" \
    -p OUTPUT_DIR_BASE "$OUTPUT_DIR"
}

export -f run_job

# (optional) export function definition so it's included in the Docker container
export PHENOPLIER_BASH_FUNCTIONS_CODE="$(declare -f run_job)"


bash scripts/run_docker_dev.sh --docker-args "-ti" \
'while IFS= read -r line; do
    echo "run_job $line"
done < <(tail -n "+2" ${PHENOPLIER_PROJECTS_ASTHMA_COPD_DATA_DIR}/traits_info.csv) | parallel -k --lb --halt 2 -j${PHENOPLIER_GENERAL_N_JOBS}'
```

## Gene correlations

```bash
run_job () {
  # read trait information
  IFS=',' read -r id desc file sample_size n_cases chromosome <<< "$1"
  
  export CODE_RELATIVE_DIR="projects/asthma-copd/nbs/20_gene_corrs"
  export CODE_DIR=${PHENOPLIER_CODE_DIR}/${CODE_RELATIVE_DIR}
  
  export GWAS_DIR=${PHENOPLIER_PROJECTS_ASTHMA_COPD_RESULTS_DIR}/final_imputed_gwas
  export INPUT_FILENAME=${file%.*}
  export SPREDIXCAN_DIR=${PHENOPLIER_PROJECTS_ASTHMA_COPD_RESULTS_DIR}/twas/spredixcan
  export SMULTIXCAN_DIR=${PHENOPLIER_PROJECTS_ASTHMA_COPD_RESULTS_DIR}/twas/smultixcan
  export OUTPUT_DIR=${PHENOPLIER_PROJECTS_ASTHMA_COPD_RESULTS_DIR}/gls_phenoplier
  
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
  
  cohort_name="$id"
  gwas_file="$INPUT_GWAS_FILEPATH"
  spredixcan_folder="$SPREDIXCAN_DIR"
  spredixcan_file_pattern="${INPUT_FILENAME}-gtex_v8-mashr-{tissue}.csv"
  smultixcan_file="${SMULTIXCAN_DIR}/${INPUT_FILENAME}-gtex_v8-mashr-smultixcan.txt"

  cd ${PHENOPLIER_CODE_DIR}

  notebook_output_folder="gene_corrs/${cohort_name,,}"
  full_notebook_output_folder="${CODE_RELATIVE_DIR}/${notebook_output_folder}"
  mkdir -p $full_notebook_output_folder

  bash nbs/run_nbs.sh \
    "${CODE_RELATIVE_DIR}/10-gene_expr_correlations.ipynb" \
    "${notebook_output_folder}/10-gene_expr_correlations-chr${chromosome}.run.ipynb" \
    -p COHORT_NAME "$cohort_name" \
    -p CHROMOSOME "$chromosome" \
    -p OUTPUT_DIR_BASE "$OUTPUT_DIR"
}

export -f run_job

# (optional) export function definition so it's included in the Docker container
export PHENOPLIER_BASH_FUNCTIONS_CODE="$(declare -f run_job)"


bash scripts/run_docker_dev.sh --docker-args "-ti" \
'while IFS= read -r line; do
  for chromosome in {1..22}; do
    echo run_job "$line,$chromosome"
  done
done < <(tail -n "+2" ${PHENOPLIER_PROJECTS_ASTHMA_COPD_DATA_DIR}/traits_info.csv) | parallel -k --lb --halt 2 -j${PHENOPLIER_GENERAL_N_JOBS}'
```

## Post-processing

```bash
run_job () {
  # read trait information
  IFS=',' read -r id desc file sample_size n_cases <<< "$1"
  
  export CODE_RELATIVE_DIR="projects/asthma-copd/nbs/20_gene_corrs"
  export CODE_DIR=${PHENOPLIER_CODE_DIR}/${CODE_RELATIVE_DIR}
  
  export OUTPUT_DIR=${PHENOPLIER_PROJECTS_ASTHMA_COPD_RESULTS_DIR}/gls_phenoplier
  
  export NUMBA_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPEN_BLAS_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1
  export OMP_NUM_THREADS=1
  
  cohort_name="$id"

  cd ${PHENOPLIER_CODE_DIR}

  notebook_output_folder="gene_corrs/${cohort_name,,}"
  full_notebook_output_folder="${CODE_RELATIVE_DIR}/${notebook_output_folder}"
  mkdir -p $full_notebook_output_folder

  bash nbs/run_nbs.sh \
    "${CODE_RELATIVE_DIR}/15-postprocess_gene_expr_correlations.ipynb" \
    "${notebook_output_folder}/15-postprocess_gene_expr_correlations.run.ipynb" \
    -p COHORT_NAME "$cohort_name" \
    -p OUTPUT_DIR_BASE "$OUTPUT_DIR"
}

export -f run_job

# (optional) export function definition so it's included in the Docker container
export PHENOPLIER_BASH_FUNCTIONS_CODE="$(declare -f run_job)"


bash scripts/run_docker_dev.sh --docker-args "-ti" \
'while IFS= read -r line; do
  echo run_job "$line"
done < <(tail -n "+2" ${PHENOPLIER_PROJECTS_ASTHMA_COPD_DATA_DIR}/traits_info.csv) | parallel -k --lb --halt 2 -j${PHENOPLIER_GENERAL_N_JOBS}'
```

## Create LV-specific correlation matrices

```bash
run_job () {
  # read trait information
  IFS=',' read -r id desc file sample_size n_cases lv_code <<< "$1"
  
  export lv_code="LV${lv_code}"
  export lv_perc=0.01
  
  export CODE_RELATIVE_DIR="projects/asthma-copd/nbs/20_gene_corrs"
  export CODE_DIR=${PHENOPLIER_CODE_DIR}/${CODE_RELATIVE_DIR}
  
  export OUTPUT_DIR=${PHENOPLIER_PROJECTS_ASTHMA_COPD_RESULTS_DIR}/gls_phenoplier
  
  export NUMBA_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPEN_BLAS_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1
  export OMP_NUM_THREADS=1
  
  cohort_name="$id"

  cd ${PHENOPLIER_CODE_DIR}

  notebook_output_folder="gene_corrs/${cohort_name,,}/18-corr_mat_per_lv/lv_perc-${lv_perc}"
  full_notebook_output_folder="${CODE_RELATIVE_DIR}/${notebook_output_folder}"
  mkdir -p $full_notebook_output_folder
  
  bash nbs/run_nbs.sh \
    "${CODE_RELATIVE_DIR}/18-create_corr_mat_per_lv.ipynb" \
    ${notebook_output_folder}/18-create_corr_mat_per_lv-${lv_code}.run.ipynb \
    -p COHORT_NAME "$cohort_name" \
    -p LV_CODE "$lv_code" \
    -p LV_PERCENTILE "$lv_perc" \
    -p OUTPUT_DIR_BASE "$OUTPUT_DIR"
}

export -f run_job

# (optional) export function definition so it's included in the Docker container
export PHENOPLIER_BASH_FUNCTIONS_CODE="$(declare -f run_job)"


bash scripts/run_docker_dev.sh --docker-args "-ti" \
'while IFS= read -r line; do
  for lv_code in {1..987}; do
    echo run_job "$line,$lv_code"
  done
done < <(tail -n "+2" ${PHENOPLIER_PROJECTS_ASTHMA_COPD_DATA_DIR}/traits_info.csv) | parallel -k --lb --halt 2 -j${PHENOPLIER_GENERAL_N_JOBS}'
```
