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
    --gwas-file ${INPUT_GWAS_FILEPATH} \
    --phenotype-name "${INPUT_FILENAME}" \
    --tissue "${tissue}" \
    --output-dir ${OUTPUT_DIR}
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






Here we need to use some templating, because we run across random phenotypes and tissues.

```bash
mkdir -p _tmp/spredixcan

# Iterate over all random phenotype ids, chromosomes and batch ids and submit a job for each combination.
# IMPORTANT: These are a lot of tasks. You might want to split jobs by chaning the range in first for line:
#   0..199
#   200..399
#   400..599
#   600..799
#   800..999
for pheno_id in {0..999}; do
  for tissue in ${PHENOPLIER_PHENOMEXCAN_PREDICTION_MODELS_MASHR_TISSUES}; do
    export pheno_id tissue
    cat cluster_jobs/01_spredixcan_job-template.sh | envsubst '${pheno_id} ${tissue}' | bsub
  done
done
```

<!-- #region  -->
The `check_jobs.sh` script could be used also to quickly assess which jobs failed (given theirs logs):
* Check whether jobs finished successfully:
<!-- #endregion -->
```bash
bash check_job.sh -i ${PHENOPLIER_RESULTS_GLS_NULL_SIMS}/twas/spredixcan -p "INFO - Sucessfully processed metaxcan association"
```

<!-- #region  -->
* Check that at least 90% of SNPs in models were used:
<!-- #endregion -->
```bash
bash check_job.sh -i ${PHENOPLIER_RESULTS_GLS_NULL_SIMS}/twas/spredixcan -p "INFO - 90 % of model's snps found"

# success output:
# Finished checking [NUMBER_OF_PHENOTYPES * 49 tissues] logs:
#  All jobs finished successfully
```

If any job failed, check `../10_gwas_harmonization/README.md`, which has python code to get a list of unfinished jobs.








## S-MultiXcan

```bash
mkdir -p _tmp/smultixcan
cat cluster_jobs/05_smultixcan_job.sh | bsub
```

<!-- #region  -->
The `check_jobs.sh` script could be used also to quickly assess which jobs failed (given theirs logs):
<!-- #endregion -->
```bash
bash check_job.sh \
  -i ${PHENOPLIER_RESULTS_GLS_NULL_SIMS}/twas/smultixcan \
  -p "INFO - Ran multi tissue"
```

<!-- #region  -->
Another check is to count how many S-PrediXcan files were processed for each random phenotype.
It should be one per tissue (49):
<!-- #endregion -->
```bash
# S-PrediXcan files
bash check_job.sh \
  -i ${PHENOPLIER_RESULTS_GLS_NULL_SIMS}/twas/smultixcan \
  -p "Level 9 - Loading metaxcan " \
  -c 49

# Tissues loaded
bash check_job.sh \
  -i ${PHENOPLIER_RESULTS_GLS_NULL_SIMS}/twas/smultixcan \
  -p "Level 9 - Processing " \
  -c 49

# which should output:
# Finished checking [NUMBER_OF_PHENOTYPES] logs:
#  All jobs finished successfully
```


<!-- #region  -->
## Monitoring jobs

Check jobs with command `bjobs`.
Or, for a constantly-updated monitoring (refreshing every 2 seconds):
<!-- #endregion -->
```bash
watch -n 2 bjobs
```

<!-- #region  -->
Logs for `random_pheno0` are in `random_pheno1.*` (indexes are different because LPC arrays cannot start with zero).

To kill running jobs:
<!-- #endregion -->
```bash
bjobs | grep RUN | cut -d ' ' -f1 | xargs -I {} bkill {}
```


# QQ plots

Notebook `15-twas-qqplot.ipynb` checks that the distribution of pvalues is as expected.




REMEMBER TO RUN QQPLOTS NOTEBOOKS WHEN ALL IS DONE
