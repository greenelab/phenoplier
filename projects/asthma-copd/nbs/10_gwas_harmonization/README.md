# Overview

This folder has the scripts to run the harmonization and imputation process across all GWAS (see `../05_gwas`).
It uses a standard pipeline for this task: https://github.com/hakyimlab/summary-gwas-imputation 

# Setup

## Penn's LPC cluster

**NOT TESTED**

```bash
# load conda environment
module load miniconda/3
conda activate ~/software/conda_envs/phenoplier

# load LPC-specific paths
. ~/projects/phenoplier/scripts/pmacs_penn/env.sh

# set the executor of commands to "bsub" (to submit the jobs)
export PHENOPLIER_JOBS_EXECUTOR="bsub"

# load in bash session all PhenoPLIER environmental variables
eval `python ~/projects/phenoplier/libs/conf.py`

# make sure they were loaded correctly
# should output something like /project/...
echo $PHENOPLIER_ROOT_DIR
```

## Desktop computer

Set the executor to bash:
```bash
. projects/asthma-copd/scripts/env.sh

export PHENOPLIER_JOBS_EXECUTOR="bash"
```

For this, it's convenient to use Docker by running the specified command between single quotes:

```bash
bash scripts/run_docker_dev.sh '[COMMAND]'
```

## Download the necessary data

```bash
python environment/scripts/setup_data.py \
  --actions \
    download_1000g_genotype_data \
    download_liftover_hg19tohg38_chain \
    download_eur_ld_regions \
    download_setup_summary_gwas_imputation
```









# Run cluster jobs

The `cluster_jobs/` folder has the job scripts to run on Penn's LPC cluster or desktop computer.
To run the jobs in order, you need to execute the command below.
The `_tmp` folder stores logs and needs to be created.

```bash
# make sure we use the right number of CPU cores
export n_jobs=${PHENOPLIER_N_JOBS:-1}
export NUMBA_NUM_THREADS=${n_jobs}
export MKL_NUM_THREADS=${n_jobs}
export OPEN_BLAS_NUM_THREADS=${n_jobs}
export NUMEXPR_NUM_THREADS=${n_jobs}
export OMP_NUM_THREADS=${n_jobs}
```

```bash
# export code dir
CODE_DIR=${PHENOPLIER_CODE_DIR}/projects/asthma-copd/nbs/10_gwas_harmonization
```


## Harmonization
```bash
# create folder for logs
mkdir -p _tmp/harmonization

GWAS_DIR="${PHENOPLIER_PROJECTS_ASTHMA_COPD_DATA_DIR}/gwas"
OUTPUT_DIR="${PHENOPLIER_PROJECTS_ASTHMA_COPD_DATA_DIR}/harmonized_gwas"
LIFTOVER_CHAIN_FILE_PATH="${PHENOPLIER_GENERAL_LIFTOVER_HG19_TO_HG38}"

# Asthma-only
GWAS_FILE_NAME="GWAS_Asthma_only_GLM_SNPs_info0.7.txt"
GWAS_SAMPLES_N_CASES=19217

bash ${CODE_DIR}/01_harmonize.sh \
  --input-gwas-file ${GWAS_DIR}/${GWAS_FILE_NAME} \
  --samples-n-cases ${GWAS_SAMPLES_N_CASES} \
  --liftover-chain-file ${LIFTOVER_CHAIN_FILE_PATH} \
  --output-dir ${OUTPUT_DIR}

# COPD-only
GWAS_FILE_NAME="GWAS_COPD_only_GLM_SNPs_info0.7.txt"
GWAS_SAMPLES_N_CASES=13055

# ACOS
GWAS_FILE_NAME="GWAS_ACO_GLM_SNPs_info0.7.txt"
GWAS_SAMPLES_N_CASES=7035
```






The `check_jobs.sh` script could be used also to quickly assess which jobs failed (given theirs logs):
```bash
bash check_job.sh -i _tmp/harmonization/
```

There should be [NUMBER OF PHENOTYPES] files in the output directory: 1000 random phenotypes.
