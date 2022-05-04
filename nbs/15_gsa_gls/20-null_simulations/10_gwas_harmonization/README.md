# Overview

This folder has the scripts to run the harmonization and imputation process across all GWAS on randomly generated phenotypes (`../05_gwas`).


# Load Penn's LPC-specific paths and PhenoPLIER configuration

Change paths accordingly.

```bash
# load conda environment
module load miniconda/3
conda activate ~/software/conda_envs/phenoplier_light/

# load LPC-specific paths
. ~/projects/phenoplier/scripts/pmacs_penn/env.sh

# load in bash session all PhenoPLIER environmental variables
eval `python ~/projects/phenoplier/libs/conf.py`

# make sure they were loaded correctly
# should output something like /project/...
echo $PHENOPLIER_ROOT_DIR
```


# Download the necessary data

```bash
python ~/projects/phenoplier/environment/scripts/setup_data.py \
  --actions \
    download_1000g_genotype_data \
    download_liftover_hg19tohg38_chain \
    download_setup_summary_gwas_imputation
```


# Run cluster jobs

The `cluster_jobs/` folder has the job scripts to run on Penn's LPC cluster.
To run the jobs in order, you need to execute the command below.
The `_tmp` folder stores logs and needs to be created.

```bash
# harmonization
mkdir -p _tmp/harmonization
cat cluster_jobs/01_harmonization_job.sh | bsub

# imputation
mkdir -p _tmp/imputations
cat cluster_jobs/05_imputation_job.sh | bsub

# postprocessing
mkdir -p _tmp/postprocessing
cat cluster_jobs/10_postprocessing.sh | bsub
```

Check jobs with command `bjobs`.
Or, for a constantly-updated monitoring (refreshing every 2 seconds):
```bash
watch -n 2 bjobs
```

Logs for `random_pheno0` are in `random_pheno1.*` (indexes are different because LPC arrays cannot start with zero).

