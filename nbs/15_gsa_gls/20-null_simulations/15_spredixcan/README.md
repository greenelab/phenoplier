# Overview

This folder has the scripts to run the PrediXcan family of methods on GWAS on randomly generated phenotypes (`../10_gwas_harmonization`).


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
    download_setup_metaxcan \
    download_predixcan_mashr_prediction_models \
    download_mashr_expression_smultixcan_snp_covariance
```


# Run cluster jobs

The `cluster_jobs/` folder has the job scripts to run on Penn's LPC cluster.
To run the jobs in order, you need to execute the command below.
The `_tmp` folder stores logs and needs to be created.

## S-PrediXcan

Here we need to use some templating, because we run across random phenotypes and tissues.

```bash
mkdir -p _tmp/spredixcan

# iterate over all random phenotype ids and tissues
# and submit a job for each combination
for pheno_id in {0..99}; do
  for tissue in ${PHENOPLIER_PHENOMEXCAN_PREDICTION_MODELS_MASHR_TISSUES}; do
    export pheno_id tissue
    cat cluster_jobs/01_spredixcan_job-template.sh | envsubst '${pheno_id} ${tissue}' | bsub
  done
done
```

The `check_jobs.sh` script could be used also to quickly assess which jobs failed (given theirs logs):
* Check whether jobs finished successfully:
```bash
bash check_job.sh -i ${PHENOPLIER_RESULTS_GLS_NULL_SIMS}/twas/spredixcan -p "INFO - Sucessfully processed metaxcan association"
```

* Check that at least 90% of SNPs in models were used:
```bash
bash check_job.sh -i ${PHENOPLIER_RESULTS_GLS_NULL_SIMS}/twas/spredixcan -p "INFO - 90 % of model's snps"

# success output:
# Finished checking 4900 logs:
#  All jobs finished successfully
```

There should be 4900 files (100 random phenotypes and 49 tissues) in the output directory.

If any job failed, check `../10_gwas_harmonization/README.md`, which has python code to get a list of unfinished jobs.


## S-MultiXcan

```bash
mkdir -p _tmp/smultixcan
cat cluster_jobs/05_smultixcan_job.sh | bsub
```

The `check_jobs.sh` script could be used also to quickly assess which jobs failed (given theirs logs):
`bash check_job.sh -i ${PHENOPLIER_RESULTS_GLS_NULL_SIMS}/twas/smultixcan -p "INFO - Ran multi tissue"`

There should be 100 files in the output directory: 100 random phenotypes.


## Monitoring jobs

Check jobs with command `bjobs`.
Or, for a constantly-updated monitoring (refreshing every 2 seconds):
```bash
watch -n 2 bjobs
```

Logs for `random_pheno0` are in `random_pheno1.*` (indexes are different because LPC arrays cannot start with zero).


# QQ plots

Notebook `15-twas-qqplot.ipynb` checks that the distribution of pvalues is as expected.
