# Overview

This folder has the scripts to run the harmonization and imputation process across all GWAS on randomly generated phenotypes (`../05_gwas`).
It uses a standard pipeline for this task: https://github.com/hakyimlab/summary-gwas-imputation 


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
    download_eur_ld_regions \
    download_setup_summary_gwas_imputation
```


# Run cluster jobs

The `cluster_jobs/` folder has the job scripts to run on Penn's LPC cluster.
To run the jobs in order, you need to execute the command below.
The `_tmp` folder stores logs and needs to be created.


## Harmonization
```bash
mkdir -p _tmp/harmonization
cat cluster_jobs/01_harmonization_job.sh | bsub
```

The `check_jobs.sh` script could be used also to quickly assess which jobs failed (given theirs logs):
```bash
bash check_job.sh -i _tmp/harmonization/
```

There should be 1000 files in the output directory: 1000 random phenotypes.

## Imputation

Here we need to use some templating, because we impute across random phenotypes, chromosomes and batch ids.

```bash
mkdir -p _tmp/imputation

# iterate over all random phenotype ids, chromosomes and
# batch ids and submit a job for each combination
for pheno_id in {0..99}; do
  for chromosome in {1..22}; do
    for batch_id in {0..9}; do
      export pheno_id chromosome batch_id
      cat cluster_jobs/05_imputation_job-template.sh | envsubst '${pheno_id} ${chromosome} ${batch_id}' | bsub
    done
  done
done
```

Check logs with: `bash check_job.sh -i _tmp/imputation/`

There should be 220,000 files in the output directory: 22 chromosomes * 10 batches * 1000 random phenotypes.
If there are less than that number, some jobs might have failed.
To see which ones failed and run them again, you can use the following python code:

```python
import os
import re
from pathlib import Path

IMPUTATION_OUTPUT_DIR = Path(
  os.environ["PHENOPLIER_RESULTS_GLS_NULL_SIMS"],
  "imputed_gwas"
).resolve()
assert IMPUTATION_OUTPUT_DIR.exists(), IMPUTATION_OUTPUT_DIR

output_files = list(f.name for f in IMPUTATION_OUTPUT_DIR.glob("*.txt"))
len(output_files)

# expected list of files
OUTPUT_FILE_TEMPLATE = "random.pheno{pheno_id}.glm-harmonized-imputed-chr{chromosome}-batch{batch_id}_{n_batches}.txt"

expected_output_files = [
  OUTPUT_FILE_TEMPLATE.format(pheno_id=p, chromosome=c, batch_id=bi, n_batches=10)
  for p in range(0,100)
  for c in range(1, 23)
  for bi in range(0, 10)
]
assert len(expected_output_files) == 100 * 10 * 22

# get files that are expected but not there
missing_files = set(expected_output_files).difference(set(output_files))

# extract pheno id, chromosome and batch id from missing files
pheno_pattern = re.compile(r"random.pheno(?P<pheno_id>[0-9]+).glm-harmonized-imputed-chr(?P<chromosome>[0-9]+)-batch(?P<batch_id>[0-9]+)_[0-9]+.txt")
missing_jobs = [pheno_pattern.search(mf).groups() for mf in missing_files]
assert len(missing_jobs) == len(missing_files)

# these are the pheno id, chromosome and batch id combinations that are missing
missing_jobs[:10]

# submit those missing jobs
for pheno_id, chromosome, batch_id in missing_jobs:
  os.system(
    f"export pheno_id={pheno_id} chromosome={chromosome} batch_id={batch_id}; " +
    "cat cluster_jobs/05_imputation_job-template.sh | envsubst '${pheno_id} ${chromosome} ${batch_id}' | bsub"
  )
```

If jobs keep failing, inspect the logs.
A reason is that the maximum time limit set in the job template is too low, exclusively in HLA regions (chromosome 6, batch id 2).
Try to increment the maximum time limit for the job (in the job template file) and then run the jobs again using the code above.

## Post-processing

```bash
mkdir -p _tmp/postprocessing
cat cluster_jobs/10_postprocessing_job.sh | bsub
```

Check logs with:
```bash
bash check_job.sh -i _tmp/postprocessing
```

Another check is to count how many parts were processed for each random phenotype.
It should be 22 chromosomes times 10 batches (220):
```bash
bash check_job.sh \
  -i _tmp/postprocessing/ \
  -p "INFO - Processing imputed random" \
  -c 220

# which should output:
# Finished checking [NUMBER_OF_PHENOTYPES] logs:
#  All jobs finished successfully
```


## Monitoring jobs

Check jobs with command `bjobs`.
Or, for a constantly-updated monitoring (refreshing every 2 seconds):
```bash
watch -n 2 bjobs -WL
```

Logs for `random_pheno0` are in `random_pheno1.*` (indexes are different because LPC arrays cannot start with zero).


# Manhattan and QQ plots

Notebook `15-gwas-qqplot.ipynb` checks that the distribution of pvalues is as expected.
