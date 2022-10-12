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

There should be [NUMBER OF PHENOTYPES] files in the output directory: 1000 random phenotypes.

## Imputation

Here we need to use some templating, because we impute across random phenotypes, chromosomes and batch ids.
Batch ids are used to split jobs more and thus parallelize across more nodes.

```bash
mkdir -p _tmp/imputation

# Iterate over all random phenotype ids, chromosomes and batch ids and submit a job for each combination.
# IMPORTANT: These are a lot of tasks. You might want to split jobs by chaning the range in first for line:
#   0..199
#   200..399
#   400..599
#   600..799
#   800..999
for pheno_id in {0..999}; do
  for chromosome in {1..22}; do
    for batch_id in {0..9}; do
      export pheno_id chromosome batch_id
      cat cluster_jobs/05_imputation_job-template.sh | envsubst '${pheno_id} ${chromosome} ${batch_id}' | bsub
    done
  done
done
```

Check logs with:
```bash
bash check_job.sh -i _tmp/imputation/
```

There should be 220,000 files in the output directory: 22 chromosomes * 10 batches * 1000 random phenotypes.
If there are less than that number, some jobs might have failed.
To see which ones failed and run them again, you can use the following python code:

```python
import os
import re
from pathlib import Path

# adjust accordingly
N_PHENOTYPES = 1000   # 1000
PHENO_ID_START = 0    # 0
PHENO_ID_END = 999    # 999

IMPUTATION_OUTPUT_DIR = Path(
  os.environ["PHENOPLIER_RESULTS_GLS_NULL_SIMS"],
  "imputed_gwas"
).resolve()
assert IMPUTATION_OUTPUT_DIR.exists(), IMPUTATION_OUTPUT_DIR

output_files = list(f.name for f in IMPUTATION_OUTPUT_DIR.glob("*.txt"))
len(output_files)

# expected list of files
OUTPUT_FILE_TEMPLATE = "random.pheno{pheno_id}.glm.linear.tsv-harmonized-imputed-chr{chromosome}-batch{batch_id}_{n_batches}.txt"

expected_output_files = [
  OUTPUT_FILE_TEMPLATE.format(pheno_id=p, chromosome=c, batch_id=bi, n_batches=10)
  for p in range(PHENO_ID_START, PHENO_ID_END+1)
  for c in range(1, 23)
  for bi in range(0, 10)
]
assert len(expected_output_files) == N_PHENOTYPES * 10 * 22

# get files that are expected but not there
missing_files = set(expected_output_files).difference(set(output_files))
len(missing_files)

# extract pheno id, chromosome and batch id from missing files
pheno_pattern = re.compile(r"random.pheno(?P<pheno_id>[0-9]+).glm.linear.tsv-harmonized-imputed-chr(?P<chromosome>[0-9]+)-batch(?P<batch_id>[0-9]+)_[0-9]+.txt")
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
It should be 22 chromosomes times 10 batches (220), see code below.
The `-p` parameter is the success pattern, a chunk of text that has to be found in the input files as certain number of times (`-c`).
```bash
bash check_job.sh \
  -i _tmp/postprocessing/ \
  -p "INFO - Processing imputed random" \
  -c 220

# which should output:
# Finished checking [NUMBER_OF_PHENOTYPES] logs:
#  All jobs finished successfully
```


## Compute set of common variants

For some reason, the postprocessing scripts previously run do not generate exactly the same set of final variants for imputed GWAS.
Each final GWAS should have the same set of imputed variants, since they were all imputed using GWAS from the same cohort and exactly the same input variants.
However, this is not the case, and for this null simulation, we need all GWAS to have the same set of variants to efficiently compute gene correlations.

So what we do is to read all final GWAS files, generate a set of common variants across all of them, and then align all final GWAS to that.


```python
import pickle
from pathlib import Path
import concurrent

import numpy as np
import pandas as pd

import conf

N_SAMPLES = 50

POST_IMPUTED_DIR = Path(
  conf.RESULTS["GLS_NULL_SIMS"],
  "post_imputed_gwas"
).resolve()
assert POST_IMPUTED_DIR.exists(), POST_IMPUTED_DIR

input_files = sorted(list(POST_IMPUTED_DIR.glob("*.txt.gz")))
len(input_files)

# sample files
np.random.seed(0)
input_files = np.random.choice(input_files, size=N_SAMPLES, replace=False)
len(input_files)

# read all GWAS and find a set of common panel_variant_id_values
def _get_gwas_variants(f):
    gwas_data = pd.read_table(f, usecols=["panel_variant_id", "zscore"])
    assert gwas_data["panel_variant_id"].is_unique
    assert gwas_data.shape == gwas_data.dropna().shape
    return f.name, set(gwas_data["panel_variant_id"])

common_variants = set()
last_n_var_ids = -1
with concurrent.futures.ProcessPoolExecutor(max_workers=conf.GENERAL["N_JOBS"]) as executor:
    for gwas_file_name, gwas_variants in executor.map(_get_gwas_variants, input_files, chunksize=10):
        if len(common_variants) == 0:
            common_variants = gwas_variants
        else:
            common_variants = common_variants.intersection(gwas_variants)
        
        n_var_ids = len(common_variants)
        same_previous = n_var_ids == last_n_var_ids
        last_n_var_ids = n_var_ids
        print(f"{gwas_file_name}, # common variants: {n_var_ids} (same? {same_previous})", flush=True)


with open(POST_IMPUTED_DIR / "common_variant_ids.pkl", 'wb') as f:
    pickle.dump(common_variants, f, protocol=pickle.HIGHEST_PROTOCOL)
```

## Save GWAS files using common variants

```bash
mkdir -p _tmp/common_var_ids
cat cluster_jobs/15-common_variant_ids_job.sh | bsub
```

Checks:
```bash
bash check_job.sh \
  -i _tmp/common_var_ids/ \
  -f '*.out' \
  -p "Filtering variants: 8325729"

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



REMEMBER TO RUN QQPLOTS NOTEBOOKS WHEN ALL IS DONE
