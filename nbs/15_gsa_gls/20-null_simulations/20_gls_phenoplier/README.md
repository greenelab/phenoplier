# Overview

This folder has the scripts to run GLS PhenoPLIER (associations between LVs/gene modules and traits on randomly generated phenotypes (`../15_spredixcan`).


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
    download_phenomexcan_rapid_gwas_pheno_info \
    download_phenomexcan_rapid_gwas_data_dict_file \
    download_uk_biobank_coding_3 \
    download_uk_biobank_coding_6 \
    download_phenomexcan_gtex_gwas_pheno_info \
    download_gene_map_id_to_name \
    download_gene_map_name_to_id \
    download_biomart_genes_hg38 \
    download_multiplier_model_z_pkl
```


# Run cluster jobs

The `cluster_jobs/` folder has the job scripts to run on Penn's LPC cluster.
To run the jobs in order, you need to execute the command below.
The `_tmp` folder stores logs and needs to be created.

## GLS PhenoPLIER

Here we need to use some templating, because we run across random phenotypes and batches.

```bash
mkdir -p _tmp/gls_phenoplier

# iterate over all random phenotype ids and batches
# and submit a job for each combination
export batch_n_splits=10

for pheno_id in {0..99}; do
  for ((batch_id=1; batch_id<=${batch_n_splits}; batch_id++)); do
    export pheno_id batch_id
    cat cluster_jobs/01_gls_phenoplier_mean_job-template.sh | envsubst '${pheno_id} ${batch_id} ${batch_n_splits}' | bsub
  done
done
```
The `check_jobs.sh` script could be used also to quickly assess which jobs failed (given theirs logs):
* Check whether jobs finished successfully:
```bash
bash check_job.sh -i _tmp/gls_phenoplier/ -p "INFO: Writing results to" -f '*.error'

# A success output would look like this:

Finished checking 1000 logs:
  All jobs finished successfully
```

There should be 1000 files (100 random phenotypes and 10 batch splits) in the output directory.

If any job failed, check `../10_gwas_harmonization/README.md`, which has python code to get a list of unfinished jobs.
It will need to be adapted for these tasks.

## Combine batches

```python
import os
import itertools
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(os.environ["PHENOPLIER_RESULTS_GLS_NULL_SIMS"]) / "phenoplier/gls/"
all_results_files = sorted(list(f.name for f in RESULTS_DIR.glob("*.tsv.gz")))

all_dfs = []
for key, group in itertools.groupby(all_results_files, lambda x: x.split("-")[0]):
    all_dfs = [pd.read_csv(RESULTS_DIR / gf, sep="\t", index_col="lv") for gf in group]
    all_dfs = pd.concat(all_dfs, axis=0)
    assert all_dfs.shape == (987, 2)
    all_dfs = all_dfs.sort_index()
    all_dfs.to_csv(RESULTS_DIR / f"{key}-gls_phenoplier.tsv.gz", sep="\t")
```


## Monitoring jobs

Check jobs with command `bjobs`.
Or, for a constantly-updated monitoring (refreshing every 2 seconds):
```bash
watch -n 2 bjobs
```

# QQ plots

Notebook `05-twas-qqplot.ipynb` checks that the distribution of pvalues is as expected.

