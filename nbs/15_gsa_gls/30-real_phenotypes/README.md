# Overview

This folder has the scripts to run GLS PhenoPLIER (associations between LVs/gene modules and traits on randomly generated phenotypes (`../15_spredixcan`).

Before running these steps, **it is necessary** to generate a correlation matrix for predicted gene expression _specific_ for these random phenotypes (see `nbs/15_gsa_gls/README.md`).


## Setup

### Penn's LPC cluster

Load Penn's LPC-specific paths and PhenoPLIER configuration.
Change paths accordingly.

```bash
# load conda environment
module load miniconda/3
conda activate ~/software/conda_envs/phenoplier/

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

### Desktop computer

Set the executor to bash:
```bash
export PHENOPLIER_JOBS_EXECUTOR="bash"
```

For this, it's convenient to use Docker by running the specified command between single quotes:

```bash
bash scripts/run_docker_dev.sh '[COMMAND]'
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
    download_multiplier_model_z_pkl \
    download_phenomexcan_unified_pheno_info \
    download_emerge_phenotypes_description
```


# Run cluster jobs

The `cluster_jobs/` folder has the job scripts to run on Penn's LPC cluster.
To run the jobs in order, you need to execute the command below.
The `_tmp` folder stores logs and needs to be created.


## Run LV-trait associations


```bash
run_job () {
  cluster_job_file="$1"
  export pheno_id="$2"
  
  mkdir -p _tmp/phenoplier/phenomexcan
  mkdir -p _tmp/phenoplier/emerge
  
  cat $cluster_job_file | envsubst '${pheno_id}' | ${PHENOPLIER_JOBS_EXECUTOR}
}

export -f run_job

# (optional) export function definition so it's included in the Docker container
export PHENOPLIER_BASH_FUNCTIONS_CODE="$(declare -f run_job)"
```


### With covariates

```bash
# recommended
rm -rf _tmp/

# phenomexcan_rapid_gwas:
zcat ${PHENOPLIER_PHENOMEXCAN_UNIFIED_PHENO_INFO_FILE} | \
  tail -n +2 | \
  awk -F'\t' '$10 == "UK Biobank"' | \
  cut -f3 | \
  parallel -j10 run_job cluster_jobs/01-phenomexcan-rapid_gwas-sub_corr-template.sh {}

# phenomexcan_astle:
zcat ${PHENOPLIER_PHENOMEXCAN_UNIFIED_PHENO_INFO_FILE} | \
  tail -n +2 | \
  awk -F'\t' '$10 == "Astle_et_al_2016"' | \
  cut -f3 | \
  parallel -j10 run_job cluster_jobs/02-phenomexcan-astle-sub_corr-template.sh {}

# phenomexcan_other:
#  there are some phenotypes in this set that need to compute the correlation matrix
zcat ${PHENOPLIER_PHENOMEXCAN_UNIFIED_PHENO_INFO_FILE} | \
  tail -n +2 | \
  awk -F'\t' '(($10 != "UK Biobank") && ($10 != "Astle_et_al_2016"))' | \
  cut -f3 | \
  parallel -j10 run_job cluster_jobs/03-phenomexcan-other-sub_corr-template.sh {}

# emerge:
#  there are some phenotypes in this set that need to compute the correlation matrix
#  this is because of the GWAS imputation, where not all variants are the same across
#  phecodes, and thus potentially not all the same genes
cat ${PHENOPLIER_EMERGE_DESC_FILE_WITH_SAMPLE_SIZE} | \
  tail -n +2 | \
  cut -f1 | \
  parallel -j10 run_job cluster_jobs/10-emerge-sub_corr-template.sh {}
```

```bash
# Adjust path (-i argument) as needed:
#  _tmp/phenoplier/phenomexcan
#  _tmp/phenoplier/emerge
bash ${PHENOPLIER_CODE_DIR}/scripts/check_job.sh \
    -i _tmp/phenoplier/emerge \
    -f '*.error' \
    -p "INFO: Writing results to"

bash ${PHENOPLIER_CODE_DIR}/scripts/check_job.sh \
    -i _tmp/phenoplier/emerge \
    -f '*.error' \
    -p "INFO: Using covariates: \['gene_size', 'gene_size_log', 'gene_density', 'gene_density_log'\]"

bash ${PHENOPLIER_CODE_DIR}/scripts/check_job.sh \
    -i _tmp/phenoplier/emerge \
    -f '*.error' \
    -p "INFO: Correlation matrix is a directory"
```


## Monitoring jobs

Check jobs with command `bjobs`.
Or, for a constantly-updated monitoring (refreshing every 2 seconds):
```bash
watch -n 2 bjobs
```

To kill running jobs:
```bash
bjobs | grep RUN | cut -d ' ' -f1 | xargs -I {} bkill {}
```

# QQ plots

Notebook `05-twas-qqplot.ipynb` checks that the distribution of pvalues is as expected.
