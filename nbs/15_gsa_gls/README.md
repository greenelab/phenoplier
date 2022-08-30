# Generalized Least Squares model (LV-trait associations)

TODO: complete
- first, we need to compute gene expression correlations
- then run gls_cli on PhenomeXcan traits

## Overview of gene expression correlations
Three notebooks need to be run to compute correlations of gene predicted expression:
1. `05-snps_into_chr_cov.ipynb`
1. `07-compile_gwas_snps_and_twas_genes.ipynb`
1. `10-gene_expr_correlations.ipynb`
1. `15-preprocess_gene_expr_correlations.ipynb`
1. `16-create_within_distance_matrices.ipynb`

For all of them, it's necessary to specify the prediction models of gene expression (see the referenced notebook above to see the accepted values for parameter `EQTL_MODEL`). For `05-snps_into_chr_cov.ipynb`, it is necessary to specify a reference panel (like 1000G or GTEx) to compute the SNP covariance matrix.

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


## `05-snps_into_chr_cov.ipynb`

TODO: update to new commands
is maybe convenient to run this one on the desktop?

Given a reference panel (in this example it's GTEx v8), this notebook computes the covariance for each chromosome of all variants present in prediction models.

```bash
compute_snps_cov () {
  ref_panel="$1"
  eqtl_models="$2"
  
  cd ${PHENOPLIER_CODE_DIR}
  
  notebook_output_folder="gene_corrs/reference_panels/${ref_panel,,}/${eqtl_models,,}"
  full_notebook_output_folder="nbs/15_gsa_gls/${notebook_output_folder}"
  mkdir -p $full_notebook_output_folder
  
  bash nbs/run_nbs.sh \
    nbs/15_gsa_gls/05-snps_into_chr_cov.ipynb \
    ${notebook_output_folder}/05-snps_into_chr_cov.run.ipynb \
    -p REFERENCE_PANEL $ref_panel \
    -p EQTL_MODEL $eqtl_models
}

# (optional) export function definition so it's included in the Docker container
export PHENOPLIER_BASH_FUNCTIONS_CODE="$(declare -f compute_snps_cov)"

compute_snps_cov 1000G MASHR
compute_snps_cov 1000G ELASTIC_NET
compute_snps_cov GTEX_V8 MASHR
compute_snps_cov GTEX_V8 ELASTIC_NET
```


## `07-compile_gwas_snps_and_twas_genes.ipynb`

Collects informations about GWAS variants and TWAS results for a particular cohort (group of GWAS with the same number of variants).

```bash
run_job () {
  cluster_job_file="$1"
  export cohort_name_param="$2"
  export gwas_file_param="$3"
  export spredixcan_folder_param="$4"
  export spredixcan_file_pattern_param="$5"
  export smultixcan_file_param="$6"
  export ref_panel_param="$7"
  export eqtl_model_param="$8"
  
  mkdir -p _tmp/compile_gwas_twas
  
  cat $cluster_job_file | envsubst '${cohort_name_param} ${gwas_file_param} ${spredixcan_folder_param} ${spredixcan_file_pattern_param} ${} ${smultixcan_file_param} ${ref_panel_param} ${eqtl_model_param}' | ${PHENOPLIER_JOBS_EXECUTOR}
}

export -f run_job

# (optional) export function definition so it's included in the Docker container
export PHENOPLIER_BASH_FUNCTIONS_CODE="$(declare -f run_job)"

#
# For null simulations
#

# GTEx v8 models and 1000G GWAS on random phenotypes
# GWAS/TWAS results need to be already generated (see `20-null_simulations`)
run_job \
    nbs/15_gsa_gls/cluster_jobs/07_gls-compile_gwas_snps_and_twas_genes-template.sh \
    1000g_eur \
    ${PHENOPLIER_RESULTS_GLS_NULL_SIMS}/final_imputed_gwas/random.pheno0.glm-imputed.txt.gz \
    ${PHENOPLIER_RESULTS_GLS_NULL_SIMS}/twas/spredixcan/ \
    random.pheno0-gtex_v8-mashr-{tissue}.csv \
    ${PHENOPLIER_RESULTS_GLS_NULL_SIMS}/twas/smultixcan/random.pheno0-gtex_v8-mashr-smultixcan.txt \
    GTEX_V8 \
    MASHR

#
# For real phenotypes
#

# (PhenomeXcan) GTEx v8 models and Rapid GWAS (one phenotype from the group is selected)
run_job \
    nbs/15_gsa_gls/cluster_jobs/07_gls-compile_gwas_snps_and_twas_genes-template.sh \
    phenomexcan_rapid_gwas \
    ${PHENOPLIER_PHENOMEXCAN_BASE_DIR}/gwas_parsing/full/22617_7112.txt.gz \
    ${PHENOPLIER_PHENOMEXCAN_BASE_DIR}/spredixcan/rapid_gwas_project/22617_7112/ \
    22617_7112-gtex_v8-{tissue}-2018_10.csv \
    ${PHENOPLIER_PHENOMEXCAN_BASE_DIR}/smultixcan/rapid_gwas_project/smultixcan_22617_7112_ccn30.tsv.gz \
    GTEX_V8 \
    MASHR

# (PhenomeXcan) GTEx v8 models and Astle GWASs
run_job \
    nbs/15_gsa_gls/cluster_jobs/07_gls-compile_gwas_snps_and_twas_genes-template.sh \
    phenomexcan_astle \
    ${PHENOPLIER_PHENOMEXCAN_BASE_DIR}/gwas_parsing/full/Astle_et_al_2016_Eosinophil_counts.txt.gz \
    ${PHENOPLIER_PHENOMEXCAN_BASE_DIR}/spredixcan/gtex_gwas/Astle_et_al_2016_Eosinophil_counts/ \
    spredixcan_igwas_gtexmashrv8_Astle_et_al_2016_Eosinophil_counts__PM__{tissue}.csv \
    ${PHENOPLIER_PHENOMEXCAN_BASE_DIR}/smultixcan/gtex_gwas/Astle_et_al_2016_Eosinophil_counts_smultixcan_imputed_gwas_gtexv8mashr_ccn30.txt.gz \
    GTEX_V8 \
    MASHR

# (PhenomeXcan) GTEx v8 models and other GWASs
run_job \
    nbs/15_gsa_gls/cluster_jobs/07_gls-compile_gwas_snps_and_twas_genes-template.sh \
    phenomexcan_other \
    ${PHENOPLIER_PHENOMEXCAN_BASE_DIR}/gwas_parsing/full/MAGNETIC_IDL.TG.txt.gz \
    ${PHENOPLIER_PHENOMEXCAN_BASE_DIR}/spredixcan/gtex_gwas/MAGNETIC_IDL.TG/ \
    spredixcan_igwas_gtexmashrv8_MAGNETIC_IDL.TG__PM__{tissue}.csv \
    ${PHENOPLIER_PHENOMEXCAN_BASE_DIR}/smultixcan/gtex_gwas/MAGNETIC_IDL.TG_smultixcan_imputed_gwas_gtexv8mashr_ccn30.txt.gz \
    GTEX_V8 \
    MASHR
```

Command for checking results:

```bash
bash scripts/check_job.sh \
    -i _tmp/compile_gwas_twas \
    -f '*.error' \
    -p "\[NbConvertApp\] Converting notebook"
```


## `10-gene_expr_correlations.ipynb`

Computes correlations among predicted gene expression.

These commands allow to run this notebook per chromosome in parallel (in a single desktop computer) or in several nodes (in a cluster).
Adjust the parameter `-jX` with X as the number of cores to use.
If you find a memory error, then try to lower the number of cores used to avoid allocating too much memory in parallel; this is specially true for elastic net models, which have many variants per chromosome.


```bash
run_job () {
  cluster_job_file="$1"
  export cohort_name_param="$2"
  export ref_panel_param="$3"
  export eqtl_model_param="$4"
  export chr_param="$5"
  
  mkdir -p _tmp/gene_corrs
  
  cat $cluster_job_file | envsubst '${cohort_name_param} ${ref_panel_param} ${eqtl_model_param} ${chr_param}' | ${PHENOPLIER_JOBS_EXECUTOR}
}

export -f run_job

# (optional) export function definition so it's included in the Docker container
export PHENOPLIER_BASH_FUNCTIONS_CODE="$(declare -f run_job)"


#
# For null simulations
#

# GTEx v8 models and 1000G GWAS on random phenotypes
parallel -j10 \
    run_job \
    nbs/15_gsa_gls/cluster_jobs/10_gls-gene_corrs-template.sh \
    1000g_eur \
    GTEX_V8 \
    MASHR \
    '{}' ::: {1..22}

#
# For real phenotypes
#

# (PhenomeXcan) GTEx v8 models and Rapid GWAS
parallel -j10 \
    run_job \
    nbs/15_gsa_gls/cluster_jobs/10_gls-gene_corrs-template.sh \
    phenomexcan_rapid_gwas \
    GTEX_V8 \
    MASHR \
    '{}' ::: {1..22}

# (PhenomeXcan) GTEx v8 models and Astle GWASs
parallel -j10 \
    run_job \
    nbs/15_gsa_gls/cluster_jobs/10_gls-gene_corrs-template.sh \
    phenomexcan_astle \
    GTEX_V8 \
    MASHR \
    '{}' ::: {1..22}

# (PhenomeXcan) GTEx v8 models and other GWASs
parallel -j10 \
    run_job \
    nbs/15_gsa_gls/cluster_jobs/10_gls-gene_corrs-template.sh \
    phenomexcan_other \
    GTEX_V8 \
    MASHR \
    '{}' ::: {1..22}
```

Command for checking:

```bash
bash scripts/check_job.sh \
    -i _tmp/gene_corrs \
    -f '*.error' \
    -p "\[NbConvertApp\] Converting notebook"
```


## `15-postprocess_gene_expr_correlations.ipynb`

```bash
run_job () {
  cluster_job_file="$1"
  export cohort_name_param="$2"
  export ref_panel_param="$3"
  export eqtl_model_param="$4"
  
  mkdir -p _tmp/post_gene_corrs
  
  cat $cluster_job_file | envsubst '${cohort_name_param} ${ref_panel_param} ${eqtl_model_param}' | ${PHENOPLIER_JOBS_EXECUTOR}
}

export -f run_job

# (optional) export function definition so it's included in the Docker container
export PHENOPLIER_BASH_FUNCTIONS_CODE="$(declare -f run_job)"

#
# For null simulations
#

# GTEx v8 models and 1000G GWAS on random phenotypes
run_job \
    nbs/15_gsa_gls/cluster_jobs/15-postprocess_gene_corrs.sh \
    1000g_eur \
    GTEX_V8 \
    MASHR

#
# For real phenotypes
#

# (PhenomeXcan) GTEx v8 models and Rapid GWAS
run_job \
    nbs/15_gsa_gls/cluster_jobs/15-postprocess_gene_corrs.sh \
    phenomexcan_rapid_gwas \
    GTEX_V8 \
    MASHR

# (PhenomeXcan) GTEx v8 models and Astle GWASs
run_job \
    nbs/15_gsa_gls/cluster_jobs/15-postprocess_gene_corrs.sh \
    phenomexcan_astle \
    GTEX_V8 \
    MASHR

# (PhenomeXcan) GTEx v8 models and other GWASs
run_job \
    nbs/15_gsa_gls/cluster_jobs/15-postprocess_gene_corrs.sh \
    phenomexcan_other \
    GTEX_V8 \
    MASHR
```

Command for checking:

```bash
bash scripts/check_job.sh \
    -i _tmp/post_gene_corrs \
    -f '*.error' \
    -p "\[NbConvertApp\] Converting notebook"
```


## `16-create_within_distance_matrices.ipynb`

```bash
run_job () {
  cluster_job_file="$1"
  export cohort_name_param="$2"
  export ref_panel_param="$3"
  export eqtl_model_param="$4"
  
  mkdir -p _tmp/create_within_dist
  
  cat $cluster_job_file | envsubst '${cohort_name_param} ${ref_panel_param} ${eqtl_model_param}' | ${PHENOPLIER_JOBS_EXECUTOR}
}

export -f run_job

# (optional) export function definition so it's included in the Docker container
export PHENOPLIER_BASH_FUNCTIONS_CODE="$(declare -f run_job)"

#
# For null simulations
#

# GTEx v8 models and 1000G GWAS on random phenotypes
run_job \
    nbs/15_gsa_gls/cluster_jobs/16-create_within_distances.sh \
    1000g_eur \
    GTEX_V8 \
    MASHR

#
# For real phenotypes
#

# (PhenomeXcan) GTEx v8 models and Rapid GWAS
run_job \
    nbs/15_gsa_gls/cluster_jobs/16-create_within_distances.sh \
    phenomexcan_rapid_gwas \
    GTEX_V8 \
    MASHR

# (PhenomeXcan) GTEx v8 models and Astle GWASs
run_job \
    nbs/15_gsa_gls/cluster_jobs/16-create_within_distances.sh \
    phenomexcan_astle \
    GTEX_V8 \
    MASHR

# (PhenomeXcan) GTEx v8 models and other GWASs
run_job \
    nbs/15_gsa_gls/cluster_jobs/16-create_within_distances.sh \
    phenomexcan_other \
    GTEX_V8 \
    MASHR
```

Command for checking:

```bash
bash scripts/check_job.sh \
    -i _tmp/post_gene_corrs \
    -f '*.error' \
    -p "\[NbConvertApp\] Converting notebook"
```



## `18-create_corr_mat_per_lv.ipynb`

```bash
run_job () {
  cluster_job_file="$1"
  export cohort_name_param="$2"
  export ref_panel_param="$3"
  export eqtl_model_param="$4"
  export lv_code_param="$5"
  export lv_perc_param="$6"
  
  mkdir -p _tmp/corr_mat_per_lv
  
  cat $cluster_job_file | envsubst '${cohort_name_param} ${ref_panel_param} ${eqtl_model_param} ${lv_code_param} ${lv_perc_param}' | ${PHENOPLIER_JOBS_EXECUTOR}
}

export -f run_job

# (optional) export function definition so it's included in the Docker container
export PHENOPLIER_BASH_FUNCTIONS_CODE="$(declare -f run_job)"

# WARNING #1: it might be better to kick off one job (like {1..1}), wait to finish so it writes
#  the metadata and gene names, and then kick off the rest (like {2..987}). The code in the
#  notebook checks if metadata/gene name file exists, but there could be a race condition
#  when running this in a cluster with a network file system.

#
# For null simulations
#

# GTEx v8 models and 1000G GWAS on random phenotypes
parallel -k --lb --halt 2 -j10 \
    run_job \
    nbs/15_gsa_gls/cluster_jobs/18-create_corr_mat_per_lv.sh \
    1000g_eur \
    GTEX_V8 \
    MASHR \
    LV{} \
    0.01 ::: {1..987}

#
# For real phenotypes
#

# (PhenomeXcan) GTEx v8 models and Rapid GWAS
parallel -k --lb --halt 2 -j10 \
    run_job \
    nbs/15_gsa_gls/cluster_jobs/18-create_corr_mat_per_lv.sh \
    phenomexcan_rapid_gwas \
    GTEX_V8 \
    MASHR \
    LV{} \
    0.01 ::: {1..987}

# (PhenomeXcan) GTEx v8 models and Astle GWASs
parallel -k --lb --halt 2 -j10 \
    run_job \
    nbs/15_gsa_gls/cluster_jobs/18-create_corr_mat_per_lv.sh \
    phenomexcan_astle \
    GTEX_V8 \
    MASHR \
    LV{} \
    0.01 ::: {1..987}

# (PhenomeXcan) GTEx v8 models and other GWASs
parallel -k --lb --halt 2 -j10 \
    run_job \
    nbs/15_gsa_gls/cluster_jobs/18-create_corr_mat_per_lv.sh \
    phenomexcan_other \
    GTEX_V8 \
    MASHR \
    LV{} \
    0.01 ::: {1..987}
```

```bash
bash scripts/check_job.sh \
    -i _tmp/corr_mat_per_lv \
    -f '*.error' \
    -p "\[NbConvertApp\] Converting notebook"
```


## GLS on PhenomeXcan

PhenomeXcan is a TWAS resource derived from 4091 publicly available GWAS.

```bash
mkdir -p _tmp/gls_phenoplier_phenomexcan
```

```python
import os
import re

import conf

smultixcan_results_dir = conf.PHENOMEXCAN["SMULTIXCAN_MASHR_RESULTS_DIR"]

# Rapid GWAS project
results_files = list(smultixcan_results_dir.rglob("*.tsv.gz"))
pheno_pattern = re.compile(r"smultixcan_(?P<pheno_code>.+)_ccn30.tsv.gz")
pheno_codes = [pheno_pattern.search(f.name).group("pheno_code") for f in results_files]
assert len(results_files) == len(pheno_codes)
for pheno_filepath, pheno_code in zip([results_files[0]], [pheno_codes[0]]):
    pheno_filepath = str(pheno_filepath)
    os.system(
        f"export pheno_filepath={pheno_filepath} pheno_code={pheno_code}; " +
        "cat cluster_jobs/01_gls_phenoplier-phenomexcan-sub_corr-template.sh | envsubst '${pheno_filepath} ${pheno_code}' | bsub"
    )


# TODO: GTEX-GWAS
results_files = list(smultixcan_results_dir.rglob("*.txt.gz"))



```