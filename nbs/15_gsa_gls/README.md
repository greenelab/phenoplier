# Gene expression correlations

Three notebooks need to be run to compute correlations of gene predicted expression:
1. `05-snps_into_chr_cov.ipynb`
1. `10-gene_expr_correlations.ipynb`
1. `15-preprocess_gene_expr_correlations.ipynb`

For all of them, it's necessary to specify the prediction models of gene expression (see the referenced notebook above to see the accepted values for parameter `EQTL_MODEL`). For `05-snps_into_chr_cov.ipynb`, it is necessary to specify a reference panel (like 1000G or GTEx) to compute the SNP covariance matrix.


## Create output directories
The three previously mentioned notebooks are run and the "output notebook" is written to a reference panel and prediction model-specific folder.

For instance, if **GTEx v8** is the reference panel (`GTEX_V8`) for the SNP covariance matrix, create the folders with the following commands:

mashr-based prediction models:
```bash
mkdir -p nbs/15_gsa_gls/gene_corrs/gtex_v8/mashr
```


Elastic net-based models:
```bash
mkdir -p nbs/15_gsa_gls/gene_corrs/gtex_v8/elastic_net
```

For **1000 Genomes** (`1000G`):
mashr-based prediction models:
```bash
mkdir -p nbs/15_gsa_gls/gene_corrs/1000g/mashr
```


Elastic net-based models:
```bash
mkdir -p nbs/15_gsa_gls/gene_corrs/1000g/elastic_net
```


## `05-snps_into_chr_cov.ipynb`

Given a reference panel (in this example it's GTEx v8), this notebook computes the covariance for each chromosome of all variants present in prediction models.

Examples for two predictions models (mashr and elastic net):

mashr-based prediction models:
```bash
DEFINE BASH FUNCTION HERE TOO

bash nbs/run_nbs.sh \
  nbs/15_gsa_gls/05-snps_into_chr_cov.ipynb \
  gene_corrs/gtex_v8/mashr/05-snps_into_chr_cov.ipynb \
  -p REFERENCE_PANEL GTEX_V8 \
  -p EQTL_MODEL MASHR
```

Elastic net-based models:
```bash
bash nbs/run_nbs.sh \
  nbs/15_gsa_gls/05-snps_into_chr_cov.ipynb \
  gene_corrs/gtex_v8/elastic_net/05-snps_into_chr_cov.ipynb \
  -p REFERENCE_PANEL GTEX_V8 \
  -p EQTL_MODEL ELASTIC_NET
```


## `10-gene_expr_correlations.ipynb`

The notebook `10-gene_expr_correlations.ipynb` allows to compute correlations among predicted gene expression.

These commands allow to run this notebook per chromosome in parallel.
Adjust the parameter `-jX` with X as the number of cores to use (if you find a memory error, then try to lower the number of cores used to avoid allocating too much memory in parallel; this is specially true for elastic net models, which has many variants per chromosome).
You can also change the prediction models used.

For example, for mashr models you can use this command:

```bash
THIS WAS NOT TESTED YET:

compute_correlations() {
  bash nbs/run_nbs.sh \
    nbs/15_gsa_gls/10-gene_expr_correlations.ipynb \
    gene_corrs/gtex_v8/mashr/10-gene_expr_correlations-chr{}.run.ipynb \
    -p chromosome {} \
    -p REFERENCE_PANEL GTEX_V8 \
    -p EQTL_MODEL MASHR
}
export -f compute_correlations

# for GTEX_V8 and MASHR
parallel \
  -k --lb --halt 2 -j3 \
  'compute_correlations {} GTEX_V8 MASHR' \
  ::: {1..22}
```


## `15-preprocess_gene_expr_correlations.ipynb`

**FIXME:** all this needs corrections


After computing the correlations, you also need to preprocess the results to generate a single correlations file.
These are the commands for both prediction models:

## Mean-based
mashr:

```bash
bash nbs/run_nbs.sh \
  nbs/15_gsa_gls/15-preprocess_gene_expr_correlations-mean.ipynb \
  gene_corrs/gtex_v8/mashr/15-preprocess_gene_expr_correlations-mean.ipynb \
  -p REFERENCE_PANEL GTEX_V8 \
  -p EQTL_MODEL MASHR
```


elastic net:

```bash
bash nbs/run_nbs.sh \
  nbs/15_gsa_gls/15-preprocess_gene_expr_correlations.ipynb \
  gene_corrs/gtex_v8/elastic_net/15-preprocess_gene_expr_correlations-mean.ipynb \
  -p REFERENCE_PANEL GTEX_V8 \
  -p EQTL_MODEL ELASTIC_NET
```


## Max-based

```bash
bash nbs/run_nbs.sh \
  nbs/15_gsa_gls/16-preprocess_gene_expr_correlations-max.ipynb \
  gene_corrs/gtex_v8/mashr/16-preprocess_gene_expr_correlations-max.ipynb \
  -p REFERENCE_PANEL GTEX_V8 \
  -p EQTL_MODEL MASHR
```


elastic net:

```bash
bash nbs/run_nbs.sh \
  nbs/15_gsa_gls/16-preprocess_gene_expr_correlations-max.ipynb \
  gene_corrs/gtex_v8/elastic_net/16-preprocess_gene_expr_correlations-max.ipynb \
  -p REFERENCE_PANEL GTEX_V8 \
  -p EQTL_MODEL ELASTIC_NET
```