# Gene expression correlations

Two notebooks need to be run to compute correlations of gene predicted expression:
1. `05-snps_into_chr_cov.ipynb`
1. `10-gene_expr_correlations.ipynb`
1. `15-preprocess_gene_expr_correlations.ipynb`

For all of them, it's necessary to specify the prediction models of gene expression (see the referenced notebook above to see the accepted values for parameter `EQTL_MODEL`).

## Create output directories
The three previously mentioned notebooks are run and the "output notebook" is written a prediction model-specific folder.
To create these folders, run:

mashr:
```bash
mkdir -p nbs/15_gsa_gls/mashr_gene_corrs
rm -f nbs/15_gsa_gls/en_gene_corrs/*
```

elastic net:
```bash
mkdir -p nbs/15_gsa_gls/en_gene_corrs
rm -f nbs/15_gsa_gls/en_gene_corrs/*
```

The `rm` command is optional, is to make sure you are not mixing different output notebooks from different steps.

## `05-snps_into_chr_cov.ipynb`

This notebook computes the covariance for each chromosome of all variants present in prediction models.

Examples for two predictions models (mashr and elastic net):

mashr:

```bash
bash nbs/run_nbs.sh nbs/15_gsa_gls/05-snps_into_chr_cov.ipynb mashr_gene_corrs/05-snps_into_chr_cov.ipynb -p EQTL_MODEL MASHR
```

elastic net:

```bash
bash nbs/run_nbs.sh nbs/15_gsa_gls/05-snps_into_chr_cov.ipynb en_gene_corrs/05-snps_into_chr_cov.ipynb -p EQTL_MODEL ELASTIC_NET
```

## `10-gene_expr_correlations.ipynb`

The notebook `10-gene_expr_correlations.ipynb` allows to compute correlations among predicted gene expression.

These commands allow to run this notebook per chromosome in parallel.
Adjust the parameter `-jX` with X as the number of cores to use (if you find a memory error, then try to lower the number of cores used to avoid allocating too much memory in parallel; this is specially true for elastic net models, which has many variants per chromosome).
You can also change the prediction models used.

For example, for mashr models you can use this command:

```bash
parallel -k --lb --halt 2 -j3 'bash nbs/run_nbs.sh nbs/15_gsa_gls/10-gene_expr_correlations.ipynb mashr_gene_corrs/10-gene_expr_correlations-chr{}.run.ipynb -p chromosome {} -p EQTL_MODEL MASHR' ::: {1..22}
```


And for elastic net models, you can use this one:

```bash
parallel -k --lb --halt 2 -j3 'bash nbs/run_nbs.sh nbs/15_gsa_gls/10-gene_expr_correlations.ipynb en_gene_corrs/10-gene_expr_correlations-chr{}.run.ipynb -p chromosome {} -p EQTL_MODEL ELASTIC_NET' ::: {1..22}
```

## `15-preprocess_gene_expr_correlations.ipynb`

After computing the correlations, you need also to preprocess the results to generate a single correlations file.
If you want to run it via command line, these are the commands for both prediction models:

```bash
# mashr models
bash nbs/run_nbs.sh nbs/15_gsa_gls/15-preprocess_gene_expr_correlations.ipynb mashr_gene_corrs/15-preprocess_gene_expr_correlations.ipynb -p EQTL_MODEL MASHR
```

```bash
# elastic net models
bash nbs/run_nbs.sh nbs/15_gsa_gls/15-preprocess_gene_expr_correlations.ipynb en_gene_corrs/15-preprocess_gene_expr_correlations.ipynb -p EQTL_MODEL ELASTIC_NET
```
