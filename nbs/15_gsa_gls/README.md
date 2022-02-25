# Gene expression correlations

The notebook `10-gene_expr_correlations.ipynb` allows to compute correlations among predicted gene expression.

These commands allow to run this notebook per chromosome in parallel.
Adjust the parameter `-jX` with X as the number of cores to use.
You can also change the prediction models used (see the referenced notebook above to see the accepted values for parameter `EQTL_MODEL`.

For example, for mashr models you can use this command:

```bash
mkdir -p nbs/15_gsa_gls/mashr_gene_corrs
rm -f nbs/15_gsa_gls/mashr_gene_corrs/*
parallel -k --lb --halt 2 -j3 'bash nbs/run_nbs.sh nbs/15_gsa_gls/10-gene_expr_correlations.ipynb mashr_gene_corrs/10-gene_expr_correlations-chr{}.run.ipynb -p chromosome {} -p EQTL_MODEL MASHR' ::: {1..22}
```


And for elastic net models, you can use this one:

```bash
mkdir -p nbs/15_gsa_gls/en_gene_corrs
rm -f nbs/15_gsa_gls/en_gene_corrs/*
parallel -k --lb --halt 2 -j3 'bash nbs/run_nbs.sh nbs/15_gsa_gls/10-gene_expr_correlations.ipynb en_gene_corrs/10-gene_expr_correlations-chr{}.run.ipynb -p chromosome {} -p EQTL_MODEL ELASTIC_NET' ::: {1..22}
```

After computing the correlations, you need also to preprocess the results to generate a single correlations file.
If you want to run it via command line, these are the commands for both prediction models:

```bash
# mashr models
parallel -k --lb --halt 2 -j3 'bash nbs/run_nbs.sh nbs/15_gsa_gls/15-preprocess_gene_expr_correlations.ipynb -p EQTL_MODEL MASHR'
```

```bash
# elastic net models
parallel -k --lb --halt 2 -j3 'bash nbs/run_nbs.sh nbs/15_gsa_gls/15-preprocess_gene_expr_correlations.ipynb -p EQTL_MODEL ELASTIC_NET'
```
