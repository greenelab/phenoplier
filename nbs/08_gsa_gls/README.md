# Gene expression correlations

These commands allow to run gene predicted expression correlations per chromosome in parallel.
Adjust the parameter `-jX` with X as the number of cores to use.

```bash
mkdir -p 08_gsa_gls/gene_corrs
rm -f 08_gsa_gls/gene_corrs/*
parallel -k --lb --halt 2 -jX 'bash run_nbs.sh 08_gsa_gls/10-gene_expr_correlations.ipynb gene_corrs/10-gene_expr_correlations-chr{}.run.ipynb -p chromosome {}' ::: {1..22}
```
