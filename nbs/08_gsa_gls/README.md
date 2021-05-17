# Gene expression correlations

This commands allows to run gene predicted expression correlations per chromosome in parallel.
Adjust the parameter `-jX` with X as the number of cores to use.

```bash
parallel -k --lb --halt 2 -jX 'bash run_nbs.sh 08_gsa_gls/10-gene_expr_correlations.ipynb 10-gene_expr_correlations-chr{}.run.ipynb -p chromosome {}' ::: {1..22}
```

