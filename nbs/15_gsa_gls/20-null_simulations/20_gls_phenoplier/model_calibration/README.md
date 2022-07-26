# Model calibration

## Some definitions

* `04-gls-explore_ols.ipynb`: this notebook summarizes the results of computing an LV-trait association using a simple OLS model (without using covariates and without using gene correlation). In other words, it assumes that gene-trait associations are independent. This serves as a sort of baseline for comparison with other models.
  * The mean type I error for this model is 0.061.

There are two subfolders here, which contain notebooks that summarizes results using different models:

* `corr_all_genes` means computing the gene correlation matrix across all genes in a chromosome (genes in different chromosomes have corelation zero)
* `corr_within_distance` means computing the gene correlation matrix across all genes in a chromosome within a given distance (5 mbps).

Each of these subfolders compare 1) a model using the entire gene correlation matrix (`full_corr`) and 2) a model using a subset of gene correlations for each LV corresponding to genes with nonzero weights (`sub_corr`).
For each of the approaches, two pairs of notebooks are provided: the first ones are run on a subset of LVs that were identified in the baseline model (OLS, see above) to have inflated p-values and another set that are well calibrated (`some_lvs`).
These are used to quickly test new models before testing on the entire set of random phenotypes.
The second pairs is simply the run on all LVs (`all_lvs`).

## `corr_all_genes` vs `corr_within_distance`

* The model performs much better using `corr_all_genes`. For example:
  * The models `corr_within_distance` with `full_corr` and `sub_corr` have a mean type I error of 0.131 and 0.067, respectively.
  * For `corr_all_genes`, these two models (`full_corr` and `sub_corr`) have 0.046 and 0.056.

Other important highlights:

* Probably it's not necessary to compute the correlation between all genes in a given chromosome, but just increase the threshold in the `corr_within_distance` approach. This would be something to test in the future.
* Another very important aspect is to compute gene correlations 1) using variants present in GWAS only, 2) and genes present in S-PrediXcan results only. This is how S-MultiXcan operates, and we need to do the same.
  * The implication of this is that a new gene correlation matrix has to be computed for each set of results.
  * If several GWAS are provided for the same cohort (such as UK Biobank), computing the gene correlations on one of them could be enough if GWASs have the same variants (at least the same variants that are predictors of gene expression).


## `full_corr` vs `sub_corr`

Here I continue the analysis using `corr_all_genes` only.

* `full_corr` seems to be better calibrated at this point (0.046 vs 0.056), slightly more conservative than `sub_corr`.
* Both qqplots seem similar, although `sub_corr` has some very small pvalues around `-log(p)=60`.

Other important highlights:

* From an implementation point of view, `full_corr` is easier, since for `sub_corr` we need to compute a matrix for each LV to make it faster.


## With covariates

Here I continue the analysis using `corr_all_genes` only.

There are some potential covariates that could confound LV-trait associations:

* Number of predictor SNPs for a gene across tissues.
* Number of principal components (PCs) for a gene in S-MultiXcan.

Below I tested some of these and others derived from them.

### Adding "effective gene size"

"Effective gene size" is the number of PCs after pruning in S-MultiXcan.

Using `full_corr` or `sub_corr`:
* Using `gene_size` as covariate alone does NOT change results.
  * Mean type I error does not change when using `gene_size` as covariate.
  * It does not correct some special LVs, like LV769
  * `full_corr` is still very conservative on LV412 (0.009 at p < 0.05) (well-calibrated with simple OLS) compared with `sub_corr` (0.051)


CHANGE BELOW IF THERE IS A DIFFERENCE BETWEEN FULL_CORR OR SUB_CORR, OTHERWISE REMOVE:

* Using `gene_size` + `gene_size_log` ADD CONCLUSION
* ADD COMPARISON WITH OLS MODEL

Using `sub_corr`:
* Using `gene_size` as covariate alone does NOT change results.

* Using `gene_size` + `gene_size_log` ADD CONCLUSION
* ADD COMPARISON WITH OLS MODEL


### Adding "gene density"

"Gene density" is the ratio of "effective gene size" and the total number of tissue results for that gene in S-PrediXcan.

Using `full_corr`:
* Using `gene_density` as covariate alone DOES NOT IMPROVE RESULTS (CHECK THIS)
* Using `gene_density` + `gene_density_log` ADD CONCLUSION
* ADD COMPARISON WITH OLS MODEL

Using `sub_corr`:
* Using `gene_density` as covariate alone DOES NOT IMPROVE RESULTS (CHECK THIS)
* Using `gene_density` + `gene_density_log` ADD CONCLUSION
* ADD COMPARISON WITH OLS MODEL
