# Projection of PhenomeXcan results into MultiPLIER space

These set of notebooks project the PhenomeXcan data (MultiXcan results) into the
MultiPLIER latent space. MultiPLIER provides a Z matrix with gene loadings (genes x
latent variables) that, as a transfer learning framework, is used to project smaller
gene expression datasets into the MultiPLIER latent space.

However, what we do here is to project TWAS (Transcriptome-wide association studies)
results into this latent space. A TWAS provides the association (p-value) and its effect
size between a gene and a trait, so we basically have a gene x trait matrix with
p-values (here we project the standardized effect sizes, i.e. z-scores: the effect size
divided by the standard error of it).

The main outcome is another matrix T (latent variables x traits), which can be used to
cluster traits, see which latent variables (representing a gene module) are associated
with a particular category of traits/disease (like respiratory traits), etc.

This folder contains different notebooks to make different types of projections. For
example, by taking just a subset of the gene loadings (for instance, those at the top
1%). Thus, `p1` means that the top 1% of gene loadings are used, and `pALL` means that
all non-zero gene loadings are used.
