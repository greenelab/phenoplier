# PhenoPLIER

![tests](https://github.com/greenelab/phenoplier/workflows/tests/badge.svg)

Transcriptome-wide association studies (TWAS) allow to compute a gene-based
association by imputing the genetically regulated gene expression and then
correlating it with the trait of interest. However, these approaches usually
look at one gene at a time, not considering that they act together to carry out
different functions. This lack of a broader view of the molecular mechanisms
involved in disease pathophysiology could harm the accurate prioritization of
causal genes, thus hampering the identification of good targets for drug
development or repurposing.

PhenoPLIER is a computational approach that integrates TWAS and pharmacological
perturbations with gene co-expression patterns (gene modules) learned from large
expression compendia across different tissues, cell types and other conditions.
PhenoPLIER aims to find novel gene module associations with complex traits, and
also new pharmarcological perturbations that can suggest alternative treatment
options when some genes are not druggable. The approach allows to infer 1) how
gene modules affect complex traits, 2) in which conditions (cell types, tissues,
etc) are those genes activated, and 3) how a gene module activity could be
modulated with different compounds.

# Setup

To prepare the environment to run the PhenoPLIER code, follow the steps in [environment](environment/).
That will create a conda environment and download the necessary data.

# Running code

Once the environment is ready, you can start your JupyterLab server by running:

```bash
bash scripts/run_nbs.sh
```

Go to `http://localhost:8892` and browse the `nbs` folder. Then run notebooks in the specified order.
