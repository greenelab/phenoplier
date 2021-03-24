# PhenoPLIER

![tests](https://github.com/greenelab/phenoplier/workflows/tests/badge.svg)
![lint](https://github.com/greenelab/phenoplier/workflows/lint/badge.svg)

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

To prepare the environment to run the PhenoPLIER code, follow the steps in
[environment](environment/). That will create a conda environment and download
the necessary data.

# Running code

## From command-line

First of all, export your settings to environmental variables, so non-Python scripts
can access them:
```bash
eval `python libs/conf.py`
```

The code to preprocess data and generate results is in the `nbs/` folder. All
notebooks are organized by directories, such as `01_preprocessing`, with file
names that indicate the order in which they should be run (if they share the prefix, then it
means they can be run in parallel). For example, to run
all notebooks for the preprocessing step, you can use this command (requires
[GNU Parallel](https://www.gnu.org/software/parallel/)):

```bash
cd nbs/
parallel -k --lb --halt 2 -j1 'bash run_nbs.sh {}' ::: 01_preprocessing/*.ipynb
```

Or if you want to run all the analysis at once, you can use:

```bash
shopt -s globstar
parallel -k --lb --halt 2 -j1 'bash run_nbs.sh {}' ::: nbs/{,**/}*.ipynb
```

## From your browser

Alternatively, you can start your JupyterLab server by running:

```bash
bash scripts/run_nbs_server.sh
```

Then, go to `http://localhost:8892`, browse the `nbs` folder, and run the
notebooks in the specified order.

