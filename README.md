# PhenoPLIER

![tests](https://github.com/greenelab/phenoplier/workflows/tests/badge.svg)

PhenoPLIER is a computational approach that integrates gene co-expression
patterns (latent variables representing gene modules) learned from large
expression compendia with TWAS (transcriptome-wide association studies) and
pharmacological perturbations. This allows to infer 1) how gene modules affect
complex traits, 2) in which conditions (cell types, tissues, etc) are those
genes activated, and 3) how those genes' activity could be modulated with
different compounds. PhenoPLIER aims to find novel gene module associations
with complex traits, and also new perturbations that can suggest alternative
treatment options when some genes are not druggable.

# Setup

To prepare the environment to run the PhenoPLIER code, follow the steps in [environment](environment/).
That will create a conda environment and download the necessary data.

# Running code

Once the environment is ready, you can start your JupyterLab server by running:

```bash
bash scripts/run_nbs.sh
```

Go to `http://localhost:8892` and browse the `nbs` folder. Then run notebooks in the specified order.
