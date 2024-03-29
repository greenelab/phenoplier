# Overview

This folder has the scripts to download and preprocess GTEx v8 data.
The user needs to request access to data-protected files in GTEx v8 to run these steps.

# Download GTEx v8 data

For this, approval must have been granted for the user.
Then follow the instructions in Gen3/AnVIL: https://anvilproject.org/learn/reference/gtex-v8-free-egress-instructions

The only files that are needed are the following:
1. `GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze.SHAPEIT2_phased.vcf.gz`
1. `GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_899Indiv_QC_TABLE.tsv`
1. `GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_flagged_donors.txt`
1. `GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze.lookup_table.txt.gz`

So adjust the file selection and download the manifest file only for those, to avoid downloading many unneeded files.
See `01_05-download.sh`.

If you receive an error with RequestNewAccessToken, then you need to update your credentials.
First, go to https://anvil.terra.bio/#profile and renew the links.
Then, go to https://gen3.theanvil.io/identity and delete the current API key and create a new one, and download the `credentials.json` file.
Finally, run this command:

```bash
~/projects/anvil/software/gen3-client configure \
  --profile=miltondp_gtex \
  --cred=/home/miltondp/credentials.json \
  --apiendpoint=https://gen3.theanvil.io
```


# Load Penn's LPC-specific paths and PhenoPLIER configuration

Change paths accordingly.

```bash
# load conda environment
module load miniconda/3
conda activate ~/software/conda_envs/phenoplier_light/

# load LPC-specific paths
. ~/projects/phenoplier/scripts/pmacs_penn/env.sh

# load in bash session all PhenoPLIER environmental variables
eval `python ~/projects/phenoplier/libs/conf.py`

# make sure they were loaded correctly
# should output something like /project/...
echo $PHENOPLIER_ROOT_DIR
```

# Download GTEx data

Run this without submitting it to the cluster queue (in PMACS), because it cannot access Internet.

```bash
bash 01_05-download.sh
```


# Preprocessing

## GTEx ancestry

```bash
mkdir -p _tmp
cat 01_10-samples_ancestry.sh | bsub
```

## Genotype dosage

```bash
mkdir -p _tmp
cat 03_05-genotype_dosage.sh | bsub
```

## Variant selection

```bash
mkdir -p _tmp
cat 05_05-variant_selection.sh | bsub
```

## Genotype compilation

```bash
mkdir -p _tmp
cat 07_05-genotype_compilation.sh | bsub
```

