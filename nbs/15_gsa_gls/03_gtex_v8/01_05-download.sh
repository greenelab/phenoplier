#!/bin/bash

# Instructions from here: https://anvilproject.org/learn/reference/gtex-v8-free-egress-instructions
# You need to generate manifest files only for some data types and file extensions (see README.md).
#
# The file does not run correctly within a job in PMACS.

export GEN3_BINARY=~/projects/anvil/software/gen3-client
if [ ! -f ${GEN3_BINARY} ]; then
  echo "ERROR: download and install gen3-client for your platform"
  exit 1
fi

cd $PHENOPLIER_EXTERNAL_GTEX_V8_DIR

mkdir haplotype_phasing
${GEN3_BINARY} download-multiple \
  --profile=miltondp_gtex \
  --manifest=~/projects/anvil/file-manifest-haplotype_phasing.json \
  --download-path=haplotype_phasing \
  --protocol=s3

mkdir analysis_supplement
${GEN3_BINARY} download-multiple \
  --profile=miltondp_gtex \
  --manifest=~/projects/anvil/file-manifest-analysis_supplement.json \
  --download-path=analysis_supplement \
  --protocol=s3

# this file is not present in dbGaP
wget \
  https://storage.googleapis.com/gtex_analysis_v8/reference/GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_838Indiv_Analysis_Freeze.lookup_table.txt.gz \
  -P analysis_supplement/

# # Commands to download other files (in case it's necessary)
# mkdir variants_annotation
# ${GEN3_BINARY} download-multiple \
#   --profile=miltondp_gtex \
#   --manifest=~/projects/anvil/file-manifest-variants_annotation-for-anni.json \
#   --download-path=variants_annotation \
#   --protocol=s3
