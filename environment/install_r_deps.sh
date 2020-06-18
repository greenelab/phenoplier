#!/bin/bash

# This file is intended to have the commands to install R dependencies.

## R: BiocManager
#$ export TAR=/bin/tar
#$ R
#> if (!requireNamespace("BiocManager", quietly = TRUE))
#    install.packages("BiocManager")

# R: clusterProfiler
#$ export TAR=/bin/tar
#$ R
#> BiocManager::install("clusterProfiler")
#...
#> BiocManager::install("org.Hs.eg.db")
#...

## (optional) PLIER
#$ export TAR=/bin/tar
#$ R
#> library(devtools)
#> install_github("wgmao/PLIER")

