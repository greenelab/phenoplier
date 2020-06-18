#!/bin/bash

# This script creates the environment from scratch using the latest version of needed packages. It's intended to use
# when trying to update and test new package versions. If you are not a developer and just want to use the PhenoPLIER
# code, you might just want to use the `environment.yml` for your platform instead of running this script.

eval "$(conda shell.bash hook)"

# create environment
conda create -y -n phenoplier \
  python=3.7 \
  pandas \
  scikit-learn \
  matplotlib \
  seaborn \
  numpy \
  scipy \
  jupyterlab \
  r-essentials \
  r-devtools \
  r-ggplot2 \
  tzlocal \
  nodejs

# activate environment
conda activate phenoplier

#
# rpy2
#
conda install -y \
    more-itertools \
    packaging \
    pluggy \
    py \
    pytest

pip install rpy2

#
# umap-learn
#
conda install -y \
    numba \
    tbb

pip install umap-learn

#
# jupyterlab extensions
#
jupyter labextension install @jupyterlab/toc

# R dependencies
bash ./install_r_deps.sh
