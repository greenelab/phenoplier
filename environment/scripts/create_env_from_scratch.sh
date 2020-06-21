#!/bin/bash

# This script is intended for developers use only, not users. It creates the environment from
# scratch using the latest version of packages.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# check that conda is installed
if ! [ -x "$(command -v conda)" ]; then
  echo 'Error: conda is not installed. Install Miniconda first.' >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

#
# create environment
#
conda create -y -n phenoplier \
  ipython \
  jupyterlab \
  matplotlib \
  nodejs \
  numpy \
  pandas \
  pip \
  python=3.8 \
  r-devtools \
  r-base \
  r-ggplot2 \
  scikit-learn \
  scipy \
  seaborn \
  tzlocal


# activate environment
conda activate phenoplier

#
# Python packages
#   Always prefer package's dependencies from conda instead of pip
#

## rpy2
conda install -y \
    more-itertools \
    packaging \
    pluggy \
    py \
    pytest

pip install rpy2

## umap-learn
conda install -y \
    numba \
    tbb

pip install umap-learn
