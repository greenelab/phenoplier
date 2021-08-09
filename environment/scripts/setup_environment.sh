#!/bin/bash
set -e

# create conda environment and install main packages
conda env create --name phenoplier --file environment.yml

conda activate phenoplier

# install other packages
bash scripts/install_other_packages.sh

# download the data
python scripts/setup_data.py

