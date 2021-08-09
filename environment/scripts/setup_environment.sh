#!/bin/bash
set -e

# create conda environment and install main packages
conda env create --name phenoplier --file environment.yml

# install other packages
conda run -n phenoplier --no-capture-output bash scripts/install_other_packages.sh

# download the data
conda run -n phenoplier --no-capture-output python scripts/setup_data.py

