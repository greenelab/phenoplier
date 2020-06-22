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
conda env create -n phenoplier -f ${SCRIPT_DIR}/environment_base.yml
