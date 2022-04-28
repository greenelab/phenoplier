#!/bin/bash

# This script installs other dependencies that cannot be directly installed
# using conda.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

#
# R dependencies
#
TAR=$(which tar) Rscript ${SCRIPT_DIR}/install_r_packages.r
