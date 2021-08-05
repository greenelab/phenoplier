#!/bin/bash --login
# Taken from here with modifications: https://pythonspeed.com/articles/activate-conda-dockerfile/
# The --login ensures the bash configuration is loaded,
# enabling Conda.
set +eu
conda activate phenoplier
set -euo pipefail

exec "$@"

