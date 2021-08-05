#!/bin/bash --login
# The --login ensures the bash configuration is loaded,
# enabling Conda.
set +eu
conda activate phenoplier
set -euo pipefail

exec "$@"

