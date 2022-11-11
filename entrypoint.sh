#!/bin/bash --login
# Taken from here with modifications: https://pythonspeed.com/articles/activate-conda-dockerfile/
# The --login ensures the bash configuration is loaded,
# enabling Conda.
set +eu
conda activate phenoplier
set -eo pipefail

# load environment variables
eval `python libs/conf.py`

# make sure bash is used, not sh
export SHELL=$(type -p bash)

# if the environment variable is present, the following code will export
# the bash functions defined in it
set -a
eval "${PHENOPLIER_BASH_FUNCTIONS_CODE}"

exec "$@"

