#!/bin/bash
set -e

# This script runs a Jupyter notebook (.ipynb) from the command line using
# papermill. When it finishes, it asks whether it should override the notebook
# or not; this can be specified with environmental variable
# PHENOPLIER_RUN_NBS_OVERRIDE=1
#
# This script must be run within the nbs/ folder.

if [ -z "${1}" ]; then
  echo "Specify notebook to run"
  exit 1
fi

# If the notebook is an output of another notebook (filename matches
# *.out.ipynb), then do not run it
pattern="*.out.ipynb"

if [[ $1 == $pattern ]]; then
  echo "Not running output notebook"
  exit 0
fi

filename="${1%.*}.run.ipynb"

papermill \
  --log-output \
  --request-save-on-cell-execute \
  $1 \
  $filename

if [ "${PHENOPLIER_RUN_NBS_OVERRIDE}" != "1" ]; then
    echo "Execution finished. Do you wish to override the notebook with the run one?"
    select yn in "Yes" "No"; do
        case $yn in
            Yes ) mv $filename $1; break;;
            No ) exit;;
        esac
    done
else
    mv $filename $1
fi

if [ "${PHENOPLIER_RUN_NBS_NO_JUPYTEXT}" != "1" ]; then
  bash ../scripts/convert_ipynb_to_py.sh ${1}
fi
