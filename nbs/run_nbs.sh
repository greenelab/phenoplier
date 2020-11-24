#!/bin/bash
set -e

#export SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ -z "${1}" ]; then
  echo "Specify notebook to run"
  exit 1
fi

filename="${1%.*}.run.ipynb"

#export PYTHONPATH=${SCRIPT_DIR}/../src/
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

