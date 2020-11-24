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
