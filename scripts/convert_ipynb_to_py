#!/bin/bash

NOTEBOOK="${1}"
if [ -z "${NOTEBOOK}" ]; then
  echo "Provide the notebook path"
  exit 1
fi

nbs_folder="$( cd "$(dirname "$NOTEBOOK")" >/dev/null 2>&1 ; pwd -P )"
nbs_base=${NOTEBOOK##*/}
nbs_fext=${nbs_base##*.}
nbs_xpref=${nbs_base%.*}

echo $nbs_folder
echo $nbs_base
echo $nbs_fext
echo $nbs_xpref

output_folder="${nbs_folder}/py"
mkdir -p $output_folder

output_file="${output_folder}/${nbs_xpref}.py"

jupytext \
  --to auto:percent \
  ${NOTEBOOK} \
  --output ${output_file}

black ${output_file}

