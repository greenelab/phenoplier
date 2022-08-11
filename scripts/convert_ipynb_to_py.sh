#!/bin/bash

# show commands being executed (for debugging purposes)
#set -x

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

NOTEBOOK="${1}"
if [ -z "${NOTEBOOK}" ]; then
  echo "Provide the notebook path"
  exit 1
fi

# if file is python script (.py), then just run black on
filename=$(basename -- "${NOTEBOOK}")
extension="${filename##*.}"
if [ "$extension" = "py" ]; then
  echo "Input file is python script, running black only"
  black ${NOTEBOOK}
  exit 0
fi

# capture whether notebook has a python or R kernel
regex="\"file_extension\": \"(\.[a-zA-Z]+)\"\,"
value=`cat ${NOTEBOOK} | grep "file_extension"`
if [[ $value =~ $regex ]]; then
  fext="${BASH_REMATCH[1]}"
else
  echo "ERROR: file extension not found"
  exit 1
fi

# select code formatter according to file extension
PIPE_CMD=("black {}")
if [ "$fext" = ".r" ] || [ "$fext" = ".R" ]; then
  PIPE_CMD=("${SCRIPT_DIR}/styler.r {}")
fi

jupytext \
  --sync \
  --pipe "${PIPE_CMD[@]}" \
  ${NOTEBOOK}

