#!/bin/bash

NOTEBOOK="${1}"
if [ -z "${NOTEBOOK}" ]; then
  echo "Provide the notebook path"
  exit 1
fi

jupytext \
  --sync \
  --pipe black \
  ${NOTEBOOK}

