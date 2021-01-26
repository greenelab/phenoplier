#!/bin/bash

# The script allows to run a JupyterLab server, listening to local connections
# only by default. If the only optional argument is given, then the server will
# request a token from users and will listen to any address (*).

PORT=8892
export PYTHONPATH=`pwd`/libs/
echo "PYTHONPATH=${PYTHONPATH}"

# export the PhenoPLIER configuration as environmental variables (this is
# helpful if the configuration is needed outside python)
eval `python libs/conf.py`

IP="127.0.0.1"
TOKEN=""
if [ ! -z "$1" ]; then
	IP="*"
	TOKEN="${1}"
fi

jupyter lab --ip="${IP}" --port ${PORT} --no-browser --NotebookApp.token="${TOKEN}"

