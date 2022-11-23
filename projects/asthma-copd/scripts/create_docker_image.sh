#/bin/bash
set -eo pipefail
IFS=$'\n\t'

# This script simply pulls the right Docker image version and tags it as "asthma-copd"

docker pull miltondp/phenoplier:2.0.0
docker tag miltondp/phenoplier:2.0.0 miltondp/phenoplier:asthma-copd
