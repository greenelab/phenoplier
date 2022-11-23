#!/bin/bash

# This file is project-specific and it exports some common environmental
# variables to run the code.  It has to be customized for your needs by changing
# the BASE_DIR and N_JOBS below.

#
# Your settings here
#

# BASE_DIR is the parent directory where the code and manuscript repos are
# located.
BASE_DIR=/home/miltondp/projects/phenoplier/projects/asthma-copd

# Project name
PROJECT_NAME=asthma-copd

# Number of CPUs to use
export PHENOPLIER_N_JOBS=10

# Docker image tag to use
DOCKER_IMAGE_TAG="asthma-copd"

#
# Do not edit below
#
export PHENOPLIER_ROOT_DIR=${BASE_DIR}/code/base
echo "PHENOPLIER_ROOT_DIR=${PHENOPLIER_ROOT_DIR}"

export PHENOPLIER_MANUSCRIPT_DIR=${BASE_DIR}/manuscript
echo "PHENOPLIER_MANUSCRIPT_DIR=${PHENOPLIER_MANUSCRIPT_DIR}"

export PHENOPLIER_DOCKER_IMAGE_TAG="${DOCKER_IMAGE_TAG:-latest}"
echo "PHENOPLIER_DOCKER_IMAGE_TAG=${PHENOPLIER_DOCKER_IMAGE_TAG}"

export PYTHONPATH=${BASE_DIR}/code/libs/:${PYTHONPATH}
echo "PYTHONPATH=${PYTHONPATH}"
