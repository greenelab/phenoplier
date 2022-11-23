#!/bin/bash

# This file is project-specific and it exports some common environmental
# variables to run the code.  It has to be customized for your needs by changing
# the BASE_DIR and N_JOBS below.

#
# Your settings here
#

# Project name
PROJECT_NAME="asthma-copd"

# BASE_DIR is the parent directory where the code and manuscript repos are
# located.
BASE_DIR=/home/miltondp/projects/phenoplier/projects/${PROJECT_NAME}

# Number of CPUs to use
export PHENOPLIER_N_JOBS=10

# Docker image tag to use
DOCKER_IMAGE_TAG="asthma-copd"

#
# Do not edit below
#
export PHENOPLIER_ROOT_DIR=${BASE_DIR}/code/base
echo -e "PHENOPLIER_ROOT_DIR=\t\t${PHENOPLIER_ROOT_DIR}"

export PHENOPLIER_MANUSCRIPT_DIR=${BASE_DIR}/manuscript
echo -e "PHENOPLIER_MANUSCRIPT_DIR=\t${PHENOPLIER_MANUSCRIPT_DIR}"

echo -e "PHENOPLIER_N_JOBS=\t\t${PHENOPLIER_N_JOBS}"

export PHENOPLIER_DOCKER_IMAGE_TAG="${DOCKER_IMAGE_TAG:-latest}"
echo -e "PHENOPLIER_DOCKER_IMAGE_TAG=\t${PHENOPLIER_DOCKER_IMAGE_TAG}"

export PYTHONPATH=${BASE_DIR}/code/libs/:${PYTHONPATH}
echo -e "PYTHONPATH=\t\t\t${PYTHONPATH}"
