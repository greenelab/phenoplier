#!/bin/bash

# This file is project-specific and it exports some common environmental
# variables to run the code.  It has to be customized for your need by changing
# the BASE_DIR and N_JOBS below.

#
# Your settings here
#

# BASE_DIR is the parent directory where the code and manuscript repos are
# located.
BASE_DIR=/home/miltondp/projects/phenoplier/greenelab

# Project name
PROJECT_NAME=phenoplier

# Number of CPUs to use
export PHENOPLIER_N_JOBS=3

# Docker image tag to use
DOCKER_IMAGE_TAG="latest"

#
# Do not edit below
#
export PHENOPLIER_ROOT_DIR=${BASE_DIR}/${PROJECT_NAME}/base
echo "PHENOPLIER_ROOT_DIR=${PHENOPLIER_ROOT_DIR}"

export PHENOPLIER_MANUSCRIPT_DIR=${BASE_DIR}/${PROJECT_NAME}_manuscript/
echo "PHENOPLIER_MANUSCRIPT_DIR=${PHENOPLIER_MANUSCRIPT_DIR}"

export PHENOPLIER_DOCKER_IMAGE_TAG="${DOCKER_IMAGE_TAG:-latest}"
echo "PHENOPLIER_DOCKER_IMAGE_TAG=${PHENOPLIER_DOCKER_IMAGE_TAG}"

export PYTHONPATH=${BASE_DIR}/${PROJECT_NAME}/libs/:${PYTHONPATH}
echo "PYTHONPATH=${PYTHONPATH}"
