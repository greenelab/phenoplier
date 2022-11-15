#/bin/bash
set -eo pipefail
IFS=$'\n\t'

#
# Settings to be modified.
#

export NAMESPACE="miltondp"
export PROJECT_NAME="phenoplier"

# Version of base image
#  Increase to force a new build of base and the creation of a new
#  conda/python environment. Ideally, this should be done only once
#  or when it is necessary to rebuild the environment.
export BASE_VERSION="2.0.0"

# Version of final image
#  This image is based on the base image but with the source code copied.
#  It is the final image that will be used. It should be run at the end
#  to have an image with the latest source code for end users.
#  The convention of the version is to be at least the version of the
#  base image.
export FINAL_VERSION="2.0.0"

#
# Argument parsing
#

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -f|--force)
      FORCE_BUILD=YES
      shift # past argument
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

#
# Build process, do not modify below.
#

export DOCKER_BUILDKIT=1
export BUILDKIT_PROGRESS=plain

CURRENT_IMAGE_ID=$(docker images --filter=reference=${NAMESPACE}/${PROJECT_NAME}:latest --format "{{.ID}}")

BASE_LATEST_LABEL="base-latest"
BASE_VERSION_LABEL="base-${BASE_VERSION}"

# if the following two variables point to the same image, then it won't
# be necessary to build a new base
CURRENT_BASE_LATEST_IMAGE_ID=$(docker images --filter=reference=${NAMESPACE}/${PROJECT_NAME}:${BASE_LATEST_LABEL} --format "{{.ID}}")
if [ -z "${CURRENT_BASE_LATEST_IMAGE_ID}" ]; then
  CURRENT_BASE_LATEST_IMAGE_ID="no base latest"
fi

CURRENT_BASE_VERSION_IMAGE_ID=$(docker images --filter=reference=${NAMESPACE}/${PROJECT_NAME}:${BASE_VERSION_LABEL} --format "{{.ID}}")
if [ -z "${CURRENT_BASE_VERSION_IMAGE_ID}" ]; then
  CURRENT_BASE_VERSION_IMAGE_ID="no base version"
fi

echo "CURRENT IMAGES:"
echo -e "Latest final image ID:\t\t${CURRENT_IMAGE_ID}"
echo -e "Latest base image ID:\t\t${CURRENT_BASE_LATEST_IMAGE_ID}"
echo -e "Base ${BASE_VERSION} image ID:\t\t${CURRENT_BASE_VERSION_IMAGE_ID}"
echo

if [ ! -z "${FORCE_BUILD}" ] || [ "${CURRENT_BASE_LATEST_IMAGE_ID}" != "${CURRENT_BASE_VERSION_IMAGE_ID}" ]; then
  echo "WARNING: base image does not exist or version was changed. Forcing creation of base image..."
  sleep 5

  docker build \
    --pull \
    --no-cache \
    -t ${NAMESPACE}/${PROJECT_NAME}:${BASE_VERSION_LABEL} \
    -t ${NAMESPACE}/${PROJECT_NAME}:${BASE_LATEST_LABEL} \
    -f Dockerfile \
    --target base .
fi

echo "Creating final image"
sleep 5

docker build \
  -t ${NAMESPACE}/${PROJECT_NAME}:${FINAL_VERSION} \
  -f Dockerfile \
  --target final .

# command to tag and keep previous version:
# docker tag ${CURRENT_IMAGE_ID} ${NAMESPACE}/${PROJECT_NAME}:prev

# command to tag as latest:
# docker tag ${NAMESPACE}/${PROJECT_NAME}:${FINAL_VERSION} ${NAMESPACE}/${PROJECT_NAME}:latest

# commands to push new version and latest:
# docker push ${NAMESPACE}/${PROJECT_NAME}:${BASE_VERSION_LABEL}
# docker push ${NAMESPACE}/${PROJECT_NAME}:${BASE_LATEST_LABEL}
#
# docker push ${NAMESPACE}/${PROJECT_NAME}:${FINAL_VERSION}
# docker push ${NAMESPACE}/${PROJECT_NAME}:latest
