#/bin/bash
set -e

export NAMESPACE="miltondp"
export PROJECT_NAME="phenoplier"
export VERSION="2.0.0-dev"

CURRENT_IMAGE_ID=$(docker images --filter=reference=${NAMESPACE}/${PROJECT_NAME}:latest --format "{{.ID}}")

export DOCKER_BUILDKIT=1
docker build \
  -t ${NAMESPACE}/${PROJECT_NAME}:${VERSION} \
  -f Dockerfile \
  --target final .

# command to tag as latest:
# docker tag ${NAMESPACE}/${PROJECT_NAME}:${VERSION} ${NAMESPACE}/${PROJECT_NAME}:latest

# command to tag and keep previous version
# docker tag ${CURRENT_IMAGE_ID} ${NAMESPACE}/${PROJECT_NAME}:prev

# command to push new version and latest
# docker push ${NAMESPACE}/${PROJECT_NAME}:${VERSION}
# docker push ${NAMESPACE}/${PROJECT_NAME}:latest
