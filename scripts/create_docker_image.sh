#/bin/bash
set -e

PROJECT_NAME="phenoplier"
VERSION="1.3.2"

CURRENT_IMAGE_ID=$(docker images --filter=reference=miltondp/${PROJECT_NAME}:latest --format "{{.ID}}")

export DOCKER_BUILDKIT=1
docker build \
  -t miltondp/${PROJECT_NAME}:${VERSION} \
  -f Dockerfile \
  --target final .

# command to tag as latest:
# docker tag miltondp/${PROJECT_NAME}:${VERSION} miltondp/${PROJECT_NAME}:latest

# command to tag and keep previous version
# docker tag ${CURRENT_IMAGE_ID} miltondp/${PROJECT_NAME}:prev

# command to push new version and latest
# docker push miltondp/${PROJECT_NAME}:${VERSION}
# docker push miltondp/${PROJECT_NAME}:latest

