#/bin/bash

PROJECT_NAME="phenoplier"
VERSION="asthma-dev"

docker build -t miltondp/${PROJECT_NAME}:${VERSION} .

read -p "'docker push' new image? " -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]; then
  # push version label
  docker push miltondp/${PROJECT_NAME}:${VERSION}

  # update description (short 100 chars)
  # update README.md in Docker Hub
fi

