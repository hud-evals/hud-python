#!/bin/bash

# Multi-architecture Docker build and push script for hud environments
# Usage: ./build-and-push-env.sh <environment|directory> <version>
# Example: ./build-and-push-env.sh browser 0.1.8
#          ./build-and-push-env.sh environments/browser 0.1.8
#          ./build-and-push-env.sh /full/path/to/env 0.1.8

set -e

ENV_INPUT=$1
VERSION=$2

if [ -z "$ENV_INPUT" ] || [ -z "$VERSION" ]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 <environment|directory> <version>"
    echo "Example: $0 browser 0.1.8"
    echo "         $0 environments/browser 0.1.8"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Check if input is a directory path or just a name
if [[ "$ENV_INPUT" == *"/"* ]] || [ -d "$ENV_INPUT" ] || [ -d "$PROJECT_ROOT/$ENV_INPUT" ]; then
    # It's a directory path
    if [ -d "$ENV_INPUT" ]; then
        ENV_PATH="$ENV_INPUT"
    else
        ENV_PATH="$PROJECT_ROOT/$ENV_INPUT"
    fi
    ENVIRONMENT=$(basename "$ENV_PATH")
else
    # It's just an environment name
    ENVIRONMENT="$ENV_INPUT"
    ENV_PATH="$PROJECT_ROOT/environments/${ENVIRONMENT}"
fi

IMAGE_NAME="hudevals/hud-${ENVIRONMENT}"

echo "Building and pushing hud-${ENVIRONMENT} version ${VERSION}"
echo "Environment path: $ENV_PATH"

if [ ! -d "$ENV_PATH" ]; then
    echo "Error: Environment directory not found: $ENV_PATH"
    exit 1
fi

if [ ! -f "$ENV_PATH/Dockerfile" ]; then
    echo "Error: Dockerfile not found in $ENV_PATH"
    exit 1
fi

cd "$ENV_PATH"
echo "Working directory: $(pwd)"

echo "Building ARM64 image..."
docker build --platform linux/arm64 -t $IMAGE_NAME:$VERSION-arm64 .

echo "Building AMD64 image..."
docker build --platform linux/amd64 -t $IMAGE_NAME:$VERSION-amd64 .

echo "Tagging images as latest..."
docker tag $IMAGE_NAME:$VERSION-arm64 $IMAGE_NAME:latest-arm64
docker tag $IMAGE_NAME:$VERSION-amd64 $IMAGE_NAME:latest-amd64

echo "Pushing ARM64 image..."
docker push $IMAGE_NAME:$VERSION-arm64
docker push $IMAGE_NAME:latest-arm64

echo "Pushing AMD64 image..."
docker push $IMAGE_NAME:$VERSION-amd64
docker push $IMAGE_NAME:latest-amd64

echo "Creating and pushing multi-arch manifest..."
docker manifest create $IMAGE_NAME:$VERSION \
    --amend $IMAGE_NAME:$VERSION-arm64 \
    --amend $IMAGE_NAME:$VERSION-amd64

docker manifest create $IMAGE_NAME:latest \
    --amend $IMAGE_NAME:latest-arm64 \
    --amend $IMAGE_NAME:latest-amd64

echo "Pushing manifests..."
docker manifest push $IMAGE_NAME:$VERSION
docker manifest push $IMAGE_NAME:latest

echo "Successfully built and pushed:"
echo "  - $IMAGE_NAME:$VERSION (multi-arch)"
echo "  - $IMAGE_NAME:latest (multi-arch)"
echo "  - $IMAGE_NAME:$VERSION-arm64"
echo "  - $IMAGE_NAME:$VERSION-amd64"
echo "  - $IMAGE_NAME:latest-arm64"
echo "  - $IMAGE_NAME:latest-amd64"

echo "Done!"
