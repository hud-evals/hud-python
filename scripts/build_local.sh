#!/bin/bash
# Build any HUD environment with local hud-python
# Usage: ./scripts/build_local_env.sh <environment_dir> [image_tag]
#
# Examples:
#   ./scripts/build_local_env.sh environments/browser
#   ./scripts/build_local_env.sh environments/browser my-custom-tag

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
ENVIRONMENT_DIR="$1"
IMAGE_TAG="${2:-local}"

# Validate arguments
if [ -z "$ENVIRONMENT_DIR" ]; then
    echo -e "${RED}Error: Environment directory is required${NC}"
    echo "Usage: $0 <environment_dir> [image_tag]"
    echo ""
    echo "Examples:"
    echo "  $0 environments/browser"
    echo "  $0 environments/browser my-custom-tag"
    exit 1
fi

# Get script directory and repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Validate environment directory exists
if [ ! -d "$REPO_ROOT/$ENVIRONMENT_DIR" ]; then
    echo -e "${RED}Error: Environment directory not found: $ENVIRONMENT_DIR${NC}"
    exit 1
fi

# Check for Dockerfile.local, fall back to Dockerfile
DOCKERFILE="$REPO_ROOT/$ENVIRONMENT_DIR/Dockerfile.local"
if [ ! -f "$DOCKERFILE" ]; then
    echo -e "${YELLOW}Warning: Dockerfile.local not found, using Dockerfile${NC}"
    DOCKERFILE="$REPO_ROOT/$ENVIRONMENT_DIR/Dockerfile"
    if [ ! -f "$DOCKERFILE" ]; then
        echo -e "${RED}Error: No Dockerfile found in $ENVIRONMENT_DIR${NC}"
        exit 1
    fi
fi

# Extract environment name from path
ENV_NAME=$(basename "$ENVIRONMENT_DIR")

# Build image name
IMAGE_NAME="hud-${ENV_NAME}:${IMAGE_TAG}"

echo -e "${GREEN}Building HUD environment with local hud-python${NC}"
echo "Environment: $ENVIRONMENT_DIR"
echo "Dockerfile: $(basename $DOCKERFILE)"
echo "Image: $IMAGE_NAME"
echo ""

# Change to repo root
# For Dockerfile.local, we need repo root context, so use docker build then hud build
cd "$REPO_ROOT"

if [ "$(basename $DOCKERFILE)" = "Dockerfile.local" ]; then
    echo -e "${YELLOW}Building with local hud-python (docker build with repo root context)...${NC}"
    echo ""
    
    # Build with docker using repo root as context
    docker build \
        -f "$DOCKERFILE" \
        -t "$IMAGE_NAME" \
        .
    
    if [ $? -ne 0 ]; then
        echo ""
        echo -e "${RED}Docker build failed!${NC}"
        exit 1
    fi
    
    echo ""
    echo -e "${GREEN}✅ Docker build complete!${NC}"
    echo ""
    
    # Update the lock file to reference our local image
    cd "$REPO_ROOT/$ENVIRONMENT_DIR"
    LOCK_FILE="hud.lock.yaml"
    
    if [ -f "$LOCK_FILE" ]; then
        echo -e "${YELLOW}Updating $LOCK_FILE...${NC}"
        # Update the local image reference
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            sed -i '' "s|local:.*|local: $IMAGE_NAME|" "$LOCK_FILE"
        else
            # Linux
            sed -i "s|local:.*|local: $IMAGE_NAME|" "$LOCK_FILE"
        fi
        echo -e "${GREEN}Updated $LOCK_FILE${NC}"
    else
        echo -e "${YELLOW}No hud.lock.yaml found - run 'hud build' first to create one${NC}"
    fi
else
    # Regular Dockerfile, use hud build normally
    cd "$REPO_ROOT/$ENVIRONMENT_DIR"
    echo -e "${YELLOW}Running hud build...${NC}"
    echo ""
    
    hud build . --tag "$IMAGE_NAME"
    
    if [ $? -ne 0 ]; then
        echo ""
        echo -e "${RED}Build failed!${NC}"
        exit 1
    fi
fi

echo ""
echo -e "${GREEN}✅ Build complete!${NC}"
echo ""
echo "To run the environment:"
echo "  hud dev"

