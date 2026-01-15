#!/bin/bash
# Run PDFBench evaluation with HUD

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HUD_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== PDFBench Evaluation ===${NC}"

# Check Docker
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker daemon is not running${NC}"
    echo "Please start Docker Desktop or the Docker daemon first."
    exit 1
fi

# Build Docker image if needed
IMAGE_NAME="hud-pdfbench:latest"
if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
    echo -e "${YELLOW}Building Docker image...${NC}"
    cd "$HUD_DIR"
    docker build -f environments/pdfbench/Dockerfile -t $IMAGE_NAME .
    echo -e "${GREEN}Docker image built successfully${NC}"
else
    echo -e "${GREEN}Docker image $IMAGE_NAME already exists${NC}"
fi

# Parse arguments
AGENT="${1:-gemini}"
TASKS="${2:-$SCRIPT_DIR/tasks_sample.json}"
MAX_STEPS="${3:-30}"

echo -e "${YELLOW}Running evaluation with:${NC}"
echo "  Agent: $AGENT"
echo "  Tasks: $TASKS"
echo "  Max steps: $MAX_STEPS"
echo ""

# Run evaluation
hud eval "$TASKS" --agent "$AGENT" --max-steps "$MAX_STEPS" -v

echo -e "${GREEN}Evaluation complete!${NC}"
