#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Check if TARGET_EVAL is set and non-empty. If not, do nothing.
if [ -z "${TARGET_EVAL}" ]; then
    echo "TARGET_EVAL is not set. Nothing to do."
fi

# Define all paths based on the Current Working Directory (CWD) to avoid ambiguity.
CWD=$(pwd)
TARGET_DIR="${CWD}/inspect_evals/${TARGET_EVAL}"

# Check if the target directory already exists.
if [ -d "${TARGET_DIR}" ]; then
    echo "Eval '${TARGET_EVAL}' already exists. Skipping download."
fi

echo "Downloading eval: ${TARGET_EVAL}"

# Create a temporary directory for the git clone.
# Using 'trap' ensures this directory is cleaned up automatically when the script exits,
# even if it fails unexpectedly.
TEMP_REPO_DIR=$(mktemp -d)
trap 'rm -rf -- "$TEMP_REPO_DIR"' EXIT

# --- Perform Git Operations ---
# Clone the repository without checking out files into the temporary directory.
git clone --filter=blob:none --no-checkout https://github.com/UKGovernmentBEIS/inspect_evals.git "${TEMP_REPO_DIR}"

# Run the directory-changing commands inside a subshell.
# This keeps the main script's context in the original directory.
(
    cd "${TEMP_REPO_DIR}"
    git sparse-checkout set "src/inspect_evals/${TARGET_EVAL}"
    git checkout
)

# --- Organize Files ---
# Create the parent directory `inspect_evals` if it doesn't exist in your project.
mkdir -p "${CWD}/inspect_evals"

# Copy the specific eval from the temporary repo to its final destination.
cp -r "${TEMP_REPO_DIR}/src/inspect_evals/${TARGET_EVAL}" "${TARGET_DIR}"

# Create __init__.py to make inspect_evals a proper Python package
touch "${CWD}/inspect_evals/__init__.py"

echo "Successfully downloaded '${TARGET_EVAL}' to '${TARGET_DIR}'"
# The 'trap' command will now execute, cleaning up the temporary directory.