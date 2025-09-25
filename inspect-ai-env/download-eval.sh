#!/bin/bash

# Default to mbpp if TARGET_EVAL is not set
TARGET_EVAL=${TARGET_EVAL:-mbpp}

# Check if eval already exists
if [ -d "/app/inspect_evals/${TARGET_EVAL}" ]; then
    echo "✅ Eval ${TARGET_EVAL} already exists, skipping download"
else
    echo "📥 Downloading eval: ${TARGET_EVAL}"

    # Download specific eval using sparse checkout
    git clone --filter=blob:none --sparse https://github.com/UKGovernmentBEIS/inspect_evals.git inspect_evals_repo
    cd inspect_evals_repo
    git sparse-checkout set src/inspect_evals/${TARGET_EVAL}
    cd ..

    # Copy to the expected location
    cp -r inspect_evals_repo/src/inspect_evals/${TARGET_EVAL} inspect_evals/
    rm -rf inspect_evals_repo

    echo "✅ Downloaded eval: ${TARGET_EVAL}"
fi