#!/bin/bash

TARGET_EVAL=${TARGET_EVAL}

# Check if eval already exists
if ! [ -d "/app/inspect_evals/${TARGET_EVAL}" ]; then

    # Download specific eval using sparse checkout
    git clone --filter=blob:none --sparse https://github.com/UKGovernmentBEIS/inspect_evals.git inspect_evals_repo
    cd inspect_evals_repo
    git sparse-checkout set src/inspect_evals/${TARGET_EVAL}
    cd ..

    # Copy to the expected location
    cp -r inspect_evals_repo/src/inspect_evals/${TARGET_EVAL} inspect_evals/${TARGET_EVAL}/
    rm -rf inspect_evals_repo

fi