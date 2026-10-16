#!/bin/sh
mkdir -p /logs/verifier
if grep -q 'required-version = "==0.0.0"' /workspace/pyproject.toml; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
