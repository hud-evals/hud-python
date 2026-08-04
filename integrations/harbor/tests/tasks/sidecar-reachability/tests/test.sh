#!/bin/sh
set -u
mkdir -p /logs/verifier

if grep -q "Directory listing" /app/sidecar.html 2>/dev/null; then
  echo 1 > /logs/verifier/reward.txt
else
  echo "the agent could not reach the declared web sidecar"
  echo 0 > /logs/verifier/reward.txt
fi
