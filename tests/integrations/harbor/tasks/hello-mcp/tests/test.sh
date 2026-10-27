#!/bin/sh
set -u
mkdir -p /logs/verifier

if [ "$(cat /app/secret.txt 2>/dev/null)" = "harbor-mcp-secret-12345" ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
