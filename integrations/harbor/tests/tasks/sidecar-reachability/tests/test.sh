#!/bin/sh
set -u
mkdir -p /logs/verifier

if grep -q "Directory listing" /app/sidecar.html 2>/dev/null \
  && grep -q "Directory listing" /app/main.html 2>/dev/null \
  && [ "$(cat /tmp/main.txt 2>/dev/null)" = "collected-from-main" ] \
  && [ "$(cat /tmp/sidecar.txt 2>/dev/null)" = "collected-from-sidecar" ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo "the agent output or collected sidecar artifact is missing"
  echo 0 > /logs/verifier/reward.txt
fi
