#!/bin/sh
mkdir -p /logs/verifier
if [ "$(wc -l < /workspace/sessions)" -eq 40 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
