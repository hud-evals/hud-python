#!/bin/sh
mkdir -p /logs/verifier
if [ "$(cat /app/outputs/result.txt 2>/dev/null)" = "from-volume" ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
