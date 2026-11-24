#!/bin/sh
# Passes only a solution that ran in the workdir with [solution].env (solve.sh fails
# without the agent's egress).
mkdir -p /logs/verifier
if [ "$(cat /workspace/result.txt 2>/dev/null)" = solved ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
