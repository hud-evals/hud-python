#!/bin/sh
# Passes only a solution that ran as the agent, in the workdir, with [solution].env
# (solve.sh fails without the agent's egress), and that the controller removed from
# /solution before grading.
mkdir -p /logs/verifier
if [ ! -e /solution ] \
  && [ "$(cat /workspace/result.txt 2>/dev/null)" = solved ] \
  && [ "$(stat -c %U /workspace/result.txt)" = solver ] \
  && [ "$(cat /logs/agent_answer.txt)" = solver ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
