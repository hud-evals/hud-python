#!/bin/sh
mkdir -p /logs/verifier
if [ "$(cat /logs/agent_answer.txt)" = "forty-two" ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
