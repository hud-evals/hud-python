#!/bin/sh
if [ "$(cat /logs/artifacts/mode.txt)" = "separate" ] \
  && [ "$(cat /logs/artifacts/model.patch)" = "diff" ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo "the verifier did not receive /logs/artifacts" >&2
  echo 0 > /logs/verifier/reward.txt
fi
