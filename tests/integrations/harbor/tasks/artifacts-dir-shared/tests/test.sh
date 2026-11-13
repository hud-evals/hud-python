#!/bin/sh
if [ "$(cat /logs/artifacts/mode.txt)" = "shared" ] \
  && [ "$(cat /tmp/model.patch)" = "diff" ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo "the verifier did not see /logs/artifacts or the collect hook's output" >&2
  echo 0 > /logs/verifier/reward.txt
fi
