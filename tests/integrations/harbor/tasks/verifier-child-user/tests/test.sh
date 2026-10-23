#!/bin/sh
set -eu
mkdir -p /logs/verifier
if su verifier -s /bin/sh -c 'test "$(cat /root/agent-output.txt)" = private'; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
