#!/bin/bash
set -u
mkdir -p /logs/verifier
fail() { echo "$1"; echo 0 > /logs/verifier/reward.txt; exit 0; }

[ "$(cat /app/identity.txt 2>/dev/null)" = "1000:2000" ] \
  || fail "the agent did not run as 1000:2000"
[ "$(stat -c %u:%g /app/identity.txt 2>/dev/null)" = "1000:2000" ] \
  || fail "the agent's file has the wrong owner"
curl -sf --max-time 10 -o /dev/null http://127.0.0.1:9000/ \
  || fail "the agent's service did not survive its command"

echo 1 > /logs/verifier/reward.txt
