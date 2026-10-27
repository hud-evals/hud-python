#!/bin/sh
set -u
mkdir -p /logs/verifier
if [ "$(cat /app/outputs/keep.txt 2>/dev/null)" = "keep" ] \
  && [ -d /app/outputs/logs ] \
  && [ ! -e /app/outputs/junk.tmp ] \
  && [ ! -e /app/outputs/logs/nested.tmp ] \
  && [ ! -L /app/outputs/logs/dangling.tmp ] \
  && [ ! -e /app/outputs/cache ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo "excluded artifact entries leaked into the verifier" >&2
  echo 0 > /logs/verifier/reward.txt
fi
