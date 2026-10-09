#!/bin/sh
set -eu
mkdir -p /logs/verifier
cd /logs/verifier
case "$SHAPE" in
  json-preferred) echo '{"reward": 0.75}' > reward.json; echo 0.1 > reward.txt ;;
  json-score) echo '{"score": 0.5}' > reward.json ;;
  json-number) echo '0.25' > reward.json ;;
  json-bool) echo 'true' > reward.json; echo 1 > reward.txt ;;
  json-infinite) echo '{"reward": Infinity}' > reward.json ;;
  json-invalid) echo '{' > reward.json ;;
  text) echo ' 0.5 ' > reward.txt ;;
  text-nan) echo nan > reward.txt ;;
  text-words) echo 'one point zero' > reward.txt ;;
  none) ;;
esac
