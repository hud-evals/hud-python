#!/bin/sh
set -eu
test "$(cat /app/result)" = nested
echo 1 > /logs/verifier/reward.txt
