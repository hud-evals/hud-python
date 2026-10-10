#!/bin/sh
set -eu
test "$(cat /workspace/result)" = managed-codex
echo 1 > /logs/verifier/reward.txt
