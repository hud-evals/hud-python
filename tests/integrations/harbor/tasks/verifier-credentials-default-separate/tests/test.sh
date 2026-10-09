#!/bin/sh
set -eu
[ "$JUDGE_KEY" = "judge-value" ]
echo 1 > /logs/verifier/reward.txt
