#!/bin/sh
set -u
mkdir -p /logs/verifier
fail() { echo "$1"; echo 0 > /logs/verifier/reward.txt; exit 0; }
[ ! -e /opt/stray.txt ] || fail "a previous verifier's file survived"
[ ! -e /opt/result.moved ] || fail "a previous verifier's move survived"
[ "$(ls /opt/result)" = "agent.txt" ] || fail "the artifact did not replace the image path"
echo stray > /opt/stray.txt
mv /opt/result /opt/result.moved
echo 1 > /logs/verifier/reward.txt
