#!/bin/bash
set -u
mkdir -p /logs/verifier
fail() { echo "$1"; echo 0 > /logs/verifier/reward.txt; exit 0; }

expected="outside_host=blocked
channel=unreachable
tests=absent
verdict=absent
harness=absent"

[ "${IMAGE_ONLY:-}" = "present" ] || fail "the verifier lost the image environment"
[ "${SHARED_ENV:-}" = "baseline" ] || fail "the verifier lost the environment configuration"
[ "${VERIFIER_ONLY:-}" = "present" ] || fail "the verifier lost its phase configuration"
[ "${PHASE:-}" = "verifier" ] || fail "the verifier did not override the baseline environment"
[ -z "${AGENT_ONLY+x}" ] || fail "the verifier inherited the agent environment"
[ "$(cat /app/startup.txt 2>/dev/null)" = "baseline" ] \
  || fail "the image entrypoint service was not available to the agent"
[ -s /app/page.html ] || fail "the agent could not fetch its allowed host"
[ "$(cat /app/probe.txt 2>/dev/null)" = "$expected" ] \
  || fail "the agent crossed a harness or network boundary"
curl -sf --max-time 30 -o /dev/null https://pypi.org/simple/ \
  || fail "the verifier inherited the agent's allowlist"
sleep 1
[ ! -e /app/stole-verifier-egress ] \
  || fail "the agent used the verifier's egress route"

echo 1 > /logs/verifier/reward.txt
sleep 1
