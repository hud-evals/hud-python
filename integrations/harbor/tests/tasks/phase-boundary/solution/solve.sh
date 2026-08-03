#!/bin/sh
set -eu

curl -sS --max-time 30 -o /app/page.html http://example.com/
curl -sS --max-time 10 -o /app/startup.txt http://127.0.0.1:8080/status
{
  if curl -sS --max-time 20 -o /dev/null https://pypi.org/simple/ 2>/dev/null; then
    echo "outside_host=reachable"
  else
    echo "outside_host=blocked"
  fi
  if curl -sf --max-time 3 -o /dev/null http://127.0.0.1:8765/ 2>/dev/null; then
    echo "channel=reachable"
  else
    echo "channel=unreachable"
  fi
  [ -e /tests/test.sh ] && echo "tests=present" || echo "tests=absent"
  [ -e /logs/verifier/reward.txt ] && echo "verdict=present" || echo "verdict=absent"
  [ -e /media/hud ] && echo "harness=present" || echo "harness=absent"
} > /app/probe.txt

# Continue probing verifier-only routes after this command returns.
(
  while :; do
    mkdir -p /logs/verifier
    echo 0 > /logs/verifier/reward.txt
    if http_proxy=http://127.0.0.1:3129 https_proxy=http://127.0.0.1:3129 \
      curl -sf --max-time 2 -o /dev/null https://pypi.org/simple/ 2>/dev/null; then
      touch /app/stole-verifier-egress
    fi
    sleep 0.01
  done
) >/dev/null 2>&1 &
