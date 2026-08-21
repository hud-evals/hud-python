# generated task metadata
#!/bin/bash
set -uo pipefail
mkdir -p /logs/verifier
mv /app/binary /app/binary.original
printf regenerated > /app/binary
mv /opt/result /opt/result.moved
env -i PATH=/usr/bin:/bin getent hosts pypi.org >/dev/null \
  || { echo "the public verifier could not resolve an external host"; exit 1; }

if grep -q "Directory listing" /app/sidecar.html 2>/dev/null \
  && grep -q "Directory listing" /app/main.html 2>/dev/null \
  && [ -c /dev/null ] \
  && [ "$(cat /tmp/main.txt 2>/dev/null)" = "collected-from-main" ] \
  && [ "$(cat /tmp/sidecar.txt 2>/dev/null)" = "collected-from-sidecar" ] \
  && [ "$(cat /app/results/keep.txt 2>/dev/null)" = "keep" ] \
  && [ ! -e /app/results/drop.tmp ] \
  && [ "$(cat /app/binary.original 2>/dev/null)" = "original" ] \
  && [ "$(cat /app/binary 2>/dev/null)" = "regenerated" ] \
  && [ "$(cat /opt/result.moved/new.txt 2>/dev/null)" = "replacement" ] \
  && [ ! -e /opt/result.moved/base.txt ] \
  && [ "$(cat /root/agent-output.txt 2>/dev/null)" = "private" ] \
  && [ "$(id -u)" = "1001" ] \
  && [ "$(stat -c %u /home/verifier/owned)" = "1001" ] \
  && [ "$PWD" = "/home/verifier" ] \
  && [ "$HOME" = "/home/verifier" ] \
  && [ "$VERIFIER_PRECEDENCE" = "verifier-image" ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo "the agent output or collected sidecar artifact is missing"
  echo 0 > /logs/verifier/reward.txt
fi
