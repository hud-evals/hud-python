#!/bin/bash
set -u
mkdir -p /logs/verifier
fail() { echo "$1"; echo 0 > /logs/verifier/reward.txt; exit 0; }

printf '127.0.0.1 verifier-added\n' >> /etc/hosts \
  || fail "the verifier could not update its hosts file"
grep -q 'verifier-added' /etc/hosts \
  || fail "the verifier hosts update was not visible"
chown -R 1000:2000 /app/data || fail "the verifier could not chown the graded tree"
tar -cf /tmp/data.tar -C /app data || fail "the verifier could not archive the graded tree"
[ "$(cat /app/data/payload.txt 2>/dev/null)" = "hello" ] \
  || fail "the payload is incorrect"
[ "$(curl -fsS --max-time 5 http://127.0.0.1:9100/status)" = "ready" ] \
  || fail "the no-network verifier lost task-local loopback"
if curl -sf --max-time 5 -o /dev/null http://example.com/ 2>/dev/null; then
  fail "the no-network verifier reached the internet"
fi

# This child inherits stdout and keeps it open after the verifier exits.
python3 -m http.server 9101 --directory /app &
sleep 1
echo 1 > /logs/verifier/reward.txt
exit 0
