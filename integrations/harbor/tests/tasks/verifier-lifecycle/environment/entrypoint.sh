#!/bin/sh
set -eu

mkdir -p /tmp/verifier-fixture
printf 'ready\n' > /tmp/verifier-fixture/status
python3 -m http.server 9100 --directory /tmp/verifier-fixture >/tmp/verifier-fixture.log 2>&1 &
exec "$@"
