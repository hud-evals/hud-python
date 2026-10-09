#!/bin/sh
set -eu

mkdir -p /tmp/harbor-startup
printf '%s\n' "${SHARED_ENV:-missing}" > /tmp/harbor-startup/status
python3 -m http.server 8080 --directory /tmp/harbor-startup >/tmp/harbor-startup.log 2>&1 &
exec "$@"
