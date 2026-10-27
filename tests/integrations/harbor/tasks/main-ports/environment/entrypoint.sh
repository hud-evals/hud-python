#!/bin/sh
set -eu
python3 -m http.server 8765 --directory /app >/tmp/inner.log 2>&1 &
python3 -m http.server 8080 --directory /app >/tmp/app.log 2>&1 &
exec "$@"
