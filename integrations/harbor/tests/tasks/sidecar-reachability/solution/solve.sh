#!/bin/sh
set -eu
curl -fsS --max-time 10 http://web:5678/ > /app/sidecar.html
