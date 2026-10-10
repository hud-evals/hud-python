#!/bin/sh
set -eu
printf '%s:%s\n' "$(id -u)" "$(id -g)" > /app/identity.txt
printf '%s\n' "${IMAGE_ONLY:-missing}" > /app/image-env.txt
python3 -m http.server 9000 --directory /app >/dev/null 2>&1 &
sleep 2
