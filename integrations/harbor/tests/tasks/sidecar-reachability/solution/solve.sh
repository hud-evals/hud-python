#!/bin/sh
set -eu
[ -c /dev/null ]
printf discarded > /dev/null
curl -fsS --max-time 10 http://web:5678/ > /app/sidecar.html
case ",${NO_PROXY:-}," in
  *,main,*) ;;
  *) echo "main is absent from NO_PROXY" >&2; exit 1 ;;
esac
curl -fsS --max-time 10 http://main:8080/ > /app/main.html
protected=/media/hud/session-"keys"
[ ! -e "$protected" ]
processes=$(ps -ef)
if printf '%s\n' "$processes" | grep -F "$protected"; then
  echo "protected bridge path is visible in the process list" >&2
  exit 1
fi
