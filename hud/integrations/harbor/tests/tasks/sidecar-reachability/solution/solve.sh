#!/bin/sh
set -eu
[ -c /dev/null ]
printf discarded > /dev/null
curl -fsS --max-time 10 http://workspace:5678/ > /app/sidecar.html
curl -fsS --max-time 10 http://workspace:5679/ >/dev/null
mkdir -p /app/results
printf keep > /app/results/keep.txt
printf drop > /app/results/drop.tmp
printf original > /app/binary
mkdir -p /opt/result
printf replacement > /opt/result/new.txt
printf private > /root/agent-output.txt
case ",${NO_PROXY:-}," in
  *,main,*) ;;
  *) echo "main is absent from NO_PROXY" >&2; exit 1 ;;
esac
curl -fsS --max-time 10 http://main:8080/ > /app/main.html
protected=/media/hud/session-"keys"
[ ! -e "$protected" ]
(while :; do sleep 1; done) >/tmp/actor-background.log 2>&1 &
printf '%s\n' "$!" > /app/actor.pid
entrypoint_visible=false
for pid in $(pgrep -x python3); do
  if tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null \
    | grep -F "python3 -m http.server 8080 --directory /app" >/dev/null; then
    entrypoint_visible=true
    break
  fi
done
if [ "$entrypoint_visible" = true ]; then
  echo "the environment entrypoint process is visible in the agent process namespace" >&2
  exit 1
fi
processes=$(ps -ef)
if printf '%s\n' "$processes" | grep -F "$protected"; then
  echo "protected bridge path is visible in the process list" >&2
  exit 1
fi
