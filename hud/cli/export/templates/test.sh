#!/bin/bash
# Generic Harbor verifier for HUD-exported tasks.
# Boots the HUD MCP server inside the running container, registers a fresh
# scenario session against the (pre-baked) filesystem state, then calls
# `hud scenario grade` and writes the reward to /logs/verifier/reward.txt.
#
# Per-task config comes from environment variables baked into the image:
#   HUD_TASK_SCENARIO  — full scenario name (env:scenario)
#   HUD_TASK_ARGS      — JSON args (may be "{}")
set -euo pipefail

REWARD_DIR=${REWARD_DIR:-/logs/verifier}
mkdir -p "$REWARD_DIR"

if [[ -z "${HUD_TASK_SCENARIO:-}" ]]; then
    echo "HUD_TASK_SCENARIO is not set" >&2
    echo 0 > "$REWARD_DIR/reward.txt"
    exit 1
fi

HUD_MCP_PORT=${HUD_MCP_PORT:-8765}
HUD_MCP_URL=${HUD_MCP_URL:-http://localhost:$HUD_MCP_PORT/mcp}
HUD_DEV_ARGS=${HUD_DEV_ARGS:-env:env}

# Bash parameter expansion stops at the first '}', so '${X:-{}}' doesn't
# default to '{}'. Use an explicit fallback instead.
ARGS=${HUD_TASK_ARGS-}
[[ -z "$ARGS" ]] && ARGS='{}'

hud dev $HUD_DEV_ARGS --port "$HUD_MCP_PORT" &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 30); do
    if (echo > "/dev/tcp/127.0.0.1/$HUD_MCP_PORT") 2>/dev/null; then
        break
    fi
    sleep 1
done

# Re-register session against the baked filesystem state.
# Setup must be idempotent on already-set-up state.
hud scenario setup "$HUD_TASK_SCENARIO" --args "$ARGS" --url "$HUD_MCP_URL" >/dev/null

RESULT=$(hud scenario grade "$HUD_TASK_SCENARIO" --answer "${ANSWER:-}" --url "$HUD_MCP_URL")
# Parse with the Python that ships with `hud` — avoids depending on jq.
REWARD=$(printf '%s' "$RESULT" | python3 -c \
    'import json,sys
try:
    print(float(json.loads(sys.stdin.read()).get("reward", 0)))
except Exception:
    print(0)')

echo "$REWARD" > "$REWARD_DIR/reward.txt"

awk -v r="$REWARD" 'BEGIN { exit (r+0 > 0) ? 0 : 1 }'
