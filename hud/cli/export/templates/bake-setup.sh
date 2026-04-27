#!/bin/bash
# Image-build-time setup baker for HUD-exported tasks.
# Runs during `docker build` of a per-task environment Dockerfile:
#   1. Boots the HUD MCP server in the background (HTTP transport).
#   2. Calls `hud scenario setup` to perform task-specific setup.
#   3. Stops the server. The mutated filesystem is committed by Docker
#      as the next image layer, so the agent gets a pre-baked env.
set -euo pipefail

if [[ -z "${HUD_TASK_SCENARIO:-}" ]]; then
    echo "HUD_TASK_SCENARIO not set; nothing to bake" >&2
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
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT

# Wait for the MCP port to start accepting connections.
for _ in $(seq 1 60); do
    if (echo > "/dev/tcp/127.0.0.1/$HUD_MCP_PORT") 2>/dev/null; then
        break
    fi
    sleep 1
done

hud scenario setup "$HUD_TASK_SCENARIO" --args "$ARGS" --url "$HUD_MCP_URL"
