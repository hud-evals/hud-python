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

export HUD_MCP_PORT=${HUD_MCP_PORT:-8765}
HUD_MCP_URL=${HUD_MCP_URL:-http://localhost:$HUD_MCP_PORT/mcp}

# Bash parameter expansion stops at the first '}', so '${X:-{}}' doesn't
# default to '{}'. Use an explicit fallback instead.
ARGS=${HUD_TASK_ARGS-}
[[ -z "$ARGS" ]] && ARGS='{}'

# Render the env's launch command. ``HUD_LAUNCH_COMMAND`` (JSON list)
# is the env image's full original CMD with ``--stdio`` rewritten to
# ``--port`` — required for envs that wrap ``hud dev`` in a multi-
# process startup (e.g. ``sh -c "uvicorn sidecars... & exec hud dev
# env:env --stdio"``); skip the prefix and the sidecars never run.
# Falls back to a bare ``hud dev env:env`` when not set.
LAUNCH_SHELL=$(python3 - <<'PY'
import json, os, shlex
raw = os.environ.get("HUD_LAUNCH_COMMAND") or ""
port = os.environ["HUD_MCP_PORT"]
if raw:
    cmd = json.loads(raw)
    out = []
    for arg in cmd:
        if arg == "--stdio":
            out.extend(["--port", port])
        elif "--stdio" in arg:
            out.append(arg.replace("--stdio", f"--port {port}"))
        else:
            out.append(str(arg))
    print(shlex.join(out))
else:
    print(f"hud dev env:env --port {shlex.quote(port)}")
PY
)

# ``setsid`` puts the launch tree in its own process group so we can
# kill the whole tree on EXIT — envs that fork sidecars (uvicorn,
# Xvfb, chromium, etc.) leave orphans behind a plain ``kill $!`` and
# leak zombie processes into the next docker layer.
setsid bash -c "$LAUNCH_SHELL" &
SERVER_PID=$!
trap 'kill -- -"$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT

# Wait for the MCP port to start accepting connections. Bumped from 60s
# because some envs do significant pre-MCP setup (uvicorn boot, Xvfb,
# chrome warmup) that easily exceeds 60s on a cold build cache.
for _ in $(seq 1 240); do
    if (echo > "/dev/tcp/127.0.0.1/$HUD_MCP_PORT") 2>/dev/null; then
        break
    fi
    sleep 1
done

hud scenario setup "$HUD_TASK_SCENARIO" --args "$ARGS" --url "$HUD_MCP_URL"
