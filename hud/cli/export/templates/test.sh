#!/bin/bash
# Generic Harbor verifier for HUD-exported tasks.
#
# By the time this runs, the image's ENTRYPOINT (entrypoint.py) has
# already:
#   - booted ``hud dev`` on $HUD_MCP_PORT,
#   - registered a scenario session against it (running ``setup_task``
#     once), and
#   - persisted the session id to /tmp/.hud_scenario_session.
#
# So all we need is ``hud scenario grade`` — which loads the saved
# session id, calls ``_hud_submit``, reads the grader resource, and
# returns the reward. No need to call ``setup`` again here, which
# would re-run the env's destructive ``setup_task`` and wipe the
# agent's edits.
#
# Per-task config from environment variables:
#   HUD_TASK_SCENARIO  — full scenario name (env:scenario)
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

# Confirm hud dev is up. If not (entrypoint.py died, or this image was
# built without one), surface a clear error rather than letting hud
# scenario grade fail with an obscure connect error.
if ! (echo > "/dev/tcp/127.0.0.1/$HUD_MCP_PORT") 2>/dev/null; then
    echo "hud dev is not running on :$HUD_MCP_PORT — entrypoint may have failed" >&2
    cat /tmp/hud-dev.log >&2 2>/dev/null || true
    echo 0 > "$REWARD_DIR/reward.txt"
    exit 1
fi

# Confirm the entrypoint registered a session.
if [[ ! -s /tmp/.hud_scenario_session ]]; then
    echo "/tmp/.hud_scenario_session is missing — entrypoint scenario setup failed" >&2
    echo 0 > "$REWARD_DIR/reward.txt"
    exit 1
fi

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
