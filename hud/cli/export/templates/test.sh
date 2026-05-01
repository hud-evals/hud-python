#!/bin/bash
# Generic Harbor verifier for HUD-exported tasks.
#
# By the time this runs, the image's ENTRYPOINT (entrypoint.py) has
# already:
#   - booted ``hud dev`` on $HUD_MCP_PORT,
#   - registered a scenario session against it (running ``setup_task``
#     once),
#   - persisted the session id to /tmp/.hud_scenario_session, and
#   - spawned a background keepalive that pings the session so it
#     survives the agent's run.
#
# So all we need is ``hud scenario grade`` — which loads the saved
# session id, calls ``_hud_submit``, reads the grader resource, and
# returns the reward. No need to call ``setup`` again here, which
# would re-run the env's destructive ``setup_task`` and wipe the
# agent's edits.
#
# Per-task config from environment variables:
#   HUD_TASK_SCENARIO  — full scenario name (env:scenario)
#   ANSWER             — optional override; if unset we extract the
#                        agent's final response from
#                        /logs/agent/claude-code.txt (Harbor's
#                        claude-code wrapper writes the stream-json
#                        there). Filesystem-graded scenarios ignore
#                        the answer; text-answer scenarios (rubric,
#                        regex, web-research) need it.
set -uo pipefail

REWARD_DIR=${REWARD_DIR:-/logs/verifier}
mkdir -p "$REWARD_DIR"

# Always-write helper. Harbor raises ``RewardFileNotFoundError`` if
# reward.txt is missing, which loses every diagnostic we'd otherwise
# have surfaced (env grader errors, dead sessions, etc.). With this
# trap, every exit path produces a parseable ``reward = 0`` and the
# real error message reaches stderr.
write_reward() {
    echo "$1" > "$REWARD_DIR/reward.txt"
}
trap '[ -f "$REWARD_DIR/reward.txt" ] || write_reward 0' EXIT

if [[ -z "${HUD_TASK_SCENARIO:-}" ]]; then
    echo "HUD_TASK_SCENARIO is not set" >&2
    write_reward 0
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
    write_reward 0
    exit 1
fi

# Confirm the entrypoint registered a session.
if [[ ! -s /tmp/.hud_scenario_session ]]; then
    echo "/tmp/.hud_scenario_session is missing — entrypoint scenario setup failed" >&2
    write_reward 0
    exit 1
fi

# Pull the agent's final response from Harbor's claude-code stream-json
# log if no explicit ANSWER was passed in. Harbor's claude-code wrapper
# tees ``--output-format=stream-json`` to /logs/agent/claude-code.txt
# and the final event is ``{"type":"result","subtype":"success",
# "result":"..."}``. For other Harbor agents (codex, oracle, nemo, ...)
# the file isn't there or has a different shape and we fall through to
# whatever the caller set ANSWER to (default: empty).
#
# Filesystem-graded scenarios (verilog-shaped) ignore the answer
# entirely — their grader inspects file state via MCP. Empty answer
# is fine. Text-answer scenarios (rubric/regex/web-research) need
# this bridge: claude-code is HUD-unaware and never calls
# ``_hud_submit`` itself, so without this extraction the env's grader
# would run against ``""``.
if [[ -z "${ANSWER:-}" && -f /logs/agent/claude-code.txt ]]; then
    ANSWER=$(python3 - <<'PY' 2>/dev/null
import json, sys
result = ""
last_assistant_text = ""
try:
    with open("/logs/agent/claude-code.txt", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Authoritative final answer: claude-code emits one of these
            # at end-of-turn. ``subtype == "success"`` carries the full
            # response in ``result``; on error there's no ``result`` so
            # we fall through to the assistant-text fallback below.
            if obj.get("type") == "result" and obj.get("subtype") == "success":
                value = obj.get("result")
                if isinstance(value, str) and value:
                    result = value
                break
            # Fallback: track the last assistant text block. Used when
            # the run errored before emitting a "result" event.
            if obj.get("type") == "assistant":
                msg = obj.get("message") or {}
                for item in msg.get("content") or []:
                    if item.get("type") == "text":
                        text = item.get("text")
                        if isinstance(text, str) and text:
                            last_assistant_text = text
except FileNotFoundError:
    pass
sys.stdout.write(result or last_assistant_text)
PY
)
fi
ANSWER=${ANSWER:-}

# Capture stdout AND stderr together so an env-side grader error
# actually surfaces in Harbor's logs instead of being swallowed by
# command substitution. We dropped ``set -e`` above so the failure
# path here is explicit; otherwise Harbor would just see a missing
# reward.txt and raise ``RewardFileNotFoundError``.
GRADE_OUTPUT=$(hud scenario grade "$HUD_TASK_SCENARIO" --answer "$ANSWER" --url "$HUD_MCP_URL" 2>&1)
GRADE_RC=$?
if [ "$GRADE_RC" -ne 0 ]; then
    echo "hud scenario grade failed (rc=$GRADE_RC):" >&2
    echo "$GRADE_OUTPUT" >&2
    # Dump the side-band logs so a session-termination diagnosis isn't
    # lost when the container exits. ``/logs/agent/keepalive.log`` is
    # the entrypoint's keepalive subprocess output (Harbor only mounts
    # /logs/{agent,verifier,artifacts}, NOT /logs itself, so writes
    # outside those subdirs disappear at container teardown).
    # ``/tmp/hud-dev.log`` is the env's ``hud dev`` server output
    # (visible from test.sh because Harbor runs the agent + verifier
    # as ``docker exec`` against the same container).
    for log in /logs/agent/keepalive.log /tmp/hud-keepalive.log /tmp/hud-dev.log; do
        if [ -f "$log" ]; then
            echo "" >&2
            echo "=== tail $log ===" >&2
            tail -50 "$log" >&2
        fi
    done
    write_reward 0
    exit 1
fi

# Parse with the Python that ships with `hud` — avoids depending on jq.
REWARD=$(printf '%s' "$GRADE_OUTPUT" | python3 -c \
    'import json,sys
try:
    print(float(json.loads(sys.stdin.read()).get("reward", 0)))
except Exception:
    print(0)')

write_reward "$REWARD"

awk -v r="$REWARD" 'BEGIN { exit (r+0 > 0) ? 0 : 1 }'
