#!/bin/bash
# Build and run one task from this Harbor-format export.
#   ./sample-run.sh                # run first task
#   ./sample-run.sh <task-slug>    # run a specific task
set -euo pipefail

EXPORT_ROOT=$(cd "$(dirname "$0")" && pwd)
TASK=${1:-__FIRST_TASK__}

if [[ ! -d "$EXPORT_ROOT/tasks/$TASK" ]]; then
    echo "Task not found: $TASK" >&2
    echo "Available:" >&2
    ls "$EXPORT_ROOT/tasks" >&2
    exit 1
fi

IMG="hud-export-task:$TASK"
docker build -t "$IMG" "$EXPORT_ROOT/tasks/$TASK/environment"

LOG_DIR="$EXPORT_ROOT/logs/$TASK"
mkdir -p "$LOG_DIR"

set +e
docker run --rm \
    -v "$EXPORT_ROOT/tasks/$TASK/tests:/tests:ro" \
    -v "$LOG_DIR:/logs" \
    "$IMG" \
    bash /tests/test.sh
RUN_RC=$?
set -e

REWARD_FILE="$LOG_DIR/verifier/reward.txt"
echo
if [[ -f "$REWARD_FILE" ]]; then
    echo "Reward: $(cat "$REWARD_FILE")  (test.sh exit=$RUN_RC)"
else
    echo "No reward.txt produced (test.sh exit=$RUN_RC). See container logs above."
fi
exit "$RUN_RC"
