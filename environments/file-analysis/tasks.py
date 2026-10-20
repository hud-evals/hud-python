"""Bind the editable task data to this environment for local runs and syncing."""

import json
from pathlib import Path

from env import env

# The same JSON supplies the console and this runnable task list. Calling the
# local template preserves binding when hud init gives the environment a name.
tasks = []
for _row in json.loads(Path(__file__).with_suffix(".json").read_text()):
    _task = env.templates[_row["id"]](**_row["args"])
    _task.slug = _row["slug"]
    _task.agent_config = _row["agent_config"]
    tasks.append(_task)

__all__ = ["env", "tasks"]
