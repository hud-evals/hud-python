"""Bind the portable Larkspur examples to this environment."""

from pathlib import Path

from hud import Taskset

from env import env

tasks = []
for _row in Taskset.from_file(Path(__file__).with_name("tasks.json")):
    _task = env.templates[_row.id](**_row.args)
    _task.slug = _row.slug
    _task.columns = _row.columns
    _task.agent_config = _row.agent_config
    tasks.append(_task)

__all__ = ["env", "tasks"]
