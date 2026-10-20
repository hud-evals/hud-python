"""Bind the shared task definitions to this environment for local evaluation."""

from pathlib import Path

from hud import Taskset

from environment.env import env as env

tasks = []
for _row in Taskset.from_file(Path(__file__).with_name("tasks.json")):
    _task = env.templates[_row.id](**_row.args)
    _task.slug = _row.slug
    _task.agent_config = _row.agent_config
    tasks.append(_task)
