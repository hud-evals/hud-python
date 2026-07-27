"""In-process wiring smoke tests (no Docker): the served surface is well-formed."""

import json
from pathlib import Path

from env import env

FIXTURE_INSTANCE = json.loads((Path(__file__).parent / "fixtures" / "instance" / "instance.json").read_text("utf-8"))


def test_env_identity():
    assert env.name == "coding"


def test_generic_template_registered():
    assert "coding-task" in env.tasks
    assert env.tasks["coding-task"].manifest_entry()["id"] == "coding-task"


def test_swe_bench_template_registered_when_instance_baked():
    """conftest points INSTANCE_DIR at the fixture instance, as an instance image would."""
    instance_id = FIXTURE_INSTANCE["instance_id"]
    assert instance_id in env.tasks
    assert env.tasks[instance_id].manifest_entry()["id"] == instance_id
