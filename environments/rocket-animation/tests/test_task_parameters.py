"""Keep the stored task brief visible and consistent with the running task."""

import importlib.util
import json
from pathlib import Path

import pytest
from hud.environment.server import TaskRunner

ROOT = Path(__file__).parents[1]


@pytest.fixture
def rocket(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path))
    monkeypatch.syspath_prepend(str(ROOT))
    spec = importlib.util.spec_from_file_location("rocket_parameters_env", ROOT / "env.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_exposes_optional_nonempty_prompt(rocket):
    schema = rocket.env.tasks["rocket-launch-and-failure"].manifest_entry()["args"]
    prompt = schema["properties"]["prompt"]
    assert prompt["type"] == "string"
    assert prompt["minLength"] == 1
    assert prompt["x-hud-hint"] == "prompt"
    assert prompt["default"] == (ROOT / "starter/README.md").read_text()
    assert "prompt" not in schema.get("required", [])


@pytest.mark.parametrize("custom_prompt", [None, "Build the rocket animation.\nKeep its motion deterministic.\n"])
async def test_task_start_uses_one_brief_for_agent_and_workspace(rocket, custom_prompt):
    args = {} if custom_prompt is None else {"prompt": custom_prompt}
    expected = rocket.DEFAULT_PROMPT if custom_prompt is None else custom_prompt
    runner = TaskRunner(rocket.env.tasks["rocket-launch-and-failure"], args)
    try:
        assert await runner.start() == {"prompt": expected}
        assert (rocket.WORKSPACE / "README.md").read_text() == expected
        assert (rocket.WORKSPACE / "scene.js").read_text() == (ROOT / "starter/scene.js").read_text()
    finally:
        await runner.cancel()


def test_stored_task_explicitly_contains_canonical_prompt():
    task = json.loads((ROOT / "tasks.json").read_text())[0]
    assert task["args"] == {"prompt": (ROOT / "starter/README.md").read_text()}
    assert "system_prompt" not in task
