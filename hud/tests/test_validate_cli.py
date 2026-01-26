from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import typer


def _load_validate_command():
    module_path = Path(__file__).resolve().parents[1] / "cli" / "validate.py"
    spec = importlib.util.spec_from_file_location("hud.cli.validate", module_path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.validate_command


def _write_tasks(path: Path, tasks: list[dict]) -> str:
    path.write_text(json.dumps(tasks), encoding="utf-8")
    return str(path)


def test_validate_command_valid(tmp_path: Path) -> None:
    validate_command = _load_validate_command()
    tasks = [
        {
            "prompt": "Say hello",
            "mcp_config": {"local": {"command": "echo", "args": ["hi"]}},
            "evaluate_tool": {"name": "done", "arguments": {}},
        }
    ]
    path = _write_tasks(tmp_path / "tasks.json", tasks)
    validate_command(path)


def test_validate_command_invalid(tmp_path: Path) -> None:
    validate_command = _load_validate_command()
    tasks = [{"mcp_config": {"local": {"command": "echo", "args": ["hi"]}}}]
    path = _write_tasks(tmp_path / "tasks.json", tasks)
    with pytest.raises(typer.Exit):
        validate_command(path)
