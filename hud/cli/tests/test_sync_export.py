"""Focused behavior for ``hud sync`` commands."""

from __future__ import annotations

import builtins
from typing import TYPE_CHECKING

import pytest
import typer

import hud.cli.sync as sync_module
from hud.cli.sync import _write_csv
from hud.cli.utils.registry import RegistryEnvironment
from hud.eval import Task, Taskset
from hud.utils.hud_console import HUDConsole
from hud.utils.platform import PlatformClient

if TYPE_CHECKING:
    from pathlib import Path


def test_write_csv_flattens_args(tmp_path: Path) -> None:
    rows = [
        Task(env="e", id="solve", args={"n": 1}, slug="one"),
        Task(env="e", id="solve", args={"n": {"x": 2}}, slug="two"),
    ]
    rows = [row.model_dump() for row in rows]

    out = tmp_path / "tasks.csv"
    _write_csv(out, rows)

    csv_text = out.read_text()
    assert "slug,id,env,arg:n" in csv_text
    assert "one,solve,e,1" in csv_text
    assert 'two,solve,e,"{""x"": 2}"' in csv_text


def test_sync_env_closed_stdin_aborts_cleanly(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(sync_module, "require_api_key", lambda _: None)
    monkeypatch.setattr(sync_module.PlatformClient, "from_settings", lambda: object())
    monkeypatch.setattr(
        sync_module,
        "list_registry_environments",
        lambda _: [RegistryEnvironment(id="env-1", name="example")],
    )

    def closed_input(_: str) -> str:
        raise OSError("stdin is closed")

    monkeypatch.setattr(builtins, "input", closed_input)

    with pytest.raises(typer.Exit) as exc_info:
        sync_module.sync_env_command(name=None, directory=str(tmp_path), yes=False)

    assert exc_info.value.exit_code == 0
    assert "Aborted." in capsys.readouterr().err


def test_task_sync_rejects_invalid_source(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "broken"\nlicense = {file = "LICENSE"}\n',
        encoding="utf-8",
    )

    with pytest.raises(typer.Exit):
        sync_module._validate_source(str(tmp_path), HUDConsole())

    output = capsys.readouterr().err
    assert "Source validation failed" in output
    assert "License file not found" in output


def test_task_sync_validates_deployed_manifest_before_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    taskset = Taskset("checks", [Task(env="demo", id="solve")])
    deployed = RegistryEnvironment(
        id="env-1",
        name="demo",
        manifest={"tasks": [{"id": "solve"}]},
    )
    summary = RegistryEnvironment(id=deployed.id, name=deployed.name)
    monkeypatch.setattr(sync_module, "resolve_registry_environments", lambda *_: [summary])
    monkeypatch.setattr(sync_module, "get_registry_environment", lambda *_: deployed)

    sync_module._validate_task_manifests(
        taskset,
        PlatformClient("https://api.example", "key"),
        HUDConsole(),
    )


def test_task_sync_normalizes_environment_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    taskset = Taskset("checks", [Task(env="Code Name", id="solve")])
    deployed = RegistryEnvironment(
        id="env-1",
        name="code-name",
        manifest={"tasks": [{"id": "solve"}]},
    )

    def resolve_registry(_platform: object, ref: str) -> list[RegistryEnvironment]:
        assert ref == "code-name"
        return [RegistryEnvironment(id=deployed.id, name=deployed.name)]

    monkeypatch.setattr(sync_module, "resolve_registry_environments", resolve_registry)
    monkeypatch.setattr(sync_module, "get_registry_environment", lambda *_: deployed)

    sync_module._validate_task_manifests(
        taskset,
        PlatformClient("https://api.example", "key"),
        HUDConsole(),
    )


def test_task_sync_accepts_a_registry_environment_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_id = "11111111-1111-1111-1111-111111111111"
    taskset = Taskset("checks", [Task(env=registry_id, id="solve")])
    deployed = RegistryEnvironment(
        id=registry_id,
        name="demo",
        manifest={"tasks": [{"id": "solve"}]},
    )
    monkeypatch.setattr(sync_module, "get_registry_environment", lambda *_: deployed)

    sync_module._validate_task_manifests(
        taskset,
        PlatformClient("https://api.example", "key"),
        HUDConsole(),
    )


def test_task_sync_rejects_unknown_task_before_upload(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    taskset = Taskset("checks", [Task(env="demo", id="missing")])
    deployed = RegistryEnvironment(
        id="env-1",
        name="demo",
        manifest={"tasks": [{"id": "solve"}]},
    )
    summary = RegistryEnvironment(id=deployed.id, name=deployed.name)
    monkeypatch.setattr(sync_module, "resolve_registry_environments", lambda *_: [summary])
    monkeypatch.setattr(sync_module, "get_registry_environment", lambda *_: deployed)

    with pytest.raises(typer.Exit):
        sync_module._validate_task_manifests(
            taskset,
            PlatformClient("https://api.example", "key"),
            HUDConsole(),
        )

    assert "does not expose task(s): missing" in capsys.readouterr().err


def test_task_sync_validates_verifier_task_ids(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    taskset = Taskset(
        "checks",
        [
            Task(
                env="actor",
                id="solve",
                verifier=Task(env="judge", id="verify"),
            )
        ],
    )
    deployed = {
        "env-actor": RegistryEnvironment(
            id="env-actor",
            name="actor",
            manifest={"tasks": [{"id": "solve"}]},
        ),
        "env-judge": RegistryEnvironment(
            id="env-judge",
            name="judge",
            manifest={"tasks": [{"id": "other"}]},
        ),
    }
    monkeypatch.setattr(
        sync_module,
        "resolve_registry_environments",
        lambda _, name: [RegistryEnvironment(id=f"env-{name}", name=name)],
    )
    monkeypatch.setattr(
        sync_module,
        "get_registry_environment",
        lambda _, registry_id: deployed[registry_id],
    )

    with pytest.raises(typer.Exit):
        sync_module._validate_task_manifests(
            taskset,
            PlatformClient("https://api.example", "key"),
            HUDConsole(),
        )

    assert "environment 'judge' does not expose task(s): verify" in capsys.readouterr().err
